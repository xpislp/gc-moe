"""Small-batch SM100 MoE with expert-owned routing and optional direct gather."""
from dataclasses import replace
from .gc_moe import validate_call


def get_block_m_for_gc_moe_ll(num_ranks, num_experts, capacity, tokens, topk,
                              *, direct_pull=False):
    """Use the same direct_pull flag here and on the operator when staging SE SF."""
    if (not 1 <= num_ranks <= 32 or num_experts <= 0 or
            num_experts % num_ranks or not 1 <= topk <= 32):
        raise ValueError('LL needs EP 1..32, divisible experts and topk 1..32')
    if not 0 <= tokens <= min(capacity, 32):
        raise ValueError("LL supports 0 <= T <= 32 within capacity")
    threshold = 32 if direct_pull else 16
    return 16 if tokens * num_ranks * topk <= threshold * num_experts else 32



def fp8_fp4_gc_moe_ll(y, l1_weights, l2_weights, sym_buffer,
                      shared_l1_weights=None, shared_l2_weights=None,
                      cumulative_local_expert_recv_stats=None,
                      recipe=(1, 1, 32), activation='swiglu',
                      activation_clamp=None, fast_math=True,
                      shared_recipe=None, activation_alpha=None, *, direct_pull=False):
    """Checked LL entry point using the existing GC buffer/weight contract.

    T <= 32 and EP <= 32. Calls sharing a buffer must be serialized on every
    rank. Shared input SF must use get_block_m_for_gc_moe_ll (the LL tile
    choice differs from the throughput API). Pass direct_pull consistently
    to both the tile query and this operator, especially when staging SE SF.

    The default uses expert-owned one-pass routing and concurrent primary-
    context dispatch/compute kernels. Up to eight route CTAs gather each X
    once into the activation ring. direct_pull=True instead uses one metadata
    CTA and an MMA-ready compute-CTA SMEM input cache, with consecutive W1
    N tiles sharing that cache. For H >= 1024, K128 groups are prefetched
    while MMA consumes earlier groups; peer addresses are resolved once per
    cache miss. There is no second SMEM rearrangement. Direct pull remains
    experimental and slower than the default LL path on the tested EP4
    GLM5.2 shapes; it is never selected implicitly.
    """
    config = validate_call(y, l1_weights, l2_weights, sym_buffer,
                           shared_l1_weights, shared_l2_weights,
                           cumulative_local_expert_recv_stats, recipe, activation,
                           activation_clamp, fast_math, shared_recipe, activation_alpha)
    if not 0 <= config.tokens <= 32 or not 1 <= sym_buffer.group.size() <= 32:
        raise ValueError('LL supports T/rank <= 32 and EP <= 32')
    if direct_pull and (config.hidden % 512 or config.hidden > 8192):
        raise ValueError('direct pull requires H divisible by 512 and H <= 8192')
    config = replace(config, block_m=get_block_m_for_gc_moe_ll(
        sym_buffer.group.size(), sym_buffer.num_experts,
        sym_buffer.num_max_tokens_per_rank, config.tokens, sym_buffer.num_topk,
        direct_pull=direct_pull))
    from ..jit_kernels.impls.sm100_fp8_fp4_gc_moe import launch_gc_moe_ll
    launch_gc_moe_ll(config, y, l1_weights, l2_weights,
                     sym_buffer._for_call(config.shared_experts),
                     shared_l1_weights, shared_l2_weights,
                     cumulative_local_expert_recv_stats, direct_pull=direct_pull)
