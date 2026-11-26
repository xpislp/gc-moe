"""Materialize compact source metadata without a warp-per-token pull loop."""
import cutlass
import cutlass.cute as cute
from cutlass import Uint32, Uint64
from ..layout.gc_moe import SIGNAL_OFFSETS
from ..layout.sym_buffer import mapped_peer_base
from ..ptx.ld_st import ld_global_u64, ld_acquire_gpu_u32, red_release_gpu_add_u32
from ..scheduler.gc_moe import cached_num_tokens


@cute.jit
def materialize_routes(signals: cute.Tensor, peers: cute.Tensor,
                       source: cute.Tensor, counts: cute.Tensor,
                       gen: Uint32, warp: Uint32, lane: Uint32,
                       experts: cutlass.Constexpr, ranks: cutlass.Constexpr,
                       capacity: cutlass.Constexpr, topk: cutlass.Constexpr,
                       bm: cutlass.Constexpr, warps: cutlass.Constexpr,
                       ring_blocks: cutlass.Constexpr, ring_tokens: cutlass.Constexpr,
                       intermediate: cutlass.Constexpr, metadata_words: cutlass.Constexpr,
                       weight_words: cutlass.Constexpr, input_weight_words: cutlass.Constexpr):
    for expert in cutlass.range(warp, Uint32(experts), warps):
        received = cached_num_tokens(counts, expert, lane, experts)
        pool_start = Uint32(0)
        for prev in cutlass.range_constexpr(experts):
            n = cached_num_tokens(counts, Uint32(prev), lane, experts)
            if Uint32(prev) < expert:
                pool_start += ((n + Uint32(bm - 1)) // Uint32(bm)) * Uint32(bm)
        prefix = Uint32(0)
        for src_rank in cutlass.range_constexpr(ranks):
            count = Uint32(ld_global_u64(signals.iterator +
                SIGNAL_OFFSETS['expert_recv_count'] // 4 +
                gen * Uint32(ranks * experts * 2) +
                (Uint32(src_rank * experts) + expert) * Uint32(2)))
            base = mapped_peer_base(peers, Uint32(src_rank), ranks)
            for slot in cutlass.range(lane, count, 32):
                index = source[(expert * Uint32(ranks) + Uint32(src_rank)) * Uint32(capacity) + slot]
                token, top_slot = index // Uint32(topk), index % Uint32(topk)
                relative = prefix + slot
                pool_token = pool_start + relative
                pool_block = pool_token // Uint32(bm)
                ring_block = pool_block % Uint32(ring_blocks)
                generation = pool_block // Uint32(ring_blocks)
                if generation > 0:
                    while ld_acquire_gpu_u32(signals.iterator +
                            SIGNAL_OFFSETS['l1_empty_count'] // 4 + ring_block) < generation * Uint32(2 * intermediate // 128):
                        pass
                weight = cute.make_ptr(Uint32, base + Uint64(input_weight_words * 4) +
                                       Uint64(index) * Uint64(4), cute.AddressSpace.gmem)
                signals[weight_words + pool_token % Uint32(ring_tokens)] = weight[0]
                signals[metadata_words + pool_token * Uint32(3)] = Uint32(src_rank)
                signals[metadata_words + pool_token * Uint32(3) + 1] = token
                signals[metadata_words + pool_token * Uint32(3) + 2] = top_slot
                increment = (Uint32(bm) - relative % Uint32(bm)
                             if relative + Uint32(1) == received else Uint32(1))
                red_release_gpu_add_u32(signals.iterator +
                    SIGNAL_OFFSETS['l1_full_count'] // 4 + ring_block, increment)
            prefix += count
