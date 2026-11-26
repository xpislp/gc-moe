"""Small-M direct peer gather into CTA-local SMEM, then K128 MMA staging.

Gather a complete input tile once. Waiting after every remote K128 copy would
serialize network latency across the K loop. The cache avoids that wait chain;
weight TMA and MMA still use the ordinary multistage K128 pipeline. No routed
input activation or input scale is written to a local global-memory ring.
"""
import cutlass
import cutlass.cute as cute
from cutlass import Uint8, Uint32, Uint64
from ..layout.sym_buffer import mapped_peer_base
from ..ptx.tma import bulk_g2s_1d, tma_store_fence
from ..ptx.ld_st import EVICT_LAST, ld_shared_u32x4, st_shared_u32x4


@cute.jit
def cache_peer_tile(signals: cute.Tensor, peers: cute.Tensor,
                    cache_x: cute.Pointer, cache_sf: cute.Pointer,
                    bar: cute.Pointer, phase: Uint32, pool_block: Uint32,
                    valid_m: Uint32, cta_rank: Uint32, lane: Uint32,
                    bm: cutlass.Constexpr, hidden: cutlass.Constexpr,
                    ranks: cutlass.Constexpr, metadata_words: cutlass.Constexpr,
                    x_words: cutlass.Constexpr, sf_words: cutlass.Constexpr):
    half = ((valid_m + Uint32(15)) & Uint32(~15)) // Uint32(2)
    start = cta_rank * half
    # Each active lane owns one logical token, including its complete K32 SF
    # row. The two CTAs duplicate SF but own disjoint activation half-tiles.
    if lane < valid_m:
        meta = metadata_words + (pool_block * Uint32(bm) + lane) * Uint32(3)
        base = mapped_peer_base(peers, Uint32(signals[meta]), ranks)
        token = Uint32(signals[meta + 1])
        sf = cute.make_ptr(Uint8, base + Uint64(sf_words * 4) +
                          Uint64(token) * Uint64(hidden // 128 * 4),
                          cute.AddressSpace.gmem, assumed_align=16)
        bulk_g2s_1d(cache_sf + lane * Uint32(hidden // 128), sf,
                     Uint32(hidden // 128 * 4), bar, cache_hint=EVICT_LAST)
        if lane >= start and lane < start + half:
            x = cute.make_ptr(Uint8, base + Uint64(x_words * 4) +
                             Uint64(token) * Uint64(hidden),
                             cute.AddressSpace.gmem, assumed_align=16)
            # One copy per row at GLM5.2 H6144. Large H is deliberately not
            # accepted by the LL contract until its cache geometry is tested.
            bulk_g2s_1d(cache_x + (lane - start) * Uint32(hidden), x,
                         Uint32(hidden), bar, cache_hint=EVICT_LAST)
    cute.arch.sync_warp()
    with cute.arch.elect_one():
        rows = min(half, valid_m - min(valid_m, start))
        cute.arch.mbarrier_arrive_and_expect_tx(
            bar, rows * Uint32(hidden) + valid_m * Uint32(hidden // 128 * 4))
    cute.arch.mbarrier_wait(bar, phase)
    cute.arch.sync_warp()


@cute.jit
def load_cached_k128(cache_x: cute.Pointer, cache_sf: cute.Pointer,
                    acts: cute.Pointer, scales: cute.Pointer,
                    full: cute.Pointer, empty: cute.Pointer,
                    kb: Uint32, phase: Uint32, valid_m: Uint32, lane: Uint32,
                    bm: cutlass.Constexpr, sf_m: cutlass.Constexpr,
                    hidden: cutlass.Constexpr):
    cute.arch.mbarrier_wait(empty, phase ^ Uint32(1))
    cta_rank, _, _ = cute.arch.block_in_cluster_idx()
    half = ((valid_m + Uint32(15)) & Uint32(~15)) // Uint32(2)
    for it in cutlass.range_constexpr(bm // 8):
        row = Uint32(it * 4) + lane // Uint32(8)
        offset = row * Uint32(128) + (lane % Uint32(8)) * Uint32(16)
        swizzled = offset ^ ((offset >> Uint32(7) & Uint32(7)) << Uint32(4))
        a, b, c, d = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
        if row < half and row + cta_rank * half < valid_m:
            ptr = cute.recast_ptr(cache_x + row * Uint32(hidden) + kb * Uint32(128) +
                                 (lane % Uint32(8)) * Uint32(16), dtype=Uint32)
            a, b, c, d = ld_shared_u32x4(ptr)
        st_shared_u32x4(cute.recast_ptr(acts + swizzled, dtype=Uint32), a, b, c, d)
    for it in cutlass.range_constexpr(sf_m // 32):
        packed_row = Uint32(it * 32) + lane
        row = (packed_row % Uint32(4)) * Uint32(32) + packed_row // Uint32(4)
        value = Uint32(0)
        if row < valid_m:
            value = Uint32(cache_sf[row * Uint32(hidden // 128) + kb])
        scales[packed_row] = value
    tma_store_fence()
    cute.arch.sync_warp()
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(full, peer_cta_rank_in_cluster=0)
