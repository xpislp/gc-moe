"""Peer copies land in final K128/SW128 MMA cache; no SMEM repacking."""
import cutlass
import cutlass.cute as cute
from cutlass import Uint8, Uint32, Uint64
from ..layout.sym_buffer import mapped_peer_base
from ..ptx.tma import (copy_async_16, copy_async_4, copy_commit, copy_wait,
                      copy_wait_six, tma_store_fence)
from ..ptx.ld_st import bulk_zero_shared


@cute.jit
def cache_peer_tile(signals: cute.Tensor, peers: cute.Tensor,
                    cache_x: cute.Pointer, cache_sf: cute.Pointer,
                    pool_block: Uint32,
                    valid_m: Uint32, cta_rank: Uint32, lane: Uint32,
                    bm: cutlass.Constexpr, hidden: cutlass.Constexpr,
                    ranks: cutlass.Constexpr, metadata_words: cutlass.Constexpr,
                    x_words: cutlass.Constexpr, sf_words: cutlass.Constexpr,
                    sf_stride: cutlass.Constexpr, cache_sf_words: cutlass.Constexpr):
    half = ((valid_m + Uint32(15)) & Uint32(~15)) // Uint32(2)
    start = cta_rank * half
    with cute.arch.elect_one():
        bulk_zero_shared(cache_x, Uint32(bm // 2 * hidden))
    cute.arch.sync_warp()
    tma_store_fence()
    # Only groups containing live rows issue peer copies. The entire cache
    # was zeroed once, so unused CTA halves need no per-K copy instructions.
    for rows in cutlass.range_constexpr(bm // 8):
        if Uint32(rows * 4) < min(half, valid_m - min(valid_m, start)):
            row = Uint32(rows * 4) + lane // Uint32(8)
            logical_row = start + row
            valid = row < half and logical_row < valid_m
            base = Uint64(signals.iterator.toint())
            token = Uint32(0)
            if valid:
                meta = metadata_words + (pool_block * Uint32(bm) + logical_row) * Uint32(3)
                base = mapped_peer_base(peers, Uint32(signals[meta]), ranks)
                token = Uint32(signals[meta + 1])
            offset = row * Uint32(128) + (lane % Uint32(8)) * Uint32(16)
            swizzled = offset ^ ((offset >> Uint32(7) & Uint32(7)) << Uint32(4))
            for kb in cutlass.range(hidden // 128):
                dst = cache_x + Uint32(kb) * Uint32(bm // 2 * 128) + swizzled
                src = cute.make_ptr(Uint8, base + Uint64(x_words * 4) +
                    Uint64(token) * Uint64(hidden) + Uint64(kb * 128) +
                    Uint64(lane % Uint32(8) * Uint32(16)), cute.AddressSpace.gmem,
                    assumed_align=16)
                copy_async_16(dst, src, Uint32(16) if valid else Uint32(0))
                if (kb + 1) % 8 == 0:
                    copy_commit()
                    copy_wait_six()
            if cutlass.const_expr((hidden // 128) % 8 != 0):
                copy_commit()
                copy_wait_six()
    # Zero the padding once, then gather the live SF words directly to the
    # final 4x32 layout. Four lanes coalesce adjacent K128 words of one token.
    with cute.arch.elect_one():
        bulk_zero_shared(cache_sf, Uint32(cache_sf_words * 4))
    cute.arch.sync_warp()
    tma_store_fence()
    for rows in cutlass.range_constexpr(bm // 8):
        if Uint32(rows * 8) < valid_m:
            row = Uint32(rows * 8) + lane // Uint32(4)
            packed = row % Uint32(32) * Uint32(4) + row // Uint32(32)
            base = Uint64(signals.iterator.toint())
            token = Uint32(0)
            if row < valid_m:
                meta = metadata_words + (pool_block * Uint32(bm) + row) * Uint32(3)
                base = mapped_peer_base(peers, Uint32(signals[meta]), ranks)
                token = Uint32(signals[meta + 1])
            for it in cutlass.range(hidden // 512):
                kb = Uint32(it * 4) + lane % Uint32(4)
                sf_src = cute.make_ptr(Uint32, base + Uint64(sf_words * 4) +
                    Uint64(token) * Uint64(hidden // 128 * 4) + Uint64(kb * 4),
                    cute.AddressSpace.gmem)
                copy_async_4(cache_sf + kb * Uint32(sf_stride) + packed, sf_src,
                             Uint32(4) if row < valid_m else Uint32(0))
                if (it + 1) % 4 == 0:
                    copy_commit()
                    copy_wait_six()
            if cutlass.const_expr((hidden // 512) % 4 != 0):
                copy_commit()
                copy_wait_six()
    copy_commit()
    copy_wait()
    cute.arch.sync_warp()
    tma_store_fence()

@cute.jit
def publish_cached_stage(full: cute.Pointer, empty: cute.Pointer, phase: Uint32):
    cute.arch.mbarrier_wait(empty, phase ^ Uint32(1))
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(full, peer_cta_rank_in_cluster=0)
