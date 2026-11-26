"""Final SMEM layouts for the small-M direct peer cache (host sizing)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class DirectPullCacheLayout:
    x: int
    sf: int
    sf_stride: int  # Uint32 words between consecutive K128 SF tiles.
    sf_words: int
    end: int


def get_direct_pull_cache_layout(shared_end: int, bm: int, hidden: int):
    if bm not in (16, 32) or hidden % 512 or not 0 < hidden <= 8192:
        raise ValueError('invalid small-M direct-pull cache geometry')
    x = (shared_end + 1023) // 1024 * 1024
    sf = x + bm // 2 * hidden
    # utccp reads 128 words, but N<=BM consumes only rows 0..BM-1.
    # Their 4x32-permuted slots fit in BM*4 words. For BM16, the
    # unused second half of one read overlaps the next K tile. The
    # final read needs an explicit guard, initialized along with padding.
    sf_stride = bm * 4
    sf_words = sf_stride * (hidden // 128) + 128 - sf_stride
    return DirectPullCacheLayout(x, sf, sf_stride, sf_words, sf + sf_words * 4)
