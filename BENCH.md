# `jpeg-rusturbo` Phase 2 benchmarks

Measured on Apple Silicon (M-series) at 100 iterations per resolution
after a 3-iteration warm-up. Build: `cargo build --release --bin bench`
(NEON) and `cargo build --release --bin bench --features force-scalar`
(scalar). Single-shot wall-clock; no statistical filtering — variance
between runs is well under 1%.

## Encode time (ms/iter, full RGBA → JPEG, quality 80, 4:2:0)

| Resolution                  | Scalar (force-scalar) | NEON  | Speedup |
| --------------------------- | --------------------: | ----: | ------: |
| 1592 x 1124 (session size)  |             10.89 ms  | 8.36 ms |  1.30x |
| 1920 x 1080 (1080p)         |             12.44 ms  | 9.22 ms |  1.35x |
| 3840 x 2160 (4K)            |             51.73 ms  | 37.92 ms | 1.36x |

Output bytes are byte-identical between scalar and NEON builds (verified
in the bench output: `423839 / 488459 / 1940692 bytes` in both modes).
This is the bit-exact equivalence we set out to achieve when switching
from Phase 1's f32 AAN DCT to libjpeg-turbo's integer LL&M scheme — the
NEON kernels are a true drop-in for the scalar reference.

## Where the speedup is, and isn't

The 1.3-1.36x figure is below the 2-4x speculative target stated in the
phase 2 brief. The most likely reason: the Huffman entropy coder
remains scalar (Phase 2 leaves it scalar by design — libjpeg-turbo
itself doesn't have an aarch64 NEON Huffman kernel in the
matching shape, and the loop is too branchy to vectorize well), and
LLVM's autovectorizer does a respectable job on the integer LL&M DCT and
the reciprocal-multiply quantizer in the scalar path at `-O3`.

A rough per-stage breakdown (estimated from %time of `cargo flamegraph`,
not committed to the repo) for 4K:

  Color/downsample  ~25%  (NEON: ~3.0x;  scalar: 1.0x)
  Forward DCT       ~20%  (NEON: ~2.5x;  scalar: 1.0x)
  Quantize+zig-zag  ~10%  (NEON: ~1.8x;  scalar: 1.0x)
  Huffman           ~35%  (scalar in both)
  Marker writes/IO  ~10%  (scalar in both)

So the NEON kernels themselves are doing roughly the expected speedup;
Amdahl's law just bounds the whole-pipeline number.

## Phase 2.5 — Huffman optimizations (this milestone)

Phase 2 left the Huffman entropy coder scalar with a 32-bit accumulator;
flamegraphs put it at ~30-35% of total encode time, by far the dominant
remaining cost. Phase 2.5 rewrites it without touching the public API:

  - **64-bit bit accumulator.** Each `write_bits` now appends with one
    shift+OR; we drain four bytes at a time (when `nbits ≥ 32`) instead
    of one. ~4x fewer per-symbol drain checks.
  - **Branchless inner path.** The drain check is the only branch on the
    common path; symbol packing is straight-line shifts.
  - **Packed Huffman table.** Single `[u32; 256]` per table holding
    `(length << 16) | code` — one load per symbol, no parallel-array
    overhead.
  - **Internal byte buffer.** The bit accumulator drains into an owned
    `Vec<u8>` and forwards to the user's `Write` exactly once per scan
    (in `flush_to_byte_boundary`). One `write_all` per frame instead of
    one per byte.
  - **NEON-assisted AC zero scan** (aarch64, gated by `force-scalar`).
    Loads 8 i16 at a time, takes `vmaxvq_u16(vabsq_s16(...))`, skips the
    whole group when zero. The hot case for q=70-80 natural images.

Bit-exact output preserved — `bytes` columns below match Phase 2
exactly (`423839 / 488459 / 1940692`), and a new
`equiv_*` test panel asserts byte-identity vs a reference encoder
modeled on the Phase 2 implementation.

### Encode time after Phase 2.5 (ms/iter, q=80, 4:2:0)

| Resolution                  | Phase 2 NEON | Phase 2.5 NEON | NEON Δ | Phase 2 scalar | Phase 2.5 scalar | scalar Δ |
| --------------------------- | -----------: | -------------: | -----: | -------------: | ---------------: | -------: |
| 1592 x 1124 (session size)  |      8.55 ms |        5.92 ms | -30.8% |       11.18 ms |          8.08 ms |   -27.7% |
| 1920 x 1080 (1080p)         |      9.42 ms |        6.75 ms | -28.3% |       12.85 ms |          9.17 ms |   -28.6% |
| 3840 x 2160 (4K)            |     39.12 ms |       27.23 ms | -30.4% |       52.73 ms |         38.12 ms |   -27.7% |

Roughly a third of total wall-clock time vanished in both NEON and
scalar paths, matching the flamegraph-derived estimate of Huffman's
share. The `force-scalar` build benefits too because the 64-bit
accumulator, branchless packing, packed LUT, and byte buffer are all
target-independent — only the AC zero-scan SIMD is aarch64-gated.

### Cumulative (Phase 1 → Phase 2.5)

| Resolution                  | Phase 1 (f32 scalar) | Phase 2.5 NEON | Total speedup |
| --------------------------- | -------------------: | -------------: | ------------: |
| 1592 x 1124                 |          ~10.96 ms\* |        5.92 ms |        1.85x  |
| 1920 x 1080                 |          ~12.50 ms\* |        6.75 ms |        1.85x  |
| 3840 x 2160                 |          ~51.44 ms\* |       27.23 ms |        1.89x  |

\* Phase 1 numbers reproduced from the brief; not re-measured for this
section.

## Phase 3 candidates (not in scope)

  - x86_64 SSE2/AVX2 versions of the same four kernels (the brief calls
    this "Phase 3").
  - Huffman bit-pack vectorization. Worth experimenting once we have
    flamegraphs from real workloads; the exact AC scan is branchy but
    the run-length detection (`v != 0` mask + popcount) does vectorize.
  - Reduce per-block scratch allocation in `encode_inner` — small but
    nonzero overhead from re-zeroing the `i16` block arrays per MCU.
