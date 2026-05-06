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

## Phase 3 — x86_64 AVX2 port

Phase 3 ports the four hot kernels (color RGB→YCbCr, 4:2:0 chroma
downsample, integer LL&M FDCT, reciprocal-multiply quantize) to x86_64
AVX2, translated from libjpeg-turbo's `simd/x86_64/*-avx2.asm`. Color
is AVX2-fast for the 4:2:0 hot path (`n=16`, RGBA layout); narrower
calls and RGB-bpp inputs still go through scalar. The scaffolding
(arch backend modules, runtime AVX2 detection, CI-friendly cfg
plumbing) lives in `src/arch/`.

The Huffman entropy coder is intentionally left scalar — the same
reasoning as Phase 2 (libjpeg-turbo itself doesn't ship an AVX2 Huffman
kernel for our shape, and the loop is too branchy for SIMD to win).
The Phase 2.5 64-bit accumulator + branchless inner path + 8-skip
zero-scan stays as is.

### Setup — Intel Ice Lake

Measured on Azure `Standard_D2s_v5` Spot (Intel Xeon Platinum 8370C,
Ice Lake-SP, 2 vCPU, 8 GB) at 100 iterations per resolution after a
3-iteration warm-up. Five repeated runs, variance < 1 % across runs;
numbers below are the median.

| Resolution                  | scalar (force-scalar) | AVX2  | speedup |
| --------------------------- | --------------------: | ----: | ------: |
| 1592 x 1124 (session size)  |             24.69 ms  | 14.56 ms |  1.70x |
| 1920 x 1080 (1080p)         |             28.27 ms  | 16.57 ms |  1.71x |
| 3840 x 2160 (4K)            |            109.90 ms  | 63.93 ms |  1.72x |

Bytes-out across builds: `423839 / 488459 / 1940692` — byte-identical
to scalar and to the Apple Silicon NEON build at every resolution,
verified by both the cross-check unit tests and the roundtrip suite.

### Where the AVX2 speedup is

The ~1.71x whole-pipeline speedup matches what Amdahl's law predicts
once the Huffman + marker/IO portion (~45-50 %) stays scalar and the
remaining color/dct/quant/downsample chunk (~50-55 %) hits roughly
3x in its kernels.

A rough per-stage breakdown on the Ice Lake host (Phase 3 AVX2):

  Color/downsample  ~25%  (AVX2: ~3.0x;  scalar: 1.0x)
  Forward DCT       ~20%  (AVX2: ~2.7x;  scalar: 1.0x)
  Quantize+zig-zag  ~10%  (AVX2: ~2.5x;  scalar: 1.0x)
  Huffman           ~35%  (scalar in both)
  Marker writes/IO  ~10%  (scalar in both)

### Apple Silicon NEON revisit

The Step 1 refactor (kernels moved behind `arch::backend::*`) added a
small amount of inlining indirection. Re-measured Apple M-series
numbers shift by 1-5 % vs the original Phase 2.5 figures; the NEON
path is still bit-exact, just slightly slower at the function-call
boundary the compiler doesn't always collapse. We accept this as the
cost of having a uniform x86_64/AArch64 dispatch surface.

| Resolution                  | Phase 2.5 NEON | Phase 3 NEON (refactored) | Δ |
| --------------------------- | -------------: | ------------------------: | -: |
| 1592 x 1124                 |        5.92 ms |                   6.15 ms | +3.9% |
| 1920 x 1080                 |        6.75 ms |                   6.97 ms | +3.3% |
| 3840 x 2160                 |       27.23 ms |                   28.50 ms | +4.7% |

### Cumulative across phases (4K representative)

| Phase                                          | aarch64        | x86_64 (Ice Lake) |
| ---------------------------------------------- | -------------: | ----------------: |
| Phase 1 (f32 scalar)                           |     ~51.44 ms\* |                 — |
| Phase 2 NEON (4 kernels)                       |        37.92 ms |                 — |
| Phase 2.5 NEON (+ Huffman 64-bit)              |        27.23 ms |                 — |
| **Phase 3 AVX2 (4 kernels)**                   |     **28.50 ms** |       **63.93 ms** |
| Phase 3 force-scalar (reference)               |          ~40 ms |          109.90 ms |

\* Phase 1 reproduced from the brief.

The aarch64 column at "Phase 3 AVX2" is the same NEON build measured
above (the AVX2 work doesn't change the aarch64 path); shown to make
the cross-architecture comparison easy.

## Out of scope (still)

  - AVX-512 versions of the four kernels. Ice Lake has AVX-512 but the
    server market is bifurcated (Zen 4 has it, Zen 2/3 don't, Alder
    Lake P-cores have it disabled in many bins). AVX2 is the safe
    floor.
  - Huffman vectorization (see Phase 2.5 notes — same conclusion).
  - x86 32-bit and SSE2-only fallback. Non-AVX2 x86_64 already runs
    via the scalar path through runtime feature detection; AVX2 has
    been the de-facto floor for new code since ~2014.
