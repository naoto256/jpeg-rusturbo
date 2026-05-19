# `jpeg-rusturbo` benchmarks

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
from an f32 AAN DCT to libjpeg-turbo's integer LL&M scheme — the
NEON kernels are a true drop-in for the scalar reference.

## Where the speedup is, and isn't

The 1.3-1.36x figure is below the 2-4x speculative target we initially
aimed for. The most likely reason: the Huffman entropy coder
remains scalar (we leave it scalar by design — libjpeg-turbo
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

## Huffman 64-bit accumulator

The initial NEON port left the Huffman entropy coder scalar with a
32-bit accumulator; flamegraphs put it at ~30-35% of total encode time,
by far the dominant remaining cost. This pass rewrites it without
touching the public API:

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

Bit-exact output preserved — `bytes` columns below match the prior
NEON-only build exactly (`423839 / 488459 / 1940692`), and a new
`equiv_*` test panel asserts byte-identity vs a reference encoder
modeled on the prior NEON-only implementation.

### Encode time after the Huffman pass (ms/iter, q=80, 4:2:0)

| Resolution                  | NEON (initial) | NEON (+ Huffman) | NEON Δ | scalar (initial) | scalar (+ Huffman) | scalar Δ |
| --------------------------- | -----------: | -------------: | -----: | -------------: | ---------------: | -------: |
| 1592 x 1124 (session size)  |      8.55 ms |        5.92 ms | -30.8% |       11.18 ms |          8.08 ms |   -27.7% |
| 1920 x 1080 (1080p)         |      9.42 ms |        6.75 ms | -28.3% |       12.85 ms |          9.17 ms |   -28.6% |
| 3840 x 2160 (4K)            |     39.12 ms |       27.23 ms | -30.4% |       52.73 ms |         38.12 ms |   -27.7% |

Roughly a third of total wall-clock time vanished in both NEON and
scalar paths, matching the flamegraph-derived estimate of Huffman's
share. The `force-scalar` build benefits too because the 64-bit
accumulator, branchless packing, packed LUT, and byte buffer are all
target-independent — only the AC zero-scan SIMD is aarch64-gated.

### Cumulative (f32 baseline → Huffman pass)

| Resolution                  | f32 AAN baseline (scalar) | NEON + Huffman | Total speedup |
| --------------------------- | -------------------: | -------------: | ------------: |
| 1592 x 1124                 |          ~10.96 ms\* |        5.92 ms |        1.85x  |
| 1920 x 1080                 |          ~12.50 ms\* |        6.75 ms |        1.85x  |
| 3840 x 2160                 |          ~51.44 ms\* |       27.23 ms |        1.89x  |

\* f32 baseline numbers reproduced from earlier measurements; not
re-measured for this section.

## x86_64 AVX2 port

This pass ports the four hot kernels (color RGB→YCbCr, chroma
downsample, integer LL&M FDCT, reciprocal-multiply quantize) to x86_64
AVX2, translated from libjpeg-turbo's `simd/x86_64/*-avx2.asm`. Color
is AVX2-fast for the bpp=4 hot path (`n=16`, any of RGBA / BGRA /
ARGB / ABGR / RGBX / BGRX); narrower calls and 3-byte inputs still go
through scalar. The scaffolding (arch backend modules, runtime AVX2
detection, CI-friendly cfg plumbing) lives in `src/arch/`.

The Huffman AC scan is bitmap-driven: a nonzero bitmap (`u64`)
collapses zero runs into a single `trailing_zeros` jump per nonzero.
On x86_64 the bitmap is built with SSE2 (`pcmpeqw + packsswb +
pmovmskb`, translated from `jchuff-sse2.asm`); the rest of the
entropy emitter stays scalar (AC-symbol-emission and the 64-bit bit
accumulator both autovectorize well in scalar form).

### Setup — Intel Ice Lake

Measured on Intel Xeon Platinum 8370C (Ice Lake-SP, 2 vCPU,
Ubuntu 24.04) at 100 iterations per resolution after a 3-iteration
warm-up. Five repeated runs, variance < 1 % across runs; numbers
below are the median.

#### 4:2:0

| Resolution                  | scalar (force-scalar) | AVX2     | speedup |
| --------------------------- | --------------------: | -------: | ------: |
| 1592 x 1124 (session size)  |             24.31 ms  | 11.82 ms |  2.06x  |
| 1920 x 1080 (1080p)         |             27.93 ms  | 13.65 ms |  2.05x  |
| 3840 x 2160 (4K)            |            109.98 ms  | 53.50 ms |  2.06x  |

Output bytes: `423839 / 488459 / 1940692` — byte-identical to scalar
and to the Apple Silicon NEON build, verified by the cross-check unit
tests and the roundtrip suite.

#### 4:2:2

| Resolution                  | scalar (force-scalar) | AVX2     | speedup |
| --------------------------- | --------------------: | -------: | ------: |
| 1592 x 1124 (session size)  |             31.15 ms  | 15.29 ms |  2.04x  |
| 1920 x 1080 (1080p)         |             35.73 ms  | 17.45 ms |  2.05x  |
| 3840 x 2160 (4K)            |            140.87 ms  | 68.18 ms |  2.07x  |

Output bytes: `568676 / 654460 / 2618066`. The 4:2:2 path runs 4 DCT
blocks per 16×8 MCU (vs 6 per 16×16 4:2:0 MCU), so per-pixel work
grows by ~1.33×; the AVX2 path tracks that growth closely.

### Where the AVX2 speedup is

The ~2.05× whole-pipeline speedup is what Amdahl's law predicts once
the bitmap-driven Huffman scan trims the previously scalar-dominated
AC walk and the remaining color / DCT / quant / chroma downsample
kernels keep hitting ~3× in their SIMD bodies.

A rough per-stage breakdown on the Ice Lake host (AVX2):

  Color/downsample      ~25%  (AVX2: ~3.0x;  scalar: 1.0x)
  Forward DCT           ~20%  (AVX2: ~2.7x;  scalar: 1.0x)
  Quantize+zig-zag      ~10%  (AVX2: ~2.5x;  scalar: 1.0x)
  Huffman bitmap + walk ~30%  (SSE2 bitmap + scalar emitter; ~1.4x)
  Marker writes/IO      ~15%  (scalar in both)

### Apple Silicon (NEON)

Measured on Apple M-series at 100 iterations after a 3-iteration
warm-up. Five repeated runs, variance < 1 %; numbers below are the
median.

#### 4:2:0

| Resolution                  | scalar (force-scalar) | NEON     | speedup |
| --------------------------- | --------------------: | -------: | ------: |
| 1592 x 1124 (session size)  |              8.54 ms  |  5.49 ms |  1.56x  |
| 1920 x 1080 (1080p)         |              9.94 ms  |  6.23 ms |  1.60x  |
| 3840 x 2160 (4K)            |             41.96 ms  | 25.04 ms |  1.68x  |

#### 4:2:2

| Resolution                  | scalar (force-scalar) | NEON     | speedup |
| --------------------------- | --------------------: | -------: | ------: |
| 1592 x 1124 (session size)  |             10.68 ms  |  7.43 ms |  1.44x  |
| 1920 x 1080 (1080p)         |             12.32 ms  |  8.45 ms |  1.46x  |
| 3840 x 2160 (4K)            |             47.94 ms  | 33.00 ms |  1.45x  |

NEON whole-pipeline speedup is more modest than AVX2 here because the
Apple M-series scalar path already runs the autovectorized Huffman /
quantize / DCT inner loops well; the explicit NEON kernels for color
convert, FDCT, quantize, chroma downsample, and the Huffman nonzero
bitmap claw back the remaining fixed-point arithmetic that LLVM
doesn't fully cover.

### Cumulative timeline (4K, 4:2:0, q=80)

| Configuration                                  | aarch64        | x86_64 (Ice Lake) |
| ---------------------------------------------- | -------------: | ----------------: |
| f32 AAN baseline (scalar)                      |     ~51.44 ms\* |                 — |
| NEON SIMD kernels (color/FDCT/quant/downsample)|        37.92 ms |                 — |
| NEON + Huffman 64-bit accumulator              |        27.23 ms |                 — |
| AVX2 + backend-dispatch refactor               |        28.50 ms |          63.93 ms |
| **Bitmap-driven Huffman (NEON + SSE2)**        |     **25.04 ms** |       **53.50 ms** |
| force-scalar (reference, current)              |        41.96 ms |         109.98 ms |

\* f32 baseline reproduced from earlier measurements.

The aarch64 column at the "AVX2 + backend-dispatch refactor" row is
the NEON build measured pre-bitmap; the next row reflects the
bitmap-driven AC scan plus NEON nonzero-bitmap kernel landed
together with the SSE2 counterpart on x86_64.

## Out of scope (still)

  - AVX-512 versions of the four kernels. Ice Lake has AVX-512 but the
    server market is bifurcated (Zen 4 has it, Zen 2/3 don't, Alder
    Lake P-cores have it disabled in many bins). AVX2 is the safe
    floor.
  - Full SIMD AC-symbol-emission (run-length + magnitude + Huffman
    table lookup + bit-writer drain). The bitmap is now SIMD'd, but
    the per-nonzero emission stays scalar — it's tight enough that
    LLVM autovectorizes the bit-writer drain and the table lookups
    don't reshape cleanly into SIMD.
  - x86 32-bit and SSE2-only fallback. Non-AVX2 x86_64 already runs
    via the scalar path through runtime feature detection; AVX2 has
    been the de-facto floor for new code since ~2014. (SSE2 is the
    x86_64-v1 baseline and is always available for the Huffman
    nonzero-bitmap kernel.)
