# `jpeg-rusturbo` benchmarks

All numbers below come from the unified harness in `src/bin/bench.rs`
(encode-side) and `tests/comparison_bench.rs` (vs the `image` crate).
Methodology is intentionally simple — single timed batch, 50 measured
iterations after a 3-iteration warm-up, no statistical filtering. The
synthetic XOR-pattern image (`make_image` in `bench.rs`) keeps the AC
histogram non-degenerate so the entropy coder isn't artificially
fast, and gives pixel-identical inputs across hosts for
apples-to-apples comparison.

Reproduce on any host with:

```sh
cargo run --release --bin bench -- --section all                          # SIMD build
cargo run --release --features force-scalar --bin bench -- --section all  # scalar fallback
cargo test  --release --test comparison_bench -- --ignored --nocapture    # vs image crate
```

## Hosts

| Label                | CPU                                | Cores | SIMD floor       |
| -------------------- | ---------------------------------- | ----- | ---------------- |
| Apple M-series       | Apple Silicon (M-series)           | 8 P+E | NEON (always-on) |
| Intel Xeon Broadwell | Xeon E5-2673 v4 @ 2.30 GHz         | 4 vCPU | AVX2            |

Intel Xeon was a fresh `Standard_D4s_v3` Linux VM (Ubuntu 24.04,
Rust stable). Apple M-series was a developer laptop with background
chat / IDE running — performance-core load was light but non-zero,
so trust the *ratios* more than the raw ms numbers there.

---

## Section A — encode pipeline

`q=80`, single thread, optimize-huffman off. Pure encode time
(JpegEncoder::encode_rgba), ms/iter.

### Apple M-series (aarch64)

| Resolution                  | 4:4:4 NEON | 4:4:4 scalar | 4:2:2 NEON | 4:2:2 scalar | 4:2:0 NEON | 4:2:0 scalar |
| --------------------------- | ---------: | -----------: | ---------: | -----------: | ---------: | -----------: |
| 1592 × 1124 (session size)  |   10.79 ms |     14.49 ms |    7.54 ms |     10.69 ms |    5.56 ms |      8.63 ms |
| 1920 × 1080 (1080p)         |   12.42 ms |     16.81 ms |    8.60 ms |     12.36 ms |    6.39 ms |      9.99 ms |
| 3840 × 2160 (4K)            |   47.55 ms |     65.58 ms |   34.18 ms |     48.31 ms |   25.80 ms |     44.68 ms |

NEON speedup vs `force-scalar`: 1.34× (4:4:4 4K) → 1.73× (4:2:0 4K).
The 4:2:0 path benefits more because chroma downsample and the
horizontal-half color-convert MCU loops are where the explicit NEON
kernels do the most work; pure 4:4:4 is closer to the LLVM
autovectorized scalar floor.

### Intel Xeon Broadwell (x86_64)

| Resolution                  | 4:4:4 AVX2 | 4:4:4 scalar | 4:2:2 AVX2 | 4:2:2 scalar | 4:2:0 AVX2 | 4:2:0 scalar |
| --------------------------- | ---------: | -----------: | ---------: | -----------: | ---------: | -----------: |
| 1592 × 1124 (session size)  |   47.11 ms |     75.69 ms |   31.07 ms |     58.16 ms |   24.53 ms |     42.40 ms |
| 1920 × 1080 (1080p)         |   54.03 ms |     90.61 ms |   35.21 ms |     64.84 ms |   26.59 ms |     49.24 ms |
| 3840 × 2160 (4K)            |  215.03 ms |    352.49 ms |  142.58 ms |    253.57 ms |  112.19 ms |    196.41 ms |

AVX2 speedup vs `force-scalar`: 1.61× (4:4:4 4K) → 1.75× (4:2:0 4K).
Broadwell is two generations behind Apple Silicon on per-clock
throughput, so absolute ms are larger; the ratios are what's
portable.

Output bytes are byte-identical across SIMD ↔ scalar and across
hosts (e.g. 4K 4:2:0 q=80 = `1940692` bytes everywhere). This is the
bit-exact equivalence we set out to preserve when porting
libjpeg-turbo's integer LL&M DCT, and is asserted in the unit tests.

---

## Section B — threads scaling

`q=80`, 4:2:0, optimize-huffman off. Threads are MCU-rows partitioned
across rayon worker pool; `threads=auto` picks `available_parallelism()`.

### Apple M-series (NEON, 8 cores)

| Resolution         | threads=1 | threads=2 | threads=4 | threads=8 | threads=auto | speedup (auto/1) |
| ------------------ | --------: | --------: | --------: | --------: | -----------: | ---------------: |
| 1920×1080 (1080p)  |   6.18 ms |   4.68 ms |   4.08 ms |   3.71 ms |      3.70 ms |            1.67× |
| 3840×2160 (4K)     |  25.39 ms |  18.37 ms |  15.70 ms |  14.52 ms |     13.90 ms |            1.83× |

### Intel Xeon Broadwell (AVX2, 4 vCPU)

| Resolution         | threads=1 | threads=2 | threads=4 | threads=8 | threads=auto | speedup (auto/1) |
| ------------------ | --------: | --------: | --------: | --------: | -----------: | ---------------: |
| 1920×1080 (1080p)  |  27.32 ms |  20.36 ms |  20.56 ms |  20.77 ms |     20.42 ms |            1.34× |
| 3840×2160 (4K)     | 111.39 ms |  81.98 ms |  76.94 ms |  78.19 ms |     76.78 ms |            1.45× |

The 4-vCPU Broadwell saturates at 2 threads — beyond that the
parallel quantize-rows stage doesn't have enough work per chunk to
offset rayon's scheduling cost. The 8-core Apple host gets clean
1.8× at 4K, less at 1080p (Amdahl: serial-emit-rows holds about a
third of total time on this content).

Output bytes are byte-identical across all thread counts (asserted
in `tests/threaded.rs`); threading partitions the per-MCU work but
the entropy coder emits the same bitstream.

---

## Section C — optimized Huffman (size)

`set_optimize_huffman(true)` enables a second pass: count symbol
frequencies, build canonical Huffman tables (T.81 K.2/K.3), re-emit
the scan. Size delta is host-independent (it's purely about the
tables); we show the Apple M-series numbers, the Intel numbers are
identical to within rounding.

### Size delta vs default tables (4K, all subsamplings × quality)

| Subsampling | q  | bytes (off) | bytes (on)  | Δ        |
| ----------- | -- | ----------: | ----------: | -------: |
| 4:4:4       | 70 |   2,804,405 |   2,644,405 |   −5.71% |
| 4:4:4       | 80 |   3,701,792 |   3,491,157 |   −5.69% |
| 4:4:4       | 90 |   5,663,832 |   5,344,940 |   −5.63% |
| 4:2:2       | 70 |   1,996,329 |   1,886,443 |   −5.50% |
| 4:2:2       | 80 |   2,618,066 |   2,478,247 |   −5.34% |
| 4:2:2       | 90 |   3,984,627 |   3,773,697 |   −5.29% |
| 4:2:0       | 70 |   1,498,562 |   1,419,961 |   −5.25% |
| 4:2:0       | 80 |   1,940,692 |   1,846,090 |   −4.87% |
| 4:2:0       | 90 |   2,905,130 |   2,771,322 |   −4.61% |

Roughly a consistent **−5%** across the matrix on synthetic content.
On natural photographic input the win typically sits in the 4–10%
range, matching `cjpeg -optimize`.

### Time cost (two-pass)

Two-pass means encode roughly doubles in wall-clock. On 4K 4:2:0 q=80:

| Host                        | off (ms) | on (ms) | factor |
| --------------------------- | -------: | ------: | -----: |
| Apple M-series (NEON)       |    26.28 |   47.24 | 1.80×  |
| Intel Xeon Broadwell (AVX2) |   111.94 |  161.54 | 1.44×  |

Worth it when bandwidth matters more than CPU; opt-in by default for
exactly that trade-off.

---

## vs `image` crate

`image` (`v0.25`) is the de-facto Rust image library. It bundles a
scalar JPEG encoder hardcoded to 4:2:0, and routes JPEG decode
through `zune-jpeg` (SIMD-accelerated). We bench both sides on the
same synthetic content from the same harness on each host.

### Encode (RGB → JPEG, q=80)

| Host                 | Resolution | ours 4:2:0 | image (4:2:0) | ratio (img/ours) |
| -------------------- | ---------- | ---------: | ------------: | ---------------: |
| Apple M-series (NEON) | 1592×1124 |    5.62 ms |      14.36 ms |            2.56× |
|                      | 1920×1080  |    6.32 ms |      16.22 ms |            2.57× |
|                      | 3840×2160  |   24.93 ms |      63.00 ms |            2.53× |
| Intel Xeon (AVX2)    | 1592×1124  |   29.75 ms |      94.71 ms |            3.18× |
|                      | 1920×1080  |   34.16 ms |     108.19 ms |            3.17× |
|                      | 3840×2160  |  134.55 ms |     432.59 ms |            3.22× |

For reference, our 4:4:4 and 4:2:2 numbers are in Section A above —
`image` only supports 4:2:0 so the ratio above is the fair encode-
vs-encode comparison.

### Decode (JPEG → RGB, our encoder's q=80 4:2:0 output)

| Host                 | Resolution | ours       | image     | ratio (img/ours) |
| -------------------- | ---------- | ---------: | --------: | ---------------: |
| Apple M-series (NEON) | 1592×1124 |   10.55 ms |   4.39 ms |            0.42× |
|                      | 1920×1080  |   12.16 ms |   4.97 ms |            0.41× |
|                      | 3840×2160  |   48.30 ms |  19.65 ms |            0.41× |
| Intel Xeon (AVX2)    | 1592×1124  |   41.56 ms |  18.32 ms |            0.44× |
|                      | 1920×1080  |   47.61 ms |  20.84 ms |            0.44× |
|                      | 3840×2160  |  186.20 ms |  97.79 ms |            0.53× |

Our decoder is still scalar (libjpeg-turbo's `jidctint` IDCT,
integer YCbCr→RGB, box / fancy upsample). `zune-jpeg` has SIMD IDCT
and color paths, which is why it leads decode by roughly the same
factor we lead encode. Decoder SIMD is on the roadmap; for now,
roundtrip throughput (encode + decode combined) is close to a wash
between the two crates on these sizes.

---

## Where the SIMD speedup is

A rough per-stage breakdown for 4K 4:2:0 q=80 encode (estimated
from `cargo flamegraph`, not committed to the repo):

```
Color/downsample      ~25%   NEON ~3.0× / AVX2 ~3.0× / scalar 1.0×
Forward DCT           ~20%   NEON ~2.5× / AVX2 ~2.7× / scalar 1.0×
Quantize+zig-zag      ~10%   NEON ~1.8× / AVX2 ~2.5× / scalar 1.0×
Huffman (64-bit acc + bitmap)  ~30%   NEON-bitmap ~1.4× / SSE2-bitmap ~1.4×
Marker writes/IO      ~15%   scalar in both
```

Per-stage SIMD bodies hit close to expected speedups; whole-pipeline
numbers are bounded by Amdahl on the (still partially scalar)
Huffman emitter and the unavoidable serial marker/IO sections.

## Out of scope

- AVX-512 versions of the four kernels. The server market is
  bifurcated (Zen 4 yes, Zen 2/3 no, Alder Lake P-cores bin-disabled);
  AVX2 stays the floor for x86_64 in this crate.
- Full SIMD AC-symbol-emission. The bitmap is SIMD; the per-nonzero
  emission stays scalar — it's tight enough that LLVM autovectorizes
  the bit-writer drain and table lookups don't reshape cleanly into
  SIMD.
- Decoder SIMD. Tracked for a later release; until then the scalar
  decoder is bit-correct and roundtrip-tested but slower than
  `zune-jpeg`.

---

## Historical context (cumulative timeline, 4K 4:2:0 q=80)

This crate started as an f32 AAN DCT scalar baseline and has moved
through four pass-shaped SIMD ports. Numbers below are from the
Apple M-series column where directly measured; Intel column only has
entries from the point AVX2 landed (the older Ice Lake host used in
the 0.3.x measurements is no longer available, so 0.5.0 Intel
numbers are on Broadwell and not directly comparable to prior
releases on that axis).

| Configuration                                  | aarch64    | x86_64           |
| ---------------------------------------------- | ---------: | ---------------: |
| f32 AAN baseline (scalar)                      |  ~51.44 ms\* |               — |
| Integer LL&M + NEON kernels                    |    37.92 ms |               — |
| + 64-bit Huffman accumulator                   |    27.23 ms |               — |
| + AVX2 + backend-dispatch refactor (0.3.x)     |    28.50 ms |          63.93 ms (Ice Lake) |
| + Bitmap-driven Huffman (NEON + SSE2)          |    25.04 ms |          53.50 ms (Ice Lake) |
| **0.5.0 current** (above + threading + optimized Huffman, single-thread baseline) |    25.80 ms |        112.19 ms (Broadwell) |
| **0.5.0 with `threads=auto`**                  |  **13.90 ms** |      **76.78 ms** (Broadwell) |

\* f32 baseline reproduced from earlier measurements; not re-measured
for 0.5.0.

The 0.5.0 row on aarch64 is flat vs 0.4.0 single-thread (no encoder
hot-path changes that release; the work was on the decoder + the
opt-in `threads`, `restart_interval`, `optimize_huffman`, and custom
quant-table APIs). The threading speedup is the new headline at 4K.
