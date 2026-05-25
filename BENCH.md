# `jpeg-rusturbo` benchmarks

All numbers below come from the unified harness in `src/bin/bench.rs`
(encode + decode) and `tests/comparison_bench.rs` (vs the `image`
crate). Methodology is intentionally simple — single timed batch, 50
measured iterations after a 3-iteration warm-up, no statistical
filtering. The synthetic XOR-pattern image (`make_image` in
`bench.rs`) keeps the AC histogram non-degenerate so the entropy
coder isn't artificially fast, and gives pixel-identical inputs
across hosts for apples-to-apples comparison.

Reproduce on any host with:

```sh
cargo run --release --bin bench -- --section all                          # SIMD build
cargo run --release --features force-scalar --bin bench -- --section all  # scalar fallback
cargo test  --release --test comparison_bench -- --ignored --nocapture    # vs image crate
```

## Hosts (0.6.0)

| Label                | CPU                                       | Cores | SIMD floor       |
| -------------------- | ----------------------------------------- | ----- | ---------------- |
| Apple M-series       | Apple Silicon (M-series)                  | 8 P+E | NEON (always-on) |
| Intel Xeon (Cascade Lake) | Xeon Platinum 8272CL @ 2.60 GHz        | 4 vCPU | AVX2 + SSE2  |

Intel host was a fresh `Standard_D4s_v3` Linux VM in Azure
`centralus` (Ubuntu 24.04, Rust stable). Azure's D4s_v3 inventory
rotates through Broadwell → Skylake → Cascade Lake → Ice Lake, so
absolute ms numbers aren't directly comparable across releases;
the SIMD-vs-scalar ratios and the vs-`image`-crate ratios are.

Apple M was a developer laptop with background chat / IDE running —
performance-core load was light but non-zero, so trust the *ratios*
more than the raw ms numbers there.

The release profile in `Cargo.toml` is `lto = "fat"` +
`codegen-units = 1` (set in 0.6.0); all numbers below are with that
profile.

---

## Section A — encode pipeline

`q=80`, single thread, optimize-huffman off. Pure encode time
(`JpegEncoder::encode_rgba`), ms/iter.

### Apple M-series (aarch64)

| Resolution                  | 4:4:4 NEON | 4:4:4 scalar | 4:2:2 NEON | 4:2:2 scalar | 4:2:0 NEON | 4:2:0 scalar |
| --------------------------- | ---------: | -----------: | ---------: | -----------: | ---------: | -----------: |
| 1592 × 1124 (session size)  |    9.96 ms |     13.83 ms |    6.90 ms |     10.16 ms |    5.19 ms |      8.20 ms |
| 1920 × 1080 (1080p)         |   11.29 ms |     15.94 ms |    7.95 ms |     11.70 ms |    6.06 ms |      9.47 ms |
| 3840 × 2160 (4K)            |   44.75 ms |     62.71 ms |   31.39 ms |     45.90 ms |   23.57 ms |     40.94 ms |

NEON speedup vs `force-scalar`: 1.38× (4:4:4 4K) → 1.74× (4:2:0 4K).

### Intel Xeon (Cascade Lake) x86_64

| Resolution                  | 4:4:4 AVX2 | 4:4:4 scalar | 4:2:2 AVX2 | 4:2:2 scalar | 4:2:0 AVX2 | 4:2:0 scalar |
| --------------------------- | ---------: | -----------: | ---------: | -----------: | ---------: | -----------: |
| 1592 × 1124 (session size)  |   34.90 ms |     64.82 ms |   20.65 ms |     48.70 ms |   15.93 ms |     38.24 ms |
| 1920 × 1080 (1080p)         |   40.15 ms |     74.78 ms |   23.58 ms |     55.83 ms |   18.30 ms |     43.85 ms |
| 3840 × 2160 (4K)            |  159.51 ms |    298.01 ms |   93.48 ms |    222.50 ms |   73.37 ms |    173.63 ms |

AVX2 speedup vs `force-scalar`: 1.87× (4:4:4 4K) → 2.37× (4:2:0 4K).
Encode-side speedup unchanged vs 0.5.0; the encode pipeline didn't
gain new SIMD kernels in 0.6.0.

Output bytes are byte-identical across SIMD ↔ scalar and across
hosts (e.g. 4K 4:2:0 q=80 = `1940692` bytes everywhere). This is the
bit-exact equivalence we set out to preserve when porting
libjpeg-turbo's integer LL&M DCT, and is asserted in the unit tests.

---

## Section B — threads scaling

`q=80`, 4:2:0, optimize-huffman off. Threads are MCU-rows partitioned
across rayon worker pool; `threads=auto` picks
`available_parallelism()`.

### Apple M-series (NEON, 8 cores)

| Resolution         | threads=1 | threads=2 | threads=4 | threads=8 | threads=auto | speedup (auto/1) |
| ------------------ | --------: | --------: | --------: | --------: | -----------: | ---------------: |
| 1920×1080 (1080p)  |   5.87 ms |   4.28 ms |   3.66 ms |   3.47 ms |      3.47 ms |            1.69× |
| 3840×2160 (4K)     |  23.82 ms |  17.09 ms |  14.39 ms |  13.31 ms |     13.31 ms |            1.79× |

### Intel Xeon (Cascade Lake, AVX2, 4 vCPU)

| Resolution         | threads=1 | threads=2 | threads=4 | threads=8 | threads=auto | speedup (auto/1) |
| ------------------ | --------: | --------: | --------: | --------: | -----------: | ---------------: |
| 1920×1080 (1080p)  |  18.35 ms |  15.22 ms |  15.99 ms |  16.03 ms |     15.39 ms |            1.19× |
| 3840×2160 (4K)     |  73.20 ms |  63.68 ms |  61.07 ms |  59.20 ms |     58.88 ms |            1.24× |

The 4-vCPU Cascade Lake saturates at 2 threads on 1080p; per-MCU
work-per-chunk isn't enough to offset rayon's scheduling cost beyond
that. The 8-core Apple host gets clean 1.8× at 4K. Output bytes are
byte-identical across all thread counts (asserted in
`tests/threaded.rs`).

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
| Apple M-series (NEON)       |    23.81 |   42.71 | 1.79×  |
| Intel Xeon (Cascade Lake AVX2) |  73.65 |  111.94 | 1.52×  |

Worth it when bandwidth matters more than CPU; opt-in by default for
exactly that trade-off.

---

## Section D — decode pipeline (NEW in 0.6.0)

`q=80`, default knobs. Times decode of the JPEGs that Section A
emits (same `make_image` source), so a single resolution × subsampling
cell is comparable across the encode / decode columns.

### Apple M-series (aarch64)

| Resolution                  | 4:4:4 NEON | 4:4:4 scalar | 4:2:2 NEON | 4:2:2 scalar | 4:2:0 NEON | 4:2:0 scalar |
| --------------------------- | ---------: | -----------: | ---------: | -----------: | ---------: | -----------: |
| 1592 × 1124 (session size)  |   10.74 ms |     14.20 ms |    7.51 ms |     12.01 ms |    5.78 ms |     10.10 ms |
| 1920 × 1080 (1080p)         |   12.47 ms |     16.05 ms |    8.56 ms |     13.57 ms |    6.58 ms |     11.42 ms |
| 3840 × 2160 (4K)            |   48.84 ms |     64.22 ms |   34.25 ms |     55.12 ms |   25.98 ms |     45.81 ms |

NEON decode speedup vs `force-scalar`: 1.32× (4:4:4 4K) → **1.76×**
(4:2:0 4K). Source: IDCT (cycle 1 / DS1 NEON jidctint), color
convert (cycle 1 / DS3 NEON ycc_row_to_rgb), fancy chroma upsample
(cycle 2 / DS5 NEON h2v2 + h2 fancy). The 4:4:4 path lacks fancy
upsample (no chroma downsampling to upsample back), so the speedup
there comes from IDCT + color only.

### Intel Xeon (Cascade Lake) x86_64

| Resolution                  | 4:4:4 AVX2 | 4:4:4 scalar | 4:2:2 AVX2 | 4:2:2 scalar | 4:2:0 AVX2 | 4:2:0 scalar |
| --------------------------- | ---------: | -----------: | ---------: | -----------: | ---------: | -----------: |
| 1592 × 1124 (session size)  |   31.97 ms |     46.38 ms |   23.19 ms |     39.08 ms |   18.62 ms |     33.48 ms |
| 1920 × 1080 (1080p)         |   37.09 ms |     53.52 ms |   26.50 ms |     45.02 ms |   21.47 ms |     38.58 ms |
| 3840 × 2160 (4K)            |  149.80 ms |    213.50 ms |  107.99 ms |    179.86 ms |   87.00 ms |    154.82 ms |

AVX2 decode speedup vs `force-scalar`: 1.43× (4:4:4 4K) → **1.78×**
(4:2:0 4K). Cycle 1 DS2 (AVX2 jidctint) + DS4 (AVX2 ycc_row_to_rgb)
+ cycle 2 DS6 (AVX2 fancy upsample) account for the gain.

---

## vs `image` crate

`image` (`v0.25`) is the de-facto Rust image library. It bundles a
scalar JPEG encoder hardcoded to 4:2:0, and ships a SIMD-accelerated
JPEG decoder. We bench both sides on the same synthetic content from
the same harness on each host.

### Encode (RGB → JPEG, q=80)

| Host                 | Resolution | ours 4:2:0 | image (4:2:0) | ratio (img/ours) |
| -------------------- | ---------- | ---------: | ------------: | ---------------: |
| Apple M-series (NEON) | 1592×1124 |    5.20 ms |      15.21 ms |            2.93× |
|                      | 1920×1080  |    5.89 ms |      17.18 ms |            2.92× |
|                      | 3840×2160  |   23.38 ms |      66.70 ms |            2.85× |
| Cascade Lake (AVX2)  | 1592×1124  |   23.19 ms |      75.32 ms |            3.25× |
|                      | 1920×1080  |   26.51 ms |      86.74 ms |            3.27× |
|                      | 3840×2160  |  105.12 ms |     344.80 ms |            3.28× |

For reference, our 4:4:4 and 4:2:2 numbers are in Section A above —
`image` only supports 4:2:0 so the ratio above is the fair encode-
vs-encode comparison.

### Decode (JPEG → RGB, our encoder's q=80 4:2:0 output)

| Host                 | Resolution | ours       | image     | ratio (img/ours) |
| -------------------- | ---------- | ---------: | --------: | ---------------: |
| Apple M-series (NEON) | 1592×1124 |    5.57 ms |   4.25 ms |            0.76× |
|                      | 1920×1080  |    6.30 ms |   4.92 ms |            0.78× |
|                      | 3840×2160  |   25.05 ms |  19.22 ms |            0.77× |
| Cascade Lake (AVX2)  | 1592×1124  |   18.59 ms |  13.43 ms |            0.72× |
|                      | 1920×1080  |   21.76 ms |  14.14 ms |            0.65× |
|                      | 3840×2160  |   88.40 ms |  68.07 ms |            0.77× |

(ratio > 1 means jpeg-rusturbo is faster)

The decode-side gap closed substantially in 0.6.0: 0.5.0 sat at
~0.41× on Apple M (image about 2.4× faster); 0.6.0 lands at **0.77×**
(image now only ~1.3× faster). The remaining gap is the Huffman
entropy decoder, which stays scalar in 0.6.0. Closing that is a
separate piece of work (post-0.6 roadmap).

---

## Where the SIMD speedup is

A rough per-stage breakdown for 4K 4:2:0 q=80 (estimated from
`cargo flamegraph`, not committed to the repo):

```
Encode side:
  Color/downsample      ~25%   NEON ~3.0× / AVX2 ~3.0× / scalar 1.0×
  Forward DCT           ~20%   NEON ~2.5× / AVX2 ~2.7× / scalar 1.0×
  Quantize+zig-zag      ~10%   NEON ~1.8× / AVX2 ~2.5× / scalar 1.0×
  Huffman (64-bit acc + bitmap)  ~30%   NEON-bitmap ~1.4× / SSE2-bitmap ~1.4×
  Marker writes/IO      ~15%   scalar in both

Decode side (new in 0.6.0):
  Entropy decode        ~35%   scalar in both (post-0.6 roadmap)
  IDCT                  ~25%   NEON ~2.0× / AVX2 ~2.5× / scalar 1.0×
  Color convert (YCC→RGB) ~20%   NEON ~6.7× / AVX2 ~3.5× / scalar 1.0×
  Chroma upsample fancy ~15%   NEON ~1.3-14× (kernel-dep) / AVX2 sim.
  Marker walk / IO      ~5%    scalar in both
```

Per-stage SIMD bodies hit close to expected speedups; whole-pipeline
numbers are bounded by Amdahl on the (still scalar) Huffman emit /
decode and the unavoidable serial marker / IO sections.

## Out of scope

- AVX-512 versions of the four kernels. The server market is
  bifurcated (Zen 4 yes, Zen 2/3 no, Alder Lake P-cores
  bin-disabled); AVX2 stays the floor for x86_64 in this crate.
- Full SIMD AC-symbol-emission on the encoder side. The bitmap is
  SIMD; the per-nonzero emission stays scalar — it's tight enough
  that LLVM autovectorizes the bit-writer drain and table lookups
  don't reshape cleanly into SIMD.
- SIMD Huffman *decode*. The bit reader + canonical-Huffman table
  walk is branchy; the cost-vs-implementation-complexity for a
  table-driven SIMD decoder is non-trivial. Tracked separately from
  the per-stage kernel ports that landed in 0.6.0.
