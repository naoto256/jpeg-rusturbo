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

## Hosts (0.7.0)

| Label                | CPU                                       | Cores | SIMD floor       |
| -------------------- | ----------------------------------------- | ----- | ---------------- |
| Apple M-series       | Apple Silicon (M-series)                  | 8 P+E | NEON (always-on) |
| Intel Xeon (Cascade Lake) | Xeon Platinum 8272CL @ 2.60 GHz        | 4 vCPU | AVX2 + SSE2  |

Intel host was a fresh 4-vCPU cloud VM (Ubuntu 24.04, Rust stable).
The shared-tenant 4-vCPU bucket we use rotates through Broadwell →
Skylake → Cascade Lake → Ice Lake as the underlying inventory
changes, so absolute ms numbers aren't directly comparable across
releases; the SIMD-vs-scalar ratios and the vs-`image`-crate ratios
are.

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
| 1592 × 1124 (session size)  |   34.09 ms |     48.47 ms |   24.43 ms |     42.88 ms |   19.51 ms |     37.40 ms |
| 1920 × 1080 (1080p)         |   38.73 ms |     56.62 ms |   28.15 ms |     49.42 ms |   22.39 ms |     43.12 ms |
| 3840 × 2160 (4K)            |  152.75 ms |    222.94 ms |  110.76 ms |    196.81 ms |   89.32 ms |    170.70 ms |

AVX2 decode speedup vs `force-scalar`: 1.46× (4:4:4 4K) → **1.91×**
(4:2:0 4K). AVX2 jidctint + AVX2 ycc_row_to_rgb + AVX2 fancy upsample
account for the gain on synthetic content. Sparse fast paths and the
combined Huffman LUT fire near-zero on XOR-pattern blocks, so the
synthetic numbers move only with the SIMD kernels themselves.

---

## Section D-natural — decode pipeline, natural-like content

The synthetic XOR-pattern image used everywhere above keeps the AC
histogram non-degenerate, but it has *no* smooth regions — so the
IDCT sparse fast paths (DC-only, rows-4-7-zero, cols-4-7-zero) which
fire on flat sky / wall / out-of-focus background blocks in real
photographs never trigger. Section D-natural runs the same decode
harness against `make_natural_image` — a procedural mix of smooth
gradients, low-amplitude noise, and sharp edges that approximates
the AC distribution of camera content.

Same JPEGs are produced by the same encoder at q=80; the natural-like
input simply produces JPEGs that are 8–12× smaller and decode 3×
faster because most blocks have only a few low-frequency coefficients
to invert.

### Apple M-series (aarch64)

| Resolution                  | 4:4:4 NEON | 4:2:2 NEON | 4:2:0 NEON |
| --------------------------- | ---------: | ---------: | ---------: |
| 1592 × 1124 (session size)  |    3.39 ms |    2.43 ms |    2.07 ms |
| 1920 × 1080 (1080p)         |    3.71 ms |    2.78 ms |    2.38 ms |
| 3840 × 2160 (4K)            |   14.43 ms |   10.80 ms |    9.10 ms |

### Intel Xeon (Cascade Lake) x86_64

| Resolution                  | 4:4:4 AVX2 | 4:4:4 scalar | 4:2:2 AVX2 | 4:2:2 scalar | 4:2:0 AVX2 | 4:2:0 scalar |
| --------------------------- | ---------: | -----------: | ---------: | -----------: | ---------: | -----------: |
| 1592 × 1124 (session size)  |   14.70 ms |     31.31 ms |   11.04 ms |     30.57 ms |    9.64 ms |     28.16 ms |
| 1920 × 1080 (1080p)         |   16.71 ms |     35.50 ms |   12.41 ms |     35.21 ms |   11.15 ms |     32.71 ms |
| 3840 × 2160 (4K)            |   65.16 ms |    143.74 ms |   51.52 ms |    143.49 ms |   43.88 ms |    130.69 ms |

AVX2 speedup vs `force-scalar` on natural content: 2.21× (4:4:4 4K) →
**2.98×** (4:2:0 4K) — wider than on synthetic XOR because both the
SIMD kernels *and* the sparse fast paths contribute.

### IDCT sparse path — on / off comparison

Toggled by forcing the sparse-detection flags in `arch::backend::dct`
to `false` at build time (revert pattern, not a runtime knob), so the
on/off pair is two binaries with otherwise-identical code.

| Subsampling × size | Apple M NEON on | NEON off | NEON gain | Cascade Lake AVX2 on | AVX2 off | AVX2 gain |
| ------------------ | --------------: | -------: | --------: | -------------------: | -------: | --------: |
| 4:4:4  1592×1124   |         3.39 ms |  4.00 ms |    +15.3% |             14.70 ms | 16.23 ms |    +10.4% |
| 4:4:4  1080p       |         3.71 ms |  4.44 ms |    +16.4% |             16.71 ms | 18.76 ms |    +12.3% |
| 4:4:4  4K          |        14.43 ms | 17.79 ms |    +18.9% |             65.16 ms | 75.43 ms |    +15.8% |
| 4:2:2  1592×1124   |         2.43 ms |  2.86 ms |    +15.0% |             11.04 ms | 12.32 ms |    +11.6% |
| 4:2:2  1080p       |         2.78 ms |  3.27 ms |    +15.0% |             12.41 ms | 14.15 ms |    +14.0% |
| 4:2:2  4K          |        10.80 ms | 13.19 ms |    +18.1% |             51.52 ms | 58.43 ms |    +13.4% |
| 4:2:0  1592×1124   |         2.07 ms |  2.33 ms |    +11.2% |              9.64 ms | 10.90 ms |    +13.1% |
| 4:2:0  1080p       |         2.38 ms |  2.68 ms |    +11.2% |             11.15 ms | 12.36 ms |    +10.9% |
| 4:2:0  4K          |         9.10 ms | 10.55 ms |    +13.7% |             43.88 ms | 49.21 ms |    +12.1% |

The pattern is consistent across both backends: sparse contributes
~11–19% of decode time on natural content, more at 4:4:4 (no chroma
upsample diluting IDCT share) and less at 4:2:0 (chroma upsample
absorbs a larger fraction of the cycle budget so the IDCT
optimization moves a smaller slice of the total). On the synthetic
XOR corpus the same on/off swap shows ≤ 2% drift — entirely noise —
which is why this measurement lives in Section D-natural and not in
Section D. The sparse paths are bit-exact with the regular kernels;
cross-check tests in `tests/decode_x86_64.rs` and
`tests/decode_neon.rs` assert that.

### Combined AC/DC Huffman LUT — on / off comparison

The combined Huffman LUT collapses the AC `(run, size)` lookup and
DC `size` lookup with the magnitude-bits read into a single peek
against a precomputed table, replacing the symbol-decode + magnitude
read split. Toggled by short-circuiting `decode_ac_fast` /
`decode_dc_fast` to `Ok(None)` at build time, so the on/off pair is
two binaries with otherwise-identical code.

| Subsampling × size | Cascade Lake AVX2 on | combined-off | gain  |
| ------------------ | -------------------: | -----------: | ----: |
| 4:4:4  1592×1124   |             14.70 ms |     14.98 ms | +1.9% |
| 4:4:4  1080p       |             16.71 ms |     17.37 ms | +3.9% |
| 4:4:4  4K          |             65.16 ms |     65.24 ms | +0.1% |
| 4:2:2  1592×1124   |             11.04 ms |     11.16 ms | +1.1% |
| 4:2:2  1080p       |             12.41 ms |     12.79 ms | +3.0% |
| 4:2:2  4K          |             51.52 ms |     51.44 ms | −0.2% |
| 4:2:0  1592×1124   |              9.64 ms |      9.61 ms | −0.3% |
| 4:2:0  1080p       |             11.15 ms |     10.99 ms | −1.5% |
| 4:2:0  4K          |             43.88 ms |     44.14 ms | +0.6% |

On Apple M (NEON) the same toggle measured +4.7% at 4K 4:2:0
natural; on Cascade Lake the effect sits at the noise floor across
the matrix. The combined LUT was motivated by the entropy-decode
profile share (37% `decode_symbol` self time, ~50–60% Huffman-related
in aggregate on Apple M), but in practice both backends'
straight-line slow paths run well enough on contemporary branch
predictors that the LUT short-circuit moves the headline only a few
percent at most. The change is kept — it's bit-exact and adds no
runtime cost when the table misses — but the realized speedup is
much smaller than the profile-share-derived ceiling suggested.

### SWAR 32-bit bit-reader refill — on / off comparison

The Huffman `BitReader::fill` path traditionally shifts one byte at
a time into the 64-bit accumulator, branching on every byte for the
`0xFF`-stuffed-by-`0x00` JPEG sequence. The SWAR variant peeks
four bytes ahead, runs a single has-byte-equal-`0xFF` test
(`(y - 0x0101_0101) & !y & 0x8080_8080`), and shifts all four bytes
in via a single 32-bit OR when the test is zero; falls back to the
per-byte path otherwise. Toggle is a build-time revert pair (the
SWAR fast-path is removed and the loop falls back to the per-byte
path), so the on/off pair is two binaries with otherwise-identical
code. Numbers are 4K natural, single-run on Apple M and 5-run median
on Cascade Lake (variance ~1–2% on both):

| Subsampling × size | Apple M NEON on | NEON off | NEON gain | Cascade Lake AVX2 on | AVX2 off | AVX2 gain |
| ------------------ | --------------: | -------: | --------: | -------------------: | -------: | --------: |
| 4:4:4  4K          |        13.92 ms | 15.00 ms |    +7.2%  |             72.55 ms | 76.86 ms |    +5.6%  |
| 4:2:2  4K          |        10.57 ms | 11.27 ms |    +6.2%  |             57.66 ms | 60.34 ms |    +4.4%  |
| 4:2:0  4K          |         8.72 ms |  9.41 ms |    +7.3%  |             49.37 ms | 52.00 ms |    +5.1%  |

The SWAR variant is **cross-arch consistent +4–7%** on natural 4K
content. Unlike the combined LUT (which sits at the noise floor at
q=80), the SWAR refill amortizes the per-byte branch overhead on
the dominant non-stuffed path and shows up in real measurements on
both NEON and AVX2 hosts. The Cascade Lake numbers here were
collected on a separate run from the Section D-natural matrix above
(different VM allocation), so the baseline absolute ms don't line
up exactly with the 43.88 ms in that table — read the *gain*
column rather than comparing raw ms across subsections. Bit-exact equivalence to the per-byte path is
asserted by the existing decode cross-check tests.

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

Both corpora are bench-driven from `tests/comparison_bench.rs`. The
*synthetic* row is the same XOR pattern used everywhere else in this
document — every block is full-AC, which is the Huffman-heavy
worst case for both decoders. The *natural-content* row drives the
same harness with `make_natural_image` (smooth sky + low-AC texture +
edge bars), the corpus introduced for Section D-natural; it is the
fairer proxy for typical web/photo input.

| Host                  | Corpus      | Resolution | ours       | image     | ratio (img/ours) |
| --------------------- | ----------- | ---------- | ---------: | --------: | ---------------: |
| Apple M-series (NEON) | synthetic   | 1592×1124  |    4.27 ms |   4.31 ms |            1.01× |
|                       | synthetic   | 1920×1080  |    4.81 ms |   4.89 ms |            1.02× |
|                       | synthetic   | 3840×2160  |   20.20 ms |  19.58 ms |            0.97× |
|                       | natural     | 1592×1124  |    1.98 ms |   1.52 ms |            0.77× |
|                       | natural     | 1920×1080  |    2.24 ms |   1.82 ms |            0.81× |
|                       | natural     | 3840×2160  |    8.67 ms |   6.80 ms |            0.78× |
| Cascade Lake (AVX2)   | synthetic   | 1592×1124  |   17.26 ms |  12.80 ms |            0.74× |
|                       | synthetic   | 1920×1080  |   19.99 ms |  15.06 ms |            0.75× |
|                       | synthetic   | 3840×2160  |   81.62 ms |  69.73 ms |            0.85× |
|                       | natural     | 1592×1124  |    9.15 ms |   4.92 ms |            0.54× |
|                       | natural     | 1920×1080  |   10.32 ms |   5.71 ms |            0.55× |
|                       | natural     | 3840×2160  |   45.60 ms |  34.27 ms |            0.75× |

(ratio > 1 means jpeg-rusturbo is faster)

Reading the numbers honestly:

- On synthetic Huffman-heavy content we now sit at parity on Apple M
  and ~0.74–0.85× on Cascade Lake. The 0.6.0 number on Apple M was
  0.77×; the 0.7.0 sparse-parity + SWAR refill changes do show up
  here even though the corpus is "worst case" for the per-pixel
  parts of the pipeline, because both optimizations also benefit
  the dominant non-stuffed Huffman path.
- On natural-content input `image` (jpeg-decoder) pulls further
  ahead, ending the 0.7.0 cycle at ~0.54–0.78×. The absolute decode
  time on our side drops by ~2× from the synthetic row to the
  natural row — sparse IDCT and the LUT/SWAR Huffman path do fire —
  but `image` drops by ~3× over the same corpus, so the *ratio*
  widens rather than narrowing. The remaining gap is in the parts
  of the entropy decoder we have not yet rewritten (combined LUT
  coverage of edge-case symbols, AC band-loop scheduling) and in
  the colour-convert / upsample stages that benefit more from
  jpeg-decoder's batch shape than from ours. This is the honest
  read; future work is queued in the 0.8.0 plan.

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

Decode side (per-stage SIMD landed in 0.6.0):
  Entropy decode        ~35%   scalar by design (serial code-length
                                dependency); 0.7.0 adds combined
                                AC/DC LUT + SWAR 32-bit refill
                                (+4–7% on natural across NEON/AVX2)
  IDCT                  ~25%   NEON ~2.0× / AVX2 ~2.5× / scalar 1.0×
                                + DC-only / sparse-row fast paths
                                  (NEON 0.6.0, AVX2 0.7.0; +11–19% of
                                   total decode on natural content,
                                   ≈ 0% on synthetic — see Section
                                   D-natural)
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
- Vector-SIMD Huffman *decode*. The bit reader + canonical-Huffman
  table walk has a serial dependency on per-symbol code length, so
  vectorizing across symbols isn't tractable. The optimizations that
  do help are scalar bit-ops: 0.7.0 lands a combined AC/DC LUT
  (table-driven path, used by baseline and progressive scans) and a
  SWAR 32-bit bit-reader refill — see the SWAR on/off and combined
  LUT on/off subsections under Section D-natural.
