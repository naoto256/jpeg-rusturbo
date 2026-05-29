# `jpeg-rusturbo` benchmarks

All numbers below come from one coherent measurement campaign per host
on 2026-05-30, driven by the harness in `benches/pipeline.rs` (encode +
decode) and `benches/vs_image.rs` (vs the `image` crate). Both
hosts ran the identical command sequence, so every table appears as a
matched pair — one per host, same shape. The rightmost column is always
the multiplier (speedup or ratio): scan it down a column and the win
reads at a glance.

Methodology is intentionally simple: a single timed batch, 50 measured
iterations after a 3-iteration warm-up, no statistical filtering. Two
synthetic corpora feed the harness:

- **synthetic** (`make_image`): an XOR/multiply pattern where every 8×8
  block is full-AC. This keeps the entropy coder honest and the IDCT
  sparse fast paths cold — the Huffman-heavy worst case for both encode
  and decode. Pixel-identical across hosts.
- **natural-like** (`make_natural_image`): ~70% smooth gradient (sky /
  wall), ~20% low-AC texture, ~10% sharp edges. After quantize this
  produces the DC-dominant + low-AC block mix of real photographic
  content, and is the fairer proxy for typical web/photo input.

Reproduce the whole campaign on any host with three commands:

```sh
cargo bench --bench pipeline -- --section all                    # SIMD build
cargo bench --bench pipeline --features force-scalar -- --section all  # scalar fallback
cargo bench --bench vs_image                                     # vs image crate
```

The release profile is `lto = "fat"` + `codegen-units = 1`; all numbers
are with that profile. Encode timings use the 4-byte RGBA input path
(the 3-byte RGB path tracks within ~1–2% on both backends as of 0.8.0);
the vs-`image` comparison feeds 3-byte RGB to both encoders, the fair
input for `image`. Output bytes are byte-identical across SIMD ↔ scalar,
across hosts, and across RGB ↔ RGBA input — the bit-exact equivalence
the crate sets out to preserve, asserted in the unit tests.

## Hosts

| Label                     | CPU                                | Cores  | SIMD floor       |
| ------------------------- | ---------------------------------- | ------ | ---------------- |
| Apple M-series            | Apple Silicon (M-series)           | 8 P+E  | NEON (always-on) |
| Intel Xeon (Cascade Lake) | Xeon Platinum 8272CL @ 2.60 GHz    | 4 vCPU | AVX2 + SSE2      |

The Cascade Lake host was a 4-vCPU Azure `D4s_v3`. The Apple M host was
a developer laptop on AC power; background load was light but non-zero,
so on that host trust the *ratios* a little more than the raw ms.

A note on the vs-`image` ratios (third chapter): they reflect *both* our
speed and `image`'s scalar-encoder baseline on that CPU. `image`'s
encoder is markedly slower on the Cascade Lake VM than on the Apple M
laptop, so the Cascade ratios run higher even though our absolute encode
is slower there. Read each ratio as "vs `image` on this host," not as a
cross-host hardware comparison.

---

# Encode

`q=80`, single thread unless noted. All times ms/iter.

## SIMD vs scalar

SIMD build vs `--features force-scalar`, at 4K. Isolates the per-backend
kernel win. The speedup is essentially scale-invariant (it tracks
per-pixel work), so the 4K row stands in for every resolution.

**Apple M-series (NEON)**

| Subsampling (4K) | SIMD     | scalar   | speedup |
| ---------------- | -------: | -------: | ------: |
| 4:4:4            | 29.17 ms | 55.37 ms |   1.90× |
| 4:2:2            | 19.69 ms | 40.34 ms |   2.05× |
| 4:2:0            | 16.06 ms | 37.27 ms |   2.32× |

**Intel Xeon (Cascade Lake, AVX2)**

| Subsampling (4K) | SIMD      | scalar    | speedup |
| ---------------- | --------: | --------: | ------: |
| 4:4:4            | 145.80 ms | 273.37 ms |   1.88× |
| 4:2:2            |  83.82 ms | 205.57 ms |   2.45× |
| 4:2:0            |  65.69 ms | 160.30 ms |   2.44× |

The win grows toward 4:2:0 as chroma downsampling shrinks the planes the
SIMD color + DCT kernels run over. 0.8.0's encoder cycle added an unsafe
`BitWriter::drain_high32`, fused AC code+magnitude `write_bits`, a NEON
`vqtbl4q` zig-zag scatter, a NEON magnitude-category precompute, and an
AVX2 3-byte RGB→YCbCr deinterleave kernel.

## Thread scaling

4:2:0, SIMD build. `set_threads(n)` partitions MCU rows across a rayon
pool; `auto` picks `available_parallelism()`. Output bytes are identical
across thread counts.

**Apple M-series (8 cores)**

| Resolution | t=1      | t=2      | t=4     | t=8     | auto    | auto vs t=1 |
| ---------- | -------: | -------: | ------: | ------: | ------: | ----------: |
| 1080p      |  3.90 ms |  2.63 ms | 2.08 ms | 1.96 ms | 1.90 ms |       2.05× |
| 4K         | 16.14 ms | 10.03 ms | 7.92 ms | 7.02 ms | 6.85 ms |       2.36× |

**Intel Xeon (Cascade Lake, 4 vCPU)**

| Resolution | t=1      | t=2      | t=4      | t=8      | auto     | auto vs t=1 |
| ---------- | -------: | -------: | -------: | -------: | -------: | ----------: |
| 1080p      | 16.69 ms | 13.70 ms | 14.40 ms | 14.69 ms | 13.71 ms |       1.22× |
| 4K         | 66.23 ms | 54.72 ms | 56.11 ms | 53.39 ms | 53.60 ms |       1.24× |

The 4-vCPU Cascade Lake saturates at 2 threads; the 8-core Apple host
reaches ~2.4× at 4K.

## Optimize-huffman

`set_optimize_huffman(true)` adds a second pass (count frequencies, build
canonical T.81 K.2/K.3 tables, re-emit). The size reduction is
host-independent (identical bytes everywhere), so it's one table.

| Subsampling | q70    | q80    | q90    |
| ----------- | -----: | -----: | -----: |
| 4:4:4       | −5.74% | −5.67% | −5.62% |
| 4:2:2       | −5.53% | −5.31% | −5.32% |
| 4:2:0       | −5.29% | −4.89% | −4.58% |

Roughly −5% on synthetic content; 4–10% on natural photos, matching
`cjpeg -optimize`. The second pass roughly doubles encode wall-clock —
4K 4:2:0 q=80 costs **2.28×** on Apple M (16.53 → 37.67 ms) and **1.71×**
on Cascade Lake (65.59 → 112.15 ms); the factor is larger on NEON
because its faster first pass leaves the second, largely scalar, pass
weighing more. Opt-in for when bandwidth matters more than CPU.

## Progressive (cost of SOF2)

This one is a *cost*, not a speedup: `set_progressive(true)` emits SOF2
instead of baseline. 4:2:0, natural-like content, SIMD build. Rightmost
columns are the time and size multipliers over baseline.

**Apple M-series (NEON)**

| Resolution | baseline | progressive | time  | size  |
| ---------- | -------: | ----------: | ----: | ----: |
| 1592×1124  |  2.34 ms |     6.06 ms | 2.59× | 1.43× |
| 1080p      |  2.67 ms |     6.95 ms | 2.60× | 1.44× |
| 4K         | 11.22 ms |    27.66 ms | 2.46× | 1.48× |

**Intel Xeon (Cascade Lake, AVX2)**

| Resolution | baseline | progressive | time  | size  |
| ---------- | -------: | ----------: | ----: | ----: |
| 1592×1124  |  7.56 ms |    20.02 ms | 2.65× | 1.43× |
| 1080p      |  8.44 ms |    23.05 ms | 2.73× | 1.44× |
| 4K         | 34.64 ms |    92.16 ms | 2.66× | 1.48× |

The ~2.5× time is spec-bound (one buffer pass + eight scan passes over
the stored coefficients). The +43–48% size is on us, not the spec: the
encoder emits `EOB0` per block rather than the multi-block `EOBn` runs
the format permits, because the Annex K reference Huffman tables this
crate ships carry no `EOBn` codes for `n ≥ 1`. A future
`encode_progressive_optimize` (optimized-Huffman for SOF2) could derive
tables with the `EOBn` symbols and recover the size. Size factor is
content-shaped and host-independent.

---

# Decode

The decoder was not reopened in 0.8.0 (an encoder cycle); these are the
0.7.x decode kernels measured fresh on the new harness. `q=80`, 4K.

## SIMD vs scalar

SIMD build vs `--features force-scalar`, on both corpora. The gap between
synthetic and natural is the whole point of the sparse / combined-LUT /
SWAR work: on synthetic (every block full-AC) only the raw SIMD kernels
fire; on natural the sparse fast paths fire on top.

**Apple M-series (NEON)**

| Corpus (4K)      | SIMD     | scalar   | speedup |
| ---------------- | -------: | -------: | ------: |
| synthetic 4:4:4  | 33.17 ms | 49.52 ms |   1.49× |
| synthetic 4:2:0  | 18.20 ms | 36.81 ms |   2.02× |
| natural   4:4:4  |  8.04 ms | 27.16 ms |   3.38× |
| natural   4:2:0  |  5.75 ms | 26.24 ms |   4.56× |

**Intel Xeon (Cascade Lake, AVX2)**

| Corpus (4K)      | SIMD      | scalar    | speedup |
| ---------------- | --------: | --------: | ------: |
| synthetic 4:4:4  | 104.00 ms | 174.08 ms |   1.67× |
| synthetic 4:2:0  |  58.13 ms | 139.44 ms |   2.40× |
| natural   4:4:4  |  32.41 ms | 110.19 ms |   3.40× |
| natural   4:2:0  |  22.51 ms | 107.98 ms |   4.80× |

Natural-content decode reaches ~4.5–4.8× scalar at 4:2:0 4K — most
blocks have only a few low-frequency coefficients, and the sparse paths
skip the rest. On synthetic input the same kernels only manage ~2×.

---

# vs `image`

`image` (`v0.25`) is the de-facto Rust image library: a scalar JPEG
encoder hardcoded to 4:2:0, and `zune-jpeg` (SIMD) for decode. Both
sides timed on the same content from the same harness, 3-byte RGB in.

## Encode (RGB → JPEG, q=80)

`image` only emits 4:2:0, so that's the apples-to-apples ratio; our
4K 4:2:2 / 4:4:4 rows are reference (denser chroma, more work).

**Apple M-series (NEON)**

| Subsampling / resolution | jpeg-rusturbo | image    | ratio |
| ------------------------ | ------------: | -------: | ----: |
| 4:2:0  1592×1124         |  3.50 ms | 15.79 ms | 4.51× |
| 4:2:0  1080p             |  3.94 ms | 17.92 ms | 4.55× |
| 4:2:0  4K                | 15.64 ms | 69.07 ms | 4.42× |
| 4:2:2  4K                | 20.41 ms | 68.89 ms | 3.38× |
| 4:4:4  4K                | 30.33 ms | 69.05 ms | 2.28× |

**Intel Xeon (Cascade Lake, AVX2)**

| Subsampling / resolution | jpeg-rusturbo | image     | ratio |
| ------------------------ | ------------: | --------: | ----: |
| 4:2:0  1592×1124         |  14.27 ms |  76.90 ms | 5.39× |
| 4:2:0  1080p             |  16.19 ms |  88.62 ms | 5.48× |
| 4:2:0  4K                |  63.40 ms | 352.09 ms | 5.55× |
| 4:2:2  4K                |  81.62 ms | 352.08 ms | 4.31× |
| 4:4:4  4K                | 144.91 ms | 352.08 ms | 2.43× |

We lead `image`'s encoder by **4.4–4.6×** on Apple M and **5.4–5.6×** on
Cascade Lake at 4:2:0. The Cascade ratio is higher because `image`'s
scalar encoder is much slower on that CPU (see the cross-host note up
top), and because 0.8.0's AVX2 3-byte RGB→YCbCr kernel removed the
scalar-color fallback that previously capped the x86 RGB-input ratio at
~3.7×.

## Decode (JPEG → RGB, our encoder's q=80 4:2:0 output)

**Apple M-series (NEON)**

| Corpus / resolution  | jpeg-rusturbo | image    | ratio |
| -------------------- | ------------: | -------: | ----: |
| synthetic 1592×1124  |  4.22 ms |  4.39 ms | 1.04× |
| synthetic 1080p      |  4.59 ms |  5.03 ms | 1.10× |
| synthetic 4K         | 19.06 ms | 19.65 ms | 1.03× |
| natural   1592×1124  |  1.35 ms |  1.55 ms | 1.15× |
| natural   1080p      |  1.55 ms |  1.86 ms | 1.20× |
| natural   4K         |  5.93 ms |  7.00 ms | 1.18× |

**Intel Xeon (Cascade Lake, AVX2)**

| Corpus / resolution  | jpeg-rusturbo | image    | ratio |
| -------------------- | ------------: | -------: | ----: |
| synthetic 1592×1124  | 13.12 ms | 12.90 ms | 0.98× |
| synthetic 1080p      | 14.94 ms | 15.04 ms | 1.01× |
| synthetic 4K         | 62.94 ms | 69.11 ms | 1.10× |
| natural   1592×1124  |  5.21 ms |  5.07 ms | 0.97× |
| natural   1080p      |  5.83 ms |  5.97 ms | 1.02× |
| natural   4K         | 28.01 ms | 34.26 ms | 1.22× |

(ratio > 1 means jpeg-rusturbo is faster.)

The decoder is at rough parity with `zune-jpeg`: ahead at 4K on both
microarchitectures (~1.1× synthetic, ~1.18–1.22× natural), within noise
at smaller resolutions, and within ~2–3% at 1592×1124 on Cascade Lake
where our per-decode setup (header parse + plane allocations) is a larger
fraction of total time. The decoder is bundled for API symmetry, not as
a speed play; this is a "we don't regress on read-back" result, not the
headline. The encoder is.

---

# Where the time goes

A rough per-stage breakdown for 4K 4:2:0 q=80 (estimated from
`cargo flamegraph`, not committed to the repo):

```
Encode side:
  Color / downsample    ~25%   NEON ~3.0× / AVX2 ~3.0× / scalar 1.0×
  Forward DCT           ~20%   NEON ~2.5× / AVX2 ~2.7× / scalar 1.0×
  Quantize + zig-zag    ~10%   NEON ~1.8× / AVX2 ~2.5× / scalar 1.0×
  Huffman (64-bit acc + bitmap)  ~30%   NEON/SSE2 bitmap ~1.4×
  Marker writes / IO    ~15%   scalar in both

Decode side (per-stage SIMD landed in the 0.6.0 / 0.7.x decoder cycles):
  Entropy decode        ~35%   combined AC/DC LUT + SWAR 32-bit refill
                                (+4–7% on natural across NEON/AVX2);
                                otherwise serial by code-length dependency
  IDCT                  ~25%   NEON ~2.0× / AVX2 ~2.5× + DC-only / sparse-row
                                fast paths (~11–19% of decode on natural,
                                ≈0% on synthetic — visible as the
                                synthetic-vs-natural gap in Decode above)
  Color convert (YCC→RGB) ~20% NEON ~6.7× / AVX2 ~3.5×
  Chroma upsample fancy ~15%   NEON ~1.3–14× (kernel-dep) / AVX2 similar
  Marker walk / IO      ~5%    scalar in both
```

Per-stage SIMD bodies hit close to expected speedups; whole-pipeline
numbers are Amdahl-bound on the partly-scalar entropy emit/decode and the
serial marker / IO sections.

# Out of scope

- **AVX-512** versions of the kernels. The server market is bifurcated
  (Zen 4 yes, Zen 2/3 no, Alder Lake P-cores bin-disabled); AVX2 stays
  the x86_64 floor for this crate.
- **AVX2 zig-zag scatter + magnitude precompute** on the encode side.
  These two 0.8.0 hot-path items are NEON-only so far, which is why the
  x86 encode SIMD-vs-scalar ratio trails NEON's slightly at 4:4:4. AVX2
  equivalents (zig-zag via `vpermd` / `vpshufb`, magnitude lookup via a
  PSHUFB bit-length table) are candidates for a later cycle.
- **Full SIMD AC-symbol emission** on the encoder side. The nonzero
  bitmap is SIMD; the per-coefficient emission stays scalar — it's tight
  enough that LLVM autovectorizes the bit-writer drain and the table
  lookups don't reshape cleanly into SIMD.
- **Vector-SIMD Huffman decode.** The bit reader + canonical-table walk
  has a serial dependency on per-symbol code length, so vectorizing
  across symbols isn't tractable. The optimizations that do help are
  scalar bit-ops: the combined AC/DC LUT and the SWAR 32-bit refill.
