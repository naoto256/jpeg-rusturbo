# `jpeg-rusturbo` benchmarks

All numbers below come from one coherent measurement campaign per host,
driven by the harness in `benches/pipeline.rs` (encode + decode) and
`benches/vs_image.rs` (vs the `image` crate). Encode-side numbers were
refreshed on 2026-06-08 after the 0.9.x encoder hot-path work. Decode
numbers are retained from the previous campaign because the decoder code
was not changed in this cycle. The rightmost column is always the
multiplier (speedup or ratio): scan it down a column and the win reads at
a glance.

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
are with that profile. Pipeline encode timings use the 4-byte RGBA input
path so SIMD vs scalar and thread-scaling rows isolate backend work on
the same byte layout. The vs-`image` comparison feeds 3-byte RGB to both
encoders, the fair input for `image`; after the 0.9.x x86 RGB hot-path
work those RGB rows can be materially faster than the RGBA pipeline
rows, especially at 4:4:4. Output bytes are byte-identical across SIMD ↔
scalar, across hosts, and across RGB ↔ RGBA input — the bit-exact
equivalence the crate sets out to preserve, asserted in the unit tests.

## Hosts

| Label                     | CPU                                | Cores  | SIMD floor       |
| ------------------------- | ---------------------------------- | ------ | ---------------- |
| Apple M-series            | Apple Silicon (M-series)           | 8 P+E  | NEON (always-on) |
| Intel Xeon (Cascade Lake) | Xeon Platinum 8272CL @ 2.60 GHz    | 4 vCPU | AVX2 + SSE2      |

The Cascade Lake host had 4 vCPUs. The Apple M host was a developer
laptop on AC power; background load was light but non-zero, so on that
host trust the *ratios* a little more than the raw ms.

A note on the vs-`image` ratios (third chapter): they reflect *both* our
speed and `image`'s scalar-encoder baseline on that CPU. `image`'s
encoder is markedly slower on the Cascade Lake host than on the Apple M
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
| 4:4:4            | 28.09 ms | 54.12 ms |   1.93× |
| 4:2:2            | 19.39 ms | 39.03 ms |   2.01× |
| 4:2:0            | 16.43 ms | 35.31 ms |   2.15× |

**Intel Xeon (Cascade Lake, AVX2)**

| Subsampling (4K) | SIMD      | scalar    | speedup |
| ---------------- | --------: | --------: | ------: |
| 4:4:4            | 142.22 ms | 274.21 ms |   1.93× |
| 4:2:2            |  81.78 ms | 207.08 ms |   2.53× |
| 4:2:0            |  62.58 ms | 160.79 ms |   2.57× |

The win grows toward 4:2:0 as chroma downsampling shrinks the planes the
SIMD color + DCT kernels run over. 0.8.0's encoder cycle added an unsafe
`BitWriter::drain_high32`, fused AC code+magnitude `write_bits`, a NEON
`vqtbl4q` zig-zag scatter, a NEON magnitude-category precompute, and an
AVX2 3-byte RGB→YCbCr deinterleave kernel. The 0.9.x hot-path pass then
batched the x86 RGB front half for 4:2:0 / 4:2:2 / 4:4:4, bringing the
Cascade Lake 4:4:4 row up to the same shape as the denser chroma modes.

## Thread scaling

4:2:0, SIMD build. `set_threads(n)` partitions MCU rows across a rayon
pool; `auto` picks `available_parallelism()`. Output bytes are identical
across thread counts.

**Apple M-series (8 cores)**

| Resolution | t=1      | t=2      | t=4     | t=8     | auto    | auto vs t=1 |
| ---------- | -------: | -------: | ------: | ------: | ------: | ----------: |
| 1080p      |  3.98 ms |  2.78 ms | 2.20 ms | 2.08 ms | 2.09 ms |       1.90× |
| 4K         | 16.34 ms | 10.81 ms | 8.60 ms | 7.89 ms | 7.35 ms |       2.22× |

**Intel Xeon (Cascade Lake, 4 vCPU)**

| Resolution | t=1      | t=2      | t=4      | t=8      | auto     | auto vs t=1 |
| ---------- | -------: | -------: | -------: | -------: | -------: | ----------: |
| 1080p      | 16.20 ms | 14.16 ms | 15.42 ms | 15.79 ms | 15.31 ms |       1.06× |
| 4K         | 64.88 ms | 55.42 ms | 54.65 ms | 53.62 ms | 53.29 ms |       1.22× |

The 4-vCPU Cascade Lake saturates by 2–4 threads and barely benefits at
1080p after the single-thread hot path was tightened; the 8-core Apple
host still scales past 2× at 4K.

## Optimize-huffman

`set_optimize_huffman(true)` adds a second pass (count frequencies, build
canonical T.81 K.2/K.3 tables, re-emit). The size reduction is
host-independent (identical bytes everywhere), so it's one table.

| Subsampling | q70    | q80    | q90    |
| ----------- | -----: | -----: | -----: |
| 4:4:4       | −5.71% | −5.69% | −5.63% |
| 4:2:2       | −5.50% | −5.34% | −5.29% |
| 4:2:0       | −5.25% | −4.87% | −4.61% |

Roughly −5% on synthetic content; 4–10% on natural photos, matching
`cjpeg -optimize`. The second pass roughly doubles encode wall-clock —
4K 4:2:0 q=80 costs **2.31×** on Apple M (16.34 → 37.75 ms) and **1.70×**
on Cascade Lake (64.86 → 110.53 ms); the factor is larger on NEON
because its faster first pass leaves the second, largely scalar, pass
weighing more. Opt-in for when bandwidth matters more than CPU.

## Progressive (cost of SOF2)

This one is a *cost*, not a speedup: `set_progressive(true)` emits SOF2
instead of baseline. 4:2:0, natural-like content, SIMD build. Rightmost
columns are the time and size multipliers over baseline.

**Apple M-series (NEON)**

| Resolution | baseline | progressive | time  | size  |
| ---------- | -------: | ----------: | ----: | ----: |
| 1592×1124  |  2.32 ms |     6.23 ms | 2.69× | 1.43× |
| 1080p      |  2.64 ms |     7.01 ms | 2.66× | 1.44× |
| 4K         | 13.01 ms |    29.07 ms | 2.23× | 1.48× |

**Intel Xeon (Cascade Lake, AVX2)**

| Resolution | baseline | progressive | time  | size  |
| ---------- | -------: | ----------: | ----: | ----: |
| 1592×1124  |  7.40 ms |    21.41 ms | 2.89× | 1.43× |
| 1080p      |  7.98 ms |    24.41 ms | 3.06× | 1.44× |
| 4K         | 33.25 ms |    97.90 ms | 2.94× | 1.48× |

The ~2.5–3.0× time is spec-bound (one buffer pass + eight scan passes over
the stored coefficients). The +43–48% size is on us, not the spec: the
standard-tables encoder emits `EOB0` per block rather than the
multi-block `EOBn` runs the format permits, because the Annex K
reference Huffman tables carry no `EOBn` codes for `n ≥ 1`.

As of 0.9.0, `set_optimize_huffman(true)` composes with
`set_progressive(true)` to close this: a two-pass encode counts symbol
frequencies per scan, builds per-scan custom Huffman tables that
include `EOBn` codes, emits one DHT per scan, and packs multi-block
end-of-band runs. The optimized-progressive output ends up **smaller**
than the corresponding baseline SOF0 — natural-like 4:2:0 q=80 at 4K
drops from 364 KB (standard-tables progressive, +48% vs baseline) to
~148 KB (**−40% vs the 246 KB baseline**); 1080p and 1592×1124 land at
−37%. Default (without `set_optimize_huffman`) is unchanged from the
table above. Size factor is content-shaped and host-independent.

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
4K 4:2:2 / 4:4:4 rows are reference (denser chroma, more work). The
same harness prints both `threads=1` and `threads=auto` for our encoder;
`image` has no comparable threading knob.

### threads=1

**Apple M-series (NEON)**

| Subsampling / resolution | jpeg-rusturbo | image    | ratio |
| ------------------------ | ------------: | -------: | ----: |
| 4:2:0  1592×1124         |  3.06 ms | 15.58 ms | 5.10× |
| 4:2:0  1080p             |  3.49 ms | 17.72 ms | 5.08× |
| 4:2:0  4K                | 13.41 ms | 67.61 ms | 5.04× |
| 4:2:2  4K                | 17.38 ms | 67.77 ms | 3.90× |
| 4:4:4  4K                | 24.56 ms | 67.67 ms | 2.76× |

**Intel Xeon (Cascade Lake, AVX2)**

| Subsampling / resolution | jpeg-rusturbo | image     | ratio |
| ------------------------ | ------------: | --------: | ----: |
| 4:2:0  1592×1124         |  12.73 ms |  70.04 ms | 5.50× |
| 4:2:0  1080p             |  14.57 ms |  80.75 ms | 5.54× |
| 4:2:0  4K                |  57.27 ms | 320.93 ms | 5.60× |
| 4:2:2  4K                |  75.01 ms | 320.91 ms | 4.28× |
| 4:4:4  4K                | 100.19 ms | 321.26 ms | 3.21× |

Single-thread, we lead `image`'s encoder by **~5×** on Apple M and
**5.5–5.6×** on Cascade Lake at 4:2:0.

### threads=auto

**Apple M-series (NEON)**

| Subsampling / resolution | jpeg-rusturbo | image    | ratio |
| ------------------------ | ------------: | -------: | ----: |
| 4:2:0  1592×1124         |  1.76 ms | 15.98 ms |  9.10× |
| 4:2:0  1080p             |  1.89 ms | 17.91 ms |  9.49× |
| 4:2:0  4K                |  6.83 ms | 68.47 ms | 10.02× |
| 4:2:2  4K                |  9.99 ms | 68.82 ms |  6.89× |
| 4:4:4  4K                | 14.13 ms | 68.49 ms |  4.85× |

**Intel Xeon (Cascade Lake, AVX2)**

| Subsampling / resolution | jpeg-rusturbo | image     | ratio |
| ------------------------ | ------------: | --------: | ----: |
| 4:2:0  1592×1124         |  10.28 ms |  70.03 ms | 6.81× |
| 4:2:0  1080p             |  12.29 ms |  80.75 ms | 6.57× |
| 4:2:0  4K                |  46.77 ms | 320.95 ms | 6.86× |
| 4:2:2  4K                |  61.73 ms | 321.28 ms | 5.20× |
| 4:4:4  4K                |  85.10 ms | 320.83 ms | 3.77× |

With `threads=auto`, the practical 4:2:0 lead rises to **9–10×** on
Apple M and **6.6–6.9×** on Cascade Lake. The Cascade ratio is higher
than its single-thread ratio but does not jump as far as Apple because
this 4-vCPU host already saturates early in the thread-scaling table.
The x86 RGB front half now has batched hot paths for all three
RGB-family subsampling modes, which is why 4:4:4 no longer falls back to
the old scalar-color shape.

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

A rough hot-path breakdown for single-thread 4K 4:2:0 q=80 encode. The
current sample was taken on Apple M with a tight 45-second loop around
`encode_rgba` and `sample` at 1 ms intervals. `cargo flamegraph` could
not run on this machine because macOS `xctrace` requires full Xcode, so
the numbers below are top-of-stack sample buckets rather than a committed
SVG flamegraph. LTO inlines the DCT / quantize front half heavily, so
that work appears in the residual inlined bucket rather than as separate
symbols.

```
Encode side:
  Huffman encode_block       ~65%  scalar entropy emit + SIMD nonzero bitmap
  Color / downsample         ~22%  extract_mcu_420 + NEON rgb_row_to_ycc
  Inlined DCT / quantize /
    zig-zag / MCU plumbing   ~12%  mostly inlined by LTO into caller
  Flush / marker / memcpy    < 1%  not a meaningful share in this sample

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

After the 0.9.x encoder front-half work, whole-pipeline encode is now
mostly Amdahl-bound on the scalar entropy emitter. The SIMD color / DCT /
quantize bodies still matter, but further front-half wins have less room
to move the 4:2:0 single-thread needle unless they also reduce the number
or cost of entropy-coded coefficients. Decode remains bounded by its
scalar entropy walk plus the per-stage SIMD kernels described above.

# Out of scope

Two buckets: things tried and measured not worth it, and things never
planned. The first bucket is recorded so it isn't re-investigated every
cycle.

## Investigated — measured no gain, not pursued

- **AVX2 zig-zag scatter** (encode). NEON ships a `vqtbl4q` (4-register,
  64-byte table-lookup) zig-zag scatter; AVX2 has no equivalent — the
  reorder needs clunky cross-128-bit-lane word permutes, while the x86
  scalar zig-zag is already cheap (it stays hot in L1 / the store
  buffer). Low-to-no expected gain; left scalar on x86. The current x86
  4:4:4 encode gains come from RGB front-half batching rather than
  zig-zag scatter.
- **AVX2 `ac_magnitudes` precompute** (encode). Ported and bit-exact,
  but measured a *wash* (≤~1.5%, within run-to-run noise) on a Broadwell
  Xeon. AVX2 lacks per-16-bit-lane CLZ and 16-bit variable shift
  (`vpsllvw` is AVX-512), so synthesizing them (float-exponent
  bit-length + 32-bit-widen `vpsllvd` mask) costs about what it saves;
  the kernel isn't a hot-enough share of encode self-time, and LLVM
  already auto-vectorizes the scalar. Left scalar on x86.
- **Decode dequantize skip-zero + zig-zag-folded quant table.** Profiled
  as a possible +2–3 ms; a same-thermal A/B showed a *net regression* on
  both backends (NEON auto-vectorizes the original loop; the x86
  variant's branch overhead offset the SSE pack). Reverted. (Distinct
  from the dequant→entropy-loop fusion that did ship in 0.7.5.)
- **Combined AC/DC Huffman LUT as a *speed* play.** It sits at the noise
  floor at q=80 (the common case). Retained as a bit-exact,
  zero-cost-on-miss, table-driven foundation — the canonical approach in
  libjpeg-turbo / zune-jpeg — but it is not a headline speedup.

General rule learned: an encode-side micro-kernel below a few percent of
self-time rarely beats LLVM-auto-vectorized scalar once you pay the
ISA-synthesis cost for a NEON op that has no AVX2 equivalent.

## By design (never planned)

- **AVX-512 / SVE** versions of the kernels. The x86 server market is
  bifurcated (Zen 4 yes, Zen 2/3 no, Alder Lake P-cores bin-disabled);
  AVX2 stays the x86_64 floor, NEON the aarch64 baseline.
- **Full SIMD AC-symbol emission** (encode). The nonzero bitmap is SIMD;
  per-coefficient emission stays scalar — LLVM autovectorizes the
  bit-writer drain and the table lookups don't reshape cleanly into SIMD.
- **Vector-SIMD Huffman decode.** The bit reader + canonical-table walk
  has a serial per-symbol code-length dependency, so vectorizing across
  symbols isn't tractable. The wins are scalar bit-ops: the combined
  AC/DC LUT and the SWAR 32-bit refill.
