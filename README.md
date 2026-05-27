# jpeg-rusturbo

[![CI](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/ci.yml/badge.svg)](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/ci.yml)
[![Release](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/release.yml/badge.svg)](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/release.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**SIMD-accelerated baseline JPEG encoder + Huffman (baseline +
progressive) decoder with libjpeg-turbo-derived kernels.** Both
sides ship NEON-on-aarch64 and AVX2-on-x86_64 kernels; non-SIMD
targets fall through to a bit-exact scalar reference. Drop-in for
`image::codecs::jpeg::JpegEncoder` on the encode side; standalone
decoder under `jpeg_rusturbo::decode`.

```rust
use jpeg_rusturbo::{JpegEncoder, PixelFormat, decode};

// Encode
let mut out = Vec::new();
let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
enc.encode_rgba(&pixels, width, height)?;

// Decode (back to RGB bytes)
let rgb = decode::decode(&out, PixelFormat::Rgb)?;
```

The encoder ships SIMD kernels for color convert, FDCT, quantize +
zig-zag, chroma downsample, and Huffman nonzero bitmap. The decoder
reads baseline (SOF0) and progressive (SOF2) Huffman streams with
SIMD IDCT, color convert (YCC → RGB), and fancy (interpolating)
chroma upsample on the standard 4:2:0 / 4:2:2 / 4:4:0 layouts. The
Huffman entropy decoder stays scalar by design — the bit-reader +
canonical-table walk has a serial dependency on per-symbol code
length that doesn't reshape into vector SIMD — but 0.7.0 lands two
scalar bit-ops refinements (combined run/size + magnitude LUT, and
a SWAR 32-bit bit-reader refill) on top of the per-stage SIMD
kernels.

## Why

This crate exists because a real workload needed a **fast JPEG
encoder in pure Rust** — `image`'s bundled encoder is solid but
purely scalar and 4:2:0-only, which leaves throughput on the table
for pipelines that emit a lot of JPEGs. The SIMD encoder here lifts
whole-pipeline throughput roughly **2.9×** on Apple Silicon and
**3.3×** on Intel Cascade Lake versus `image`'s encoder on the same
content, supports 4:4:4 / 4:2:2 / 4:2:0, and opts in to
multi-threaded MCU-row encode, optimized (two-pass) Huffman tables,
restart markers, and custom quantization tables. **Encode speed is
the headline.**

The decoder is bundled for API symmetry — read your own JPEGs back
without reaching for another crate — rather than as a speed play.
It gained per-stage SIMD kernels in 0.6.0 (IDCT / color convert /
fancy upsample) and progressively closed the gap to `image`'s SIMD
decoder over 0.7.x; **as of 0.7.5 we are faster than `image` at 4K
on both microarchitectures and both corpora** (~1.06–1.10× on
synthetic Huffman-heavy content, ~1.19–1.21× on natural-content),
while matching coverage — baseline + progressive Huffman with
fancy chroma upsample, output in any of eight pixel layouts. The
0.7.5 hot-path changes (entropy + dequant fusion, AVX2 PSHUFB
RGB interleave, uninit-alloc) are why; the smaller-resolution
fixed-cost overhead means we still trail by 5–11% at 1592×1124 /
1080p on Cascade Lake, which is queued for 0.8.x.

## Performance

50-iteration single-batch timings from `src/bin/bench.rs` and
`tests/comparison_bench.rs`, q=80, 4:2:0. Two hosts: Apple M-series
(NEON) and Intel Xeon Platinum 8272CL (Cascade Lake, AVX2). Full
per-section breakdown in [BENCH.md](BENCH.md).

### vs `image` crate — encode (RGB → JPEG, single thread)

| Resolution                  | ours (Apple M) | image (Apple M) | ratio   | ours (Cascade Lake) | image (Cascade Lake) | ratio   |
| --------------------------- | -------------: | --------------: | ------: | ------------------: | -------------------: | ------: |
| 1592 × 1124 (session size)  |        5.20 ms |        15.21 ms |  2.93×  |            23.19 ms |             75.32 ms |  3.25×  |
| 1920 × 1080 (1080p)         |        5.89 ms |        17.18 ms |  2.92×  |            26.51 ms |             86.74 ms |  3.27×  |
| 3840 × 2160 (4K)            |       23.38 ms |        66.70 ms |  2.85×  |           105.12 ms |            344.80 ms |  3.28×  |

### vs `image` crate — decode (JPEG → RGB)

`image` 0.25 ships `zune-jpeg` as its JPEG decoder, also fully
SIMD-accelerated. Our decoder gained per-stage SIMD kernels in
0.6.0 (IDCT / color convert / fancy upsample). 0.7.0 added AVX2
IDCT sparse parity, a combined AC/DC Huffman LUT, and a SWAR
32-bit Huffman bit-reader refill. 0.7.5 fuses dequantize into
the entropy-decode block loop (eliminating the per-block dequant
pass), replaces an x86_64 scalar RGB-interleave loop with a
PSHUFB shuffle kernel, and skips zero-fill on the per-decode
plane / output Vec allocations.

Both decoders are timed on the same harness
(`tests/comparison_bench.rs`) against two corpora: the synthetic
XOR pattern used everywhere else in the docs (Huffman-heavy worst
case — every block is full-AC) and a procedural natural-content
image (smooth sky + low-AC texture + edge bars) that is the fairer
proxy for typical web/photo input.

**Synthetic (worst case):**

| Resolution                  | ours (Apple M) | image (Apple M) | ratio   | ours (Cascade Lake) | image (Cascade Lake) | ratio   |
| --------------------------- | -------------: | --------------: | ------: | ------------------: | -------------------: | ------: |
| 1592 × 1124 (session size)  |        3.91 ms |         4.31 ms |  1.10×  |            12.90 ms |             12.30 ms |  0.95×  |
| 1920 × 1080 (1080p)         |        4.53 ms |         4.93 ms |  1.09×  |            14.61 ms |             14.24 ms |  0.97×  |
| 3840 × 2160 (4K)            |       18.29 ms |        19.34 ms |  1.06×  |            62.04 ms |             68.25 ms |  1.10×  |

**Natural content (procedural sky + texture + edges):**

| Resolution                  | ours (Apple M) | image (Apple M) | ratio   | ours (Cascade Lake) | image (Cascade Lake) | ratio   |
| --------------------------- | -------------: | --------------: | ------: | ------------------: | -------------------: | ------: |
| 1592 × 1124 (session size)  |        1.33 ms |         1.55 ms |  1.17×  |             5.10 ms |              4.56 ms |  0.89×  |
| 1920 × 1080 (1080p)         |        1.53 ms |         1.85 ms |  1.21×  |             5.70 ms |              5.29 ms |  0.93×  |
| 3840 × 2160 (4K)            |        5.87 ms |         7.00 ms |  1.19×  |            26.79 ms |             32.54 ms |  1.21×  |

(ratio > 1 means jpeg-rusturbo is faster)

At 4K — the resolution that dominates real workloads — we sit
ahead of `image` on both microarchitectures and on both corpora:
1.06×–1.10× on synthetic Huffman-heavy content and 1.19×–1.21×
on natural-content. At smaller resolutions the picture is more
mixed on Cascade Lake (we lose by 5–11% at 1592×1124 / 1080p on
both corpora) because the per-decode fixed-cost overhead is a
larger fraction of total time there; the relative gain from the
0.7.5 hot-path changes (entropy + dequant fusion, AVX2 RGB
interleave, uninit-alloc) scales with the amount of per-pixel
work and so shows up most strongly at 4K. Apple M's lower
fixed-cost overhead means even small-resolution decode comes
out ahead.

### Threading and optimized Huffman

`set_threads(n)` partitions encode across MCU rows; `threads=auto`
picks `available_parallelism()`. Bit-identical output across thread
counts.

| Host                     | Resolution | threads=1 | threads=auto | speedup |
| ------------------------ | ---------- | --------: | -----------: | ------: |
| Apple M (8 cores)        | 1080p      |   5.87 ms |      3.47 ms |   1.69× |
| Apple M (8 cores)        | 4K         |  23.82 ms |     13.31 ms |   1.79× |
| Cascade Lake (4 vCPU)    | 1080p      |  18.35 ms |     15.39 ms |   1.19× |
| Cascade Lake (4 vCPU)    | 4K         |  73.20 ms |     58.88 ms |   1.24× |

`set_optimize_huffman(true)` enables a per-image two-pass Huffman
build (T.81 K.2/K.3). Typical size reduction ~5% across
subsampling × quality on synthetic content (4–10% on natural
photos), at roughly 1.5–1.8× encode wall-clock. Opt-in for when
bandwidth matters more than CPU.

## Quick start

```toml
# Cargo.toml
[dependencies]
jpeg-rusturbo = "0.7"
```

### Encode

```rust
use jpeg_rusturbo::{ChromaSubsampling, JpegEncoder, PixelFormat};
use std::fs::File;
use std::io::BufWriter;

fn save(path: &str, rgba: &[u8], w: u32, h: u32) -> std::io::Result<()> {
    let f = BufWriter::new(File::create(path)?);
    let mut enc = JpegEncoder::new_with_quality(f, 80);
    enc.set_subsampling(ChromaSubsampling::Yuv420); // default; explicit for clarity
    enc.encode_rgba(rgba, w, h)
}

// Non-RGB[A] byte layouts go through the generic `encode` entry point.
fn save_bgra(path: &str, bgra: &[u8], w: u32, h: u32) -> std::io::Result<()> {
    let f = BufWriter::new(File::create(path)?);
    let mut enc = JpegEncoder::new_with_quality(f, 80);
    enc.encode(bgra, w, h, PixelFormat::Bgra)
}
```

The encoder accepts `&[u8]` in any of eight pixel layouts — `Rgb`,
`Bgr`, `Rgba`, `Bgra`, `Argb`, `Abgr`, `Rgbx`, `Bgrx` (alpha or pad
byte dropped). Quality is clamped to `1..=100`; subsampling defaults
to 4:2:0, with 4:2:2 and 4:4:4 available via `set_subsampling`.

### Decode

```rust
use jpeg_rusturbo::{decode, PixelFormat};

let jpeg_bytes: &[u8] = /* … */;
let rgb = decode::decode(jpeg_bytes, PixelFormat::Rgb)?;
// `rgb.len() == width * height * 3`

// Inspect dimensions without decoding:
let dec = decode::Decoder::new(jpeg_bytes)?;
let info = dec.info();
println!("{}x{}, {} components", info.width, info.height, info.components);
let pixels = dec.decode(PixelFormat::Rgba)?;
```

Output can be requested in any of the eight pixel layouts. Both
**baseline (SOF0) and progressive (SOF2)** Huffman streams are
accepted; arithmetic-coded (SOF9-15), hierarchical, and lossless
modes return `DecodeError::Unsupported`.

## Features

### Encoder

- **NEON on AArch64** — color convert, FDCT, quantize + zig-zag,
  4:2:0 / 4:2:2 chroma downsample, and the Huffman nonzero bitmap.
- **AVX2 on x86_64** — color convert, FDCT, quantize + zig-zag, 4:2:0
  / 4:2:2 chroma downsample. Runtime `is_x86_feature_detected!` falls
  back to scalar on non-AVX2 CPUs.
- **SSE2 on x86_64** — Huffman nonzero bitmap
  (`pcmpeqw + packsswb + pmovmskb`, translated from
  `jchuff-sse2.asm`). SSE2 is the x86_64 baseline, no runtime gate.
- **Eight input pixel layouts** — `Rgb`, `Bgr`, `Rgba`, `Bgra`,
  `Argb`, `Abgr`, `Rgbx`, `Bgrx` via the generic
  `JpegEncoder::encode` entry point.
- **Three chroma modes** — 4:4:4, 4:2:2, 4:2:0.
- **Multi-threaded MCU-row encode** — `set_threads(n)` (or `0` =
  `available_parallelism()`) partitions quantize + AC bitmap work
  across rayon workers. Serial when `n == 1`. Bit-identical output
  across thread counts.
- **Optimized (two-pass) Huffman** — `set_optimize_huffman(true)`
  builds canonical Huffman tables (T.81 K.2/K.3) from per-image
  symbol frequencies. Typical 5–10% size reduction at ~1.5–1.8×
  encode cost.
- **Restart markers** — `set_restart_interval(n)` emits RSTm every
  `n` MCUs (DRI segment + interleaved RSTm), for error resilience or
  parallel decode by downstream readers.
- **Custom quantization tables** — `set_quant_tables(luma, chroma)`
  accepts two `[u8; 64]` arrays in natural (row-major) order;
  bypasses the built-in quality-driven Annex K scaling.
- **Scalar fallback** — on every target, opt-in via the `force-scalar`
  Cargo feature, or used automatically on architectures without a
  SIMD backend.
- **Bit-exact across backends** — cross-check tests assert that NEON,
  scalar, and AVX2 / SSE2 produce byte-identical JPEG output.

### Decoder

- **Baseline (SOF0) + Progressive (SOF2) Huffman** — full scan loop
  with DC first / DC refine / AC first / AC refine bands, EOBRUN
  bookkeeping, multi-SOS streams (per-component baseline scans
  included), and restart-marker (RSTn) handling.
- **Fancy (interpolating) chroma upsample** — libjpeg-turbo's
  `h2v2_fancy` / `h2v1_fancy` 2-tap (3 center, 1 neighbor) filter on
  the standard 4:2:0 / 4:2:2 / 4:4:0 layouts, with NEON / AVX2 SIMD
  kernels. Wider sampling factors (4:1:1 etc.) fall back to box
  replication.
- **`jidctint`-style IDCT** — bit-exact against libjpeg-turbo's
  integer reference, with NEON and AVX2 ported kernels. Scalar
  fallback retained for `force-scalar` and non-SIMD targets.
  Includes **DC-only and sparse-row fast paths** that detect blocks
  with rows 4–7 (or all AC) zero — common on smooth regions in
  natural photographs — and skip the corresponding butterflies. Worth
  +11–19% of total decode time on natural content; see
  [BENCH.md](BENCH.md) Section D-natural.
- **NEON / AVX2 YCC → RGB color convert** — per-row converter ported
  from libjpeg-turbo, runtime-dispatched alongside IDCT and upsample.
- **Scalar Huffman entropy decoder** — bit-reader + canonical-table
  walk; the serial dependency on per-symbol code length doesn't
  reshape into vector SIMD, so the entropy decoder is scalar by
  design. 0.7.0 lands two scalar bit-ops refinements on top: a
  combined run/size + magnitude LUT (table-driven path, ~97% slot
  coverage on standard JPEG tables) for both AC and DC terms,
  including the progressive DC-first and AC-first scans; and a SWAR
  32-bit bit-reader refill that fills the `u64` accumulator four
  bytes at a time when no `0xFF` byte stuffing is present (checked
  via `(y - 0x0101_0101) & !y & 0x8080_8080`). The SWAR refill
  delivers +4–7% on natural 4K content across both NEON and AVX2;
  the combined LUT is at the noise floor at q=80 on Cascade Lake
  and Apple M but is retained as a table-driven foundation —
  bit-exact, zero runtime cost on misses, and the canonical
  approach used by libjpeg-turbo and zune-jpeg.
- **Eight output pixel layouts** via the same `PixelFormat` enum the
  encoder accepts.
- **Cross-decoder validation** — `tests/comparison_progressive.rs`
  asserts per-channel agreement with `image`'s decoder (≤ 3 / channel,
  ≥ 40 dB PSNR) on a vendored 5-fixture corpus (baseline grayscale,
  baseline 4:2:0 odd-size, progressive 4:2:0, two progressive 4:4:4
  sizes).
- **Known gaps** — arithmetic / hierarchical / lossless decode are
  not in scope.

The encoder's Huffman AC scan is bitmap-driven: a `u64` nonzero
bitmap collapses zero runs into a single `trailing_zeros` jump per
nonzero. The bitmap itself is SIMD on both architectures (NEON / SSE2);
the per-nonzero symbol emission stays scalar. See [BENCH.md](BENCH.md)
for the per-stage breakdown.

## Architecture (brief)

The crate's surface is intentionally small. Encode side:
`JpegEncoder`, `ChromaSubsampling`, `PixelFormat`, `encode`,
`encode_rgb`, `encode_rgba`. Decode side: `decode::Decoder`,
`decode::decode`, `decode::ImageInfo`, `decode::DecodeError`.
Per-architecture kernels live behind `arch::backend::*`, selected at
compile time:

```
aarch64 + !force-scalar  →  arch::neon
x86_64  + !force-scalar  →  arch::x86_64   (AVX2; runtime fallback to scalar)
otherwise                →  arch::scalar
```

Adding a new backend (e.g. WebAssembly SIMD) is "drop a new file with
the kernel modules + add a `cfg` arm in `arch/mod.rs`"; see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full layout.

## Status

Pre-1.0, single-author project. The encoder produces standard
baseline JPEG that decodes round-trip-equivalent through any
conforming decoder; the decoder reads baseline and progressive
Huffman JPEGs that conform to ITU-T T.81 (verified against
`image`'s decoder on a vendored fixture corpus). Public API has
settled but `0.x` reserves the right to evolve before `1.0`.

The 0.5.0 line shipped the encoder differentiation work: optimized
two-pass Huffman, multi-thread MCU-row encode, restart intervals,
and custom quantization tables. **0.6.0 landed decoder SIMD** —
NEON + AVX2 kernels for IDCT, YCC → RGB color convert, and fancy
chroma upsample — closing the decode-side gap vs `image` from
~0.41× to ~0.77× on the standard 4:2:0 path. The release profile
also tightened to `lto = "fat"` + `codegen-units = 1`. **0.7.0
landed two decode-side refinements**: AVX2 IDCT sparse parity
(porting the NEON DC-only and rows/cols-4–7-zero fast paths to
x86_64), and a cross-arch SWAR 32-bit Huffman bit-reader refill.
Vector-SIMD Huffman remains impractical due to the serial dependency
on per-symbol code length; the table-driven combined LUT (for both
AC and DC, including progressive scans) is the canonical alternative
used by libjpeg-turbo and `zune-jpeg`. **0.7.5 closes the decoder
cycle**: fusing dequantize into the entropy-decode block loop
(eliminating the per-block intermediate `zz_coef` and the
`for k in 0..64` dequant pass), replacing the x86_64 RGB 3-byte
interleave's scalar VPEXTRB loop with a PSHUFB shuffle kernel
(saved ~6 ms on Cascade Lake 4K), and skipping zero-fill on the
per-decode plane / output Vec allocations (saved ~5 ms on Cascade
Lake 4K). Combined effect: 4K 4:2:0 natural-content decode drops
from 41.8 ms → 22.1 ms on Cascade Lake and 8.6 ms → 5.7 ms on
Apple M, putting both microarchitectures ahead of `image` /
`zune-jpeg` on the same fixture. **Trellis quantization** (mozjpeg-style RDO per-block
search) remains under consideration for a later release, depending
on demand.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the issue / PR policy.

## License

Licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option. Third-party attributions
(libjpeg-turbo, image) are listed in [NOTICE.md](NOTICE.md).
