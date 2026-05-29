# jpeg-rusturbo

[![CI](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/ci.yml/badge.svg)](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/ci.yml)
[![Release](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/release.yml/badge.svg)](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/release.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**SIMD-accelerated JPEG encoder (baseline SOF0 + progressive
SOF2 + EXIF / ICC metadata pass-through) and Huffman decoder
(baseline + progressive), with libjpeg-turbo-derived kernels.**
Both sides ship NEON-on-aarch64 and AVX2-on-x86_64 kernels;
non-SIMD targets fall through to a bit-exact scalar reference.
Drop-in for `image::codecs::jpeg::JpegEncoder` on the encode side;
standalone decoder under `jpeg_rusturbo::decode`.

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
length that doesn't reshape into vector SIMD — but 0.7.0 landed two
scalar bit-ops refinements (combined run/size + magnitude LUT, and
a SWAR 32-bit bit-reader refill) on top of the per-stage SIMD
kernels.

## Why

This crate exists because a real workload needed a **fast JPEG
encoder in pure Rust** — `image`'s bundled encoder is solid but
purely scalar and 4:2:0-only, which leaves throughput on the table
for pipelines that emit a lot of JPEGs. The SIMD encoder here
lifts whole-pipeline throughput roughly **4.5× on Apple Silicon**
and **5.5× on Intel Cascade Lake** versus `image`'s encoder at
4:2:0 (was 2.9× / 3.3× in 0.7.5; the 0.8.0 encoder hot-path pass —
four NEON items plus a new AVX2 3-byte RGB→YCbCr kernel — pushes
both well past 4×, the Intel figure further because `image`'s
scalar encoder is slower on that CPU). It supports
4:4:4 / 4:2:2 / 4:2:0, **progressive (SOF2) output**
alongside baseline, **EXIF / ICC pass-through** for re-encode
pipelines, multi-threaded MCU-row encode, optimized (two-pass)
Huffman tables, restart markers, and custom quantization tables.
**Encode speed is the headline.**

The decoder is bundled for API symmetry — read your own JPEGs back
without reaching for another crate — rather than as a speed play.
It gained per-stage SIMD kernels in 0.6.0 (IDCT / color convert /
fancy upsample) and progressively closed the gap to `image`'s SIMD
decoder over 0.7.x; **as of 0.7.5 we are faster than `image` at 4K
on both microarchitectures and both corpora** (~1.03–1.10× on
synthetic Huffman-heavy content, ~1.18–1.22× on natural-content),
while matching coverage — baseline + progressive Huffman with
fancy chroma upsample, output in any of eight pixel layouts. The
smaller-resolution fixed-cost overhead means we trail by only
~2–3% at 1592×1124 on Cascade Lake (at parity or ahead from 1080p
up). The decoder side wasn't reopened in 0.8.0 (that cycle was
encoder-focused — see Status below) and the small Cascade Lake
gap remains an open item.

## Performance

50-iteration single-batch timings from `src/bin/bench.rs` and
`tests/comparison_bench.rs`, q=80, 4:2:0. Two hosts: Apple M-series
(NEON) and Intel Xeon Platinum 8272CL (Cascade Lake, AVX2). Full
per-section breakdown in [BENCH.md](BENCH.md).

### vs `image` crate — encode (RGB → JPEG, single thread)

`image`'s encoder is scalar end to end and 4:2:0-only; jpeg-rusturbo
runs SIMD across the whole encode front end — color convert, forward
DCT, quantize + zig-zag, chroma downsample — plus a SIMD nonzero-bitmap
Huffman path. Both encoders are fed identical 3-byte RGB and timed on
the same harness (`tests/comparison_bench.rs`). That structural
difference — vectorized pipeline vs scalar — is the gap below.

**Apple M-series (NEON)**

| Resolution (4:2:0)          | jpeg-rusturbo | image    | ratio |
| --------------------------- | ------------: | -------: | ----: |
| 1592 × 1124 (session size)  |       3.50 ms | 15.79 ms | 4.51× |
| 1920 × 1080 (1080p)         |       3.94 ms | 17.92 ms | 4.55× |
| 3840 × 2160 (4K)            |      15.64 ms | 69.07 ms | 4.42× |

**Intel Xeon (Cascade Lake, AVX2)**

| Resolution (4:2:0)          | jpeg-rusturbo | image     | ratio |
| --------------------------- | ------------: | --------: | ----: |
| 1592 × 1124 (session size)  |      14.27 ms |  76.90 ms | 5.39× |
| 1920 × 1080 (1080p)         |      16.19 ms |  88.62 ms | 5.48× |
| 3840 × 2160 (4K)            |      63.40 ms | 352.09 ms | 5.55× |

The lead is **4.4–4.6× on Apple M** and **5.4–5.6× on Cascade Lake**,
and — unlike decode — it holds roughly flat across resolution: encode
is per-pixel SIMD work end to end, with no per-call fixed-cost cliff at
smaller sizes. The gap widened from 0.7.5's 2.9× / 3.3× through the
0.8.0 hot-path pass: four NEON items, two of which (`drain_high32`, the
AC code+magnitude `write_bits` fusion) also help AVX2, plus a new AVX2
3-byte RGB→YCbCr kernel that brought RGB-input encode onto the AVX2
color path — Cascade Lake 4K 4:2:0 dropped 95.8 → 63.4 ms, lifting its
ratio from ~3.3× to ~5.5×. The Cascade figure runs higher than Apple's
mainly because `image`'s scalar encoder is slower on that CPU; read
each ratio as "vs `image` on this host." Per-item breakdown in
[BENCH.md](BENCH.md).

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

**Apple M-series (NEON)**

| Corpus / resolution     | jpeg-rusturbo | image    | ratio |
| ----------------------- | ------------: | -------: | ----: |
| synthetic  1592 × 1124  |       4.22 ms |  4.39 ms | 1.04× |
| synthetic  1080p        |       4.59 ms |  5.03 ms | 1.10× |
| synthetic  4K           |      19.06 ms | 19.65 ms | 1.03× |
| natural    1592 × 1124  |       1.35 ms |  1.55 ms | 1.15× |
| natural    1080p        |       1.55 ms |  1.86 ms | 1.20× |
| natural    4K           |       5.93 ms |  7.00 ms | 1.18× |

**Intel Xeon (Cascade Lake, AVX2)**

| Corpus / resolution     | jpeg-rusturbo | image    | ratio |
| ----------------------- | ------------: | -------: | ----: |
| synthetic  1592 × 1124  |      13.12 ms | 12.90 ms | 0.98× |
| synthetic  1080p        |      14.94 ms | 15.04 ms | 1.01× |
| synthetic  4K           |      62.94 ms | 69.11 ms | 1.10× |
| natural    1592 × 1124  |       5.21 ms |  5.07 ms | 0.97× |
| natural    1080p        |       5.83 ms |  5.97 ms | 1.02× |
| natural    4K           |      28.01 ms | 34.26 ms | 1.22× |

(ratio > 1 means jpeg-rusturbo is faster)

At 4K — the resolution that dominates real workloads — we sit
ahead of `image` on both microarchitectures and on both corpora:
1.03×–1.10× on synthetic Huffman-heavy content and 1.18×–1.22×
on natural-content. At smaller resolutions Cascade Lake is roughly
at parity — within ~2–3% at 1592×1124, level or ahead from 1080p
up — because the per-decode fixed-cost overhead is a larger
fraction of total time there; the relative gain from the 0.7.5
hot-path changes (entropy + dequant fusion, AVX2 RGB interleave,
uninit-alloc) scales with the amount of per-pixel work and so
shows up most strongly at 4K. Apple M's lower fixed-cost overhead
means even small-resolution decode comes out ahead.

### Threading and optimized Huffman

`set_threads(n)` partitions encode across MCU rows; `threads=auto`
picks `available_parallelism()`. Bit-identical output across thread
counts.

**Apple M-series (8 cores)**

| Resolution | threads=1 | threads=auto | auto vs t=1 |
| ---------- | --------: | -----------: | ----------: |
| 1080p      |   3.90 ms |      1.90 ms |       2.05× |
| 4K         |  16.14 ms |      6.85 ms |       2.36× |

**Intel Xeon (Cascade Lake, 4 vCPU)**

| Resolution | threads=1 | threads=auto | auto vs t=1 |
| ---------- | --------: | -----------: | ----------: |
| 1080p      |  16.69 ms |     13.71 ms |       1.22× |
| 4K         |  66.23 ms |     53.60 ms |       1.24× |

`set_optimize_huffman(true)` enables a per-image two-pass Huffman
build (T.81 K.2/K.3). Typical size reduction ~5% across
subsampling × quality on synthetic content (4–10% on natural
photos), at roughly 1.7× encode wall-clock on AVX2 and ~2.3× on
NEON (the second, largely scalar, pass weighs more where the first
pass is faster). Opt-in for when bandwidth matters more than CPU.

## Quick start

```toml
# Cargo.toml
[dependencies]
jpeg-rusturbo = "0.8"
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
- **Progressive (SOF2) output** — `set_progressive(true)` emits an
  8-scan successive-approximation progressive JPEG (DC interleaved
  first + per-component AC first at `Al=1`, then DC interleaved
  refine + per-component AC refine at `Al=0`). All four T.81 Annex G
  scan types implemented. Decodable by every conforming progressive
  decoder including the one in this crate. Default off — baseline
  output is bit-identical to pre-0.8.0 when this setter isn't called.
- **EXIF / ICC metadata pass-through** — `set_exif(Option<Vec<u8>>)`
  / `set_icc_profile(Option<Vec<u8>>)` route raw blobs through as
  APP1 / APP2 segments immediately after the JFIF APP0. ICC profiles
  larger than ~65 KB are split across multiple APP2 segments per
  the ICC.1 multi-segment embedding convention. Use case: decode →
  operate → re-encode pipelines that would otherwise drop the
  camera EXIF / color profile.
- **Multi-threaded MCU-row encode** — `set_threads(n)` (or `0` =
  `available_parallelism()`) partitions quantize + AC bitmap work
  across rayon workers. Serial when `n == 1`. Bit-identical output
  across thread counts.
- **Optimized (two-pass) Huffman** — `set_optimize_huffman(true)`
  builds canonical Huffman tables (T.81 K.2/K.3) from per-image
  symbol frequencies. Typical ~5% (synthetic) / 4–10% (natural)
  size reduction at ~1.7× (AVX2) / ~2.3× (NEON) encode cost.
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
  [BENCH.md](BENCH.md) (Decode chapter).
- **NEON / AVX2 YCC → RGB color convert** — per-row converter ported
  from libjpeg-turbo, runtime-dispatched alongside IDCT and upsample.
- **Scalar Huffman entropy decoder** — bit-reader + canonical-table
  walk; the serial dependency on per-symbol code length doesn't
  reshape into vector SIMD, so the entropy decoder is scalar by
  design. 0.7.0 landed two scalar bit-ops refinements on top: a
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
baseline and progressive JPEG that decodes round-trip-equivalent
through any conforming decoder; the decoder reads baseline and
progressive Huffman JPEGs that conform to ITU-T T.81 (verified against
`image`'s decoder on a vendored fixture corpus). Public API has
settled but `0.x` reserves the right to evolve before `1.0`.

The current release, **0.8.0**, closes the encoder cycle: a hot-path
SIMD pass (~4.5× / ~5.5× vs `image` at 4:2:0 on Apple Silicon /
Cascade Lake) plus progressive (SOF2) output and EXIF / ICC
pass-through. The 0.6.0 / 0.7.x cycles before it built out the decoder
SIMD path. Full per-release history is in [CHANGELOG.md](CHANGELOG.md).

Still under consideration for a later release: **trellis quantization**
(mozjpeg-style RDO per-block search) and an `encode_progressive_optimize`
path (optimized-Huffman for SOF2, to recover the per-block-EOB0 size
cost). Vector-SIMD Huffman decode stays out of scope — the bit-reader +
canonical-table walk has a serial per-symbol code-length dependency that
doesn't vectorize.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the issue / PR policy.

## License

Licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option. Third-party attributions
(libjpeg-turbo, image) are listed in [NOTICE.md](NOTICE.md).
