# jpeg-rusturbo

[![CI](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/ci.yml/badge.svg)](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/ci.yml)
[![Release](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/release.yml/badge.svg)](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/release.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**SIMD-accelerated baseline JPEG encoder + Huffman (baseline +
progressive) decoder with libjpeg-turbo-derived kernels.** Drop-in
for `image::codecs::jpeg::JpegEncoder` on the encode side; standalone
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

The encoder ships NEON-on-aarch64 and AVX2-on-x86_64 kernels
translated from libjpeg-turbo; non-SIMD targets fall through to a
bit-exact scalar reference. The decoder reads baseline (SOF0) and
progressive (SOF2) Huffman streams with fancy (interpolating) chroma
upsample on the standard 4:2:0 / 4:2:2 / 4:4:0 layouts; it stays
scalar by design — the SIMD-encode advantage is the headline win, so
SIMD decode kernels are on the post-0.5 roadmap.

## Why

`image` crate's bundled JPEG support is solid but the encoder is
purely scalar and 4:2:0-only. Our SIMD encoder lifts whole-pipeline
throughput roughly **2.5×** on Apple Silicon and **3.2×** on Intel
Broadwell versus `image`'s encoder on the same content, supports
4:4:4 / 4:2:2 / 4:2:0, and opts in to multi-threaded MCU-row encode,
optimized (two-pass) Huffman tables, restart markers, and custom
quantization tables. The decoder is a scalar `jidctint` reference;
it sits **~2.4× behind** `image`'s SIMD decoder (which goes through
`zune-jpeg`) but matches it on coverage — baseline + progressive
Huffman with fancy chroma upsample, output in any of eight pixel
layouts. On a roundtrip workload the two crates come out roughly
even; the "SIMD-encode + cover-everything-on-decode" combination is
the shape this crate targets.

## Performance

50-iteration single-batch timings from `src/bin/bench.rs` and
`tests/comparison_bench.rs`, q=80, 4:2:0. Two hosts: Apple M-series
(NEON) and Intel Xeon E5-2673 v4 (Broadwell, AVX2). Full per-section
breakdown in [BENCH.md](BENCH.md).

### vs `image` crate — encode (RGB → JPEG, single thread)

| Resolution                  | ours (Apple M) | image (Apple M) | ratio   | ours (Broadwell) | image (Broadwell) | ratio   |
| --------------------------- | -------------: | --------------: | ------: | ---------------: | ----------------: | ------: |
| 1592 × 1124 (session size)  |        5.62 ms |        14.36 ms |  2.56×  |         29.75 ms |          94.71 ms |  3.18×  |
| 1920 × 1080 (1080p)         |        6.32 ms |        16.22 ms |  2.57×  |         34.16 ms |         108.19 ms |  3.17×  |
| 3840 × 2160 (4K)            |       24.93 ms |        63.00 ms |  2.53×  |        134.55 ms |         432.59 ms |  3.22×  |

### vs `image` crate — decode (JPEG → RGB)

`image` uses `zune-jpeg` (SIMD-accelerated); our decoder is scalar
by design. Decoder SIMD is on the post-0.5 roadmap; 0.4.0 widened
decode *coverage* (progressive + fancy upsample) rather than perf.

| Resolution                  | ours (Apple M) | image (Apple M) | ratio   | ours (Broadwell) | image (Broadwell) | ratio   |
| --------------------------- | -------------: | --------------: | ------: | ---------------: | ----------------: | ------: |
| 1592 × 1124 (session size)  |       10.55 ms |         4.39 ms |  0.42×  |         41.56 ms |          18.32 ms |  0.44×  |
| 1920 × 1080 (1080p)         |       12.16 ms |         4.97 ms |  0.41×  |         47.61 ms |          20.84 ms |  0.44×  |
| 3840 × 2160 (4K)            |       48.30 ms |        19.65 ms |  0.41×  |        186.20 ms |          97.79 ms |  0.53×  |

(ratio > 1 means jpeg-rusturbo is faster)

### Threading and optimized Huffman (0.5.0)

`set_threads(n)` partitions encode across MCU rows; `threads=auto`
picks `available_parallelism()`. Bit-identical output across thread
counts.

| Host                | Resolution | threads=1 | threads=auto | speedup |
| ------------------- | ---------- | --------: | -----------: | ------: |
| Apple M (8 cores)   | 1080p      |   6.18 ms |      3.70 ms |   1.67× |
| Apple M (8 cores)   | 4K         |  25.39 ms |     13.90 ms |   1.83× |
| Broadwell (4 vCPU)  | 1080p      |  27.32 ms |     20.42 ms |   1.34× |
| Broadwell (4 vCPU)  | 4K         | 111.39 ms |     76.78 ms |   1.45× |

`set_optimize_huffman(true)` enables a per-image two-pass Huffman
build (T.81 K.2/K.3). Typical size reduction ~5% across
subsampling × quality on synthetic content (4–10% on natural
photos), at roughly 1.5–1.8× encode wall-clock. Opt-in for when
bandwidth matters more than CPU.

## Quick start

```toml
# Cargo.toml
[dependencies]
jpeg-rusturbo = "0.5"
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
  the standard 4:2:0 / 4:2:2 / 4:4:0 layouts. Wider sampling factors
  (4:1:1 etc.) fall back to box replication.
- **Scalar `jidctint`-style IDCT** — bit-exact against libjpeg-turbo's
  integer reference. Decoder SIMD kernels are on the post-0.5
  roadmap.
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
and custom quantization tables. The next focus is **decoder SIMD**
(IDCT / color convert / upsample kernels in NEON + AVX2, to close
the decode-side gap vs `zune-jpeg`). **Trellis quantization** (the
mozjpeg-style rate-distortion-optimized per-block search) is under
consideration for 0.6.x or later, depending on demand.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the issue / PR policy.

## License

Licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option. Third-party attributions
(libjpeg-turbo, image) are listed in [NOTICE.md](NOTICE.md).
