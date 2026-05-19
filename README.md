# jpeg-rusturbo

[![CI](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/ci.yml/badge.svg)](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/ci.yml)
[![Release](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/release.yml/badge.svg)](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/release.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Baseline JPEG encode + decode with libjpeg-turbo-derived SIMD
kernels.** Drop-in for `image::codecs::jpeg::JpegEncoder` on the
encode side; standalone decoder under `jpeg_rusturbo::decode`.

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
bit-exact scalar reference. The decoder added in 0.3.0 is currently
scalar (libjpeg-turbo's `jidctint` reference); SIMD decode kernels
and progressive (SOF2) scan support are on the 0.4.0 roadmap.

## Why

`image` crate's bundled JPEG support is solid but the encoder is
purely scalar. Our SIMD encoder lifts whole-pipeline throughput
roughly **2.5×** on Apple Silicon and **3.9×** on Intel Ice Lake
versus `image`'s encoder on the same content. The decoder on the
other hand is a scalar reference today; it sits **~2.5× behind**
`image`'s SIMD decoder (which goes through `zune-jpeg` under the
hood). On a roundtrip workload the two crates are roughly even.

## Performance

100-iteration medians of 5 repeated runs, q=80, 4:2:0. Two hosts:
Apple M-series (NEON) and Intel Xeon Platinum 8370C (Ice Lake-SP).

### vs `image` crate — encode (RGB → JPEG)

| Resolution                  | ours (Apple M) | image (Apple M) | ratio   | ours (Xeon 8370C) | image (Xeon 8370C) | ratio   |
| --------------------------- | -------------: | --------------: | ------: | ----------------: | -----------------: | ------: |
| 1592 × 1124 (session size)  |        5.66 ms |        14.49 ms |  2.56×  |          16.03 ms |           62.01 ms |  3.87×  |
| 1920 × 1080 (1080p)         |        6.20 ms |        15.99 ms |  2.58×  |          18.34 ms |           71.32 ms |  3.89×  |
| 3840 × 2160 (4K)            |       24.33 ms |        61.75 ms |  2.54×  |          72.09 ms |          276.49 ms |  3.84×  |

### vs `image` crate — decode (JPEG → RGB)

`image` uses `zune-jpeg` (SIMD-accelerated); our decoder is currently
scalar. Decoder SIMD is the headline 0.4.0 work item.

| Resolution                  | ours (Apple M) | image (Apple M) | ratio   | ours (Xeon 8370C) | image (Xeon 8370C) | ratio   |
| --------------------------- | -------------: | --------------: | ------: | ----------------: | -----------------: | ------: |
| 1592 × 1124 (session size)  |       11.11 ms |         4.22 ms |  0.38×  |          23.66 ms |            9.30 ms |  0.39×  |
| 1920 × 1080 (1080p)         |       12.68 ms |         4.88 ms |  0.38×  |          27.01 ms |           10.70 ms |  0.40×  |
| 3840 × 2160 (4K)            |       50.17 ms |        19.04 ms |  0.38×  |         107.64 ms |           46.47 ms |  0.43×  |

(ratio > 1 means jpeg-rusturbo is faster)

### Encoder SIMD vs scalar (same host)

| Resolution                  | scalar (Apple M) | NEON (Apple M) | scalar (Xeon 8370C) | AVX2 (Xeon 8370C) |
| --------------------------- | ---------------: | -------------: | ------------------: | ----------------: |
| 1592 × 1124                 |          8.54 ms |        5.49 ms |            24.31 ms |          11.82 ms |
| 1920 × 1080                 |          9.94 ms |        6.23 ms |            27.93 ms |          13.65 ms |
| 3840 × 2160                 |         41.96 ms |       25.04 ms |           109.98 ms |          53.50 ms |

NEON ~1.5× scalar; AVX2 ~2.0× scalar on the same hardware. Output
bytes are byte-identical across SIMD and scalar paths. Full
breakdown in [BENCH.md](BENCH.md).

## Quick start

```toml
# Cargo.toml
[dependencies]
jpeg-rusturbo = "0.3"
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

Output can be requested in any of the eight pixel layouts. The
decoder is currently **baseline (SOF0) only** — progressive (SOF2)
JPEGs return `DecodeError::Unsupported`. Progressive support is on
the 0.4.0 roadmap.

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
- **Scalar fallback** — on every target, opt-in via the `force-scalar`
  Cargo feature, or used automatically on architectures without a
  SIMD backend.
- **Bit-exact across backends** — cross-check tests assert that NEON,
  scalar, and AVX2 / SSE2 produce byte-identical JPEG output.

### Decoder (new in 0.3.0)

- **Baseline Huffman (SOF0)** — DC + AC scan decode, libjpeg-turbo
  `jidctint`-style scalar IDCT, box-replication chroma upsample, and
  YCbCr → RGB convert into any of the eight output `PixelFormat`
  layouts. Restart markers (RSTn) supported.
- **Cross-decoder validation** — the test suite asserts per-pixel
  agreement with `image`'s decoder (≤ 3 / channel, ≥ 40 dB PSNR) on
  our encoder's output.
- **Self-roundtrip tests** — encode → decode → PSNR floor, run
  across 4:4:4 / 4:2:2 / 4:2:0 at multiple sizes / edge cases.
- **0.4.0 roadmap** — progressive (SOF2), fancy (interpolating)
  upsample, NEON + AVX2 IDCT / YCbCr→RGB / upsample kernels.

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
conforming decoder; the decoder reads any baseline JPEG that
conforms to ITU-T T.81 (verified against `image`'s decoder in the
test suite). Public API has settled but `0.x` reserves the right to
evolve before `1.0`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the issue / PR policy.

## License

Licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option. Third-party attributions
(libjpeg-turbo, image) are listed in [NOTICE.md](NOTICE.md).
