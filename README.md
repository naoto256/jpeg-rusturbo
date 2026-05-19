# jpeg-rusturbo

[![CI](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/ci.yml/badge.svg)](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/ci.yml)
[![Release](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/release.yml/badge.svg)](https://github.com/naoto256/jpeg-rusturbo/actions/workflows/release.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

**Baseline JPEG encoder with libjpeg-turbo-derived SIMD kernels —
drop-in for `image::codecs::jpeg::JpegEncoder`.**

```rust
use jpeg_rusturbo::JpegEncoder;

let mut out = Vec::new();
let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
enc.encode_rgba(&pixels, width, height)?;
// `out` is now a complete baseline JPEG.
```

The public encoder API mirrors `image::codecs::jpeg::JpegEncoder` so
call sites swap with a `use` change. Internally, hot paths (color
conversion, FDCT, quantize, 4:2:0 and 4:2:2 chroma downsample)
dispatch to NEON on AArch64 and AVX2 on x86_64 via translations of
libjpeg-turbo's SIMD sources; non-SIMD targets fall through to a
bit-exact scalar reference.

## Why

The `image` crate's bundled JPEG encoder is solid but scalar. On a 4K
RGBA frame at q=80 it spends nearly all of its time on color
conversion, DCT, quantize, and Huffman — work that vectorizes well.
Pulling those kernels off the scalar path lifts whole-pipeline
throughput by **~1.5× on Apple Silicon** and **~2.0× on Intel
Ice Lake** versus scalar code on the same hardware, without changing
the bytes that come out.

## Performance

100-iteration medians of 5 repeated runs, q=80.

### 4:2:0

| Resolution                  | Apple M-series (NEON) | Intel Xeon Platinum 8370C (AVX2) |
| --------------------------- | --------------------: | -------------------------------: |
| 1592 × 1124 (session size)  |              5.49 ms  |                          11.82 ms |
| 1920 × 1080 (1080p)         |              6.23 ms  |                          13.65 ms |
| 3840 × 2160 (4K)            |             25.04 ms  |                          53.50 ms |

### 4:2:2

| Resolution                  | Apple M-series (NEON) | Intel Xeon Platinum 8370C (AVX2) |
| --------------------------- | --------------------: | -------------------------------: |
| 1592 × 1124 (session size)  |              7.43 ms  |                          15.29 ms |
| 1920 × 1080 (1080p)         |              8.45 ms  |                          17.45 ms |
| 3840 × 2160 (4K)            |             33.00 ms  |                          68.18 ms |

Scalar-fallback ratios on the same hosts are 1.44–1.68× slower than
NEON (Apple M-series) and 2.04–2.07× slower than AVX2 (Intel
Ice Lake). Output bytes are byte-identical across SIMD and scalar
paths at every resolution; cross-check unit tests + the roundtrip
suite assert this. Full breakdown including per-stage profiling is
in [BENCH.md](BENCH.md).

## Quick start

```toml
# Cargo.toml
[dependencies]
jpeg-rusturbo = { git = "..." }   # crates.io publication TBD
```

```rust
use jpeg_rusturbo::{ChromaSubsampling, JpegEncoder};
use std::fs::File;
use std::io::BufWriter;

fn save(path: &str, rgba: &[u8], w: u32, h: u32) -> std::io::Result<()> {
    let f = BufWriter::new(File::create(path)?);
    let mut enc = JpegEncoder::new_with_quality(f, 80);
    enc.set_subsampling(ChromaSubsampling::Yuv420); // default; explicit for clarity
    enc.encode_rgba(rgba, w, h)
}
```

The encoder accepts `&[u8]` in either RGB (3 bytes/pixel) or RGBA
(4 bytes/pixel, alpha dropped — JPEG has no alpha channel) and writes
to any `std::io::Write`. Quality is clamped to 1..=100; subsampling
defaults to 4:2:0, with 4:2:2 and 4:4:4 available via
`set_subsampling`.

## Features

- **NEON on AArch64** — full set of four hot kernels.
- **AVX2 on x86_64** — same four kernels. Runtime
  `is_x86_feature_detected!` falls back to scalar on non-AVX2 CPUs;
  the result is cached, so only the first call pays the check.
- **Scalar fallback** — on every target, opt-in via the `force-scalar`
  Cargo feature, or used automatically on architectures without a
  SIMD backend.
- **Bit-exact across backends** — cross-check tests assert that NEON,
  scalar, and AVX2 produce byte-identical JPEG output.
- **`image::codecs::jpeg::JpegEncoder`-shaped public API** — port a
  call site by swapping the `use` line.

The Huffman entropy coder stays scalar by design: it autovectorizes
well in trivial form, the AC scan is too branchy for SIMD to win, and
libjpeg-turbo upstream itself doesn't ship an AArch64-NEON / AVX2
Huffman kernel. See [BENCH.md](BENCH.md)'s Huffman and AVX2 sections
for the reasoning trail.

## Architecture (brief)

The crate is a single library with a small public surface
(`JpegEncoder`, `ChromaSubsampling`, `encode_rgb`, `encode_rgba`).
Per-architecture kernels live behind `arch::backend::*`, selected at
compile time:

```
aarch64 + !force-scalar  →  arch::neon
x86_64  + !force-scalar  →  arch::x86_64   (AVX2; runtime fallback to scalar)
otherwise                →  arch::scalar
```

Adding a new backend (e.g. WebAssembly SIMD) is "drop a new file with
four kernel modules + add a `cfg` arm in `arch/mod.rs`"; see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full layout.

## Status

Pre-1.0, single-author project. The encoder produces standard baseline
JPEG that decodes round-trip-equivalent through any conforming decoder
(verified via `image`'s decoder in the test suite). The public API has
settled but `0.1` reserves the right to evolve before `1.0`.

The public API is intentionally identical to `image`'s
`JpegEncoder`, so call sites can swap implementations with a `use`
change.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the issue / PR policy.

## License

Licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option. Third-party attributions
(libjpeg-turbo, image) are listed in [NOTICE.md](NOTICE.md).
