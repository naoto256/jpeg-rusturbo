# Changelog

All notable changes to `jpeg-rusturbo` are documented here. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/)
(pre-1.0: minor versions may carry behavioural changes, patch versions
do not).

Performance figures are summarized per release; the reproducible
breakdown lives in [BENCH.md](BENCH.md).

## [Unreleased — 0.9.0]

Filling in the non-perf coverage gaps from the 0.8.0 encoder cycle.

### Added

- **Grayscale encode** — new `JpegEncoder::encode_grayscale` convenience
  entry point (and new `PixelFormat::Gray` accepted by the generic
  `encode`) takes a single-byte-per-pixel buffer and emits a
  1-component (luma-only) JPEG with no chroma DQT / DHT / SOF / SOS
  overhead and no RGB→YCbCr conversion. Composes with
  `set_optimize_huffman`, custom quant tables, restart markers, EXIF /
  ICC pass-through. `set_subsampling` and `set_threads` are silently
  ignored on the grayscale path (no chroma to subsample; serial-only
  for now). `set_progressive(true)` combined with `encode_grayscale`
  returns `io::ErrorKind::Unsupported` — progressive grayscale is not
  yet implemented. On the decode side, `PixelFormat::Gray` now extracts
  the Y plane directly with no upsample / no color convert, working
  for both 1-component grayscale sources and 3-component color sources
  (Cb/Cr discarded). Default behaviour for the existing 8 color pixel
  formats is byte-identical to 0.8.0.

- **Optimized-Huffman progressive (SOF2) encode** — `set_optimize_huffman(true)`
  now composes with `set_progressive(true)`. A two-pass encode counts the
  symbol frequencies of each progressive scan (including the `EOBn`
  symbols its EOBRUN emission strategy produces), builds optimal
  per-scan T.81 K.3 length-limited Huffman tables, emits a DHT segment
  before each SOS, and packs multi-block end-of-band runs. Bit-exact
  decode equivalence with the non-optimize progressive path; collapses
  the size cost progressive normally carries vs baseline SOF0
  (natural-like 4:2:0 q=80: 4K progressive 364 KB → 148 KB, **−40% vs
  the corresponding baseline-SOF0 248 KB**; 1080p −37%; 1592×1124 −37%).
  Default-off behaviour is unchanged — without `set_optimize_huffman`,
  progressive output is bit-identical to 0.8.0.

## [0.8.0] — 2026-05-30

The encoder cycle: a hot-path SIMD pass plus two new encoder-surface
features. Default-off behaviour is bit-identical to 0.7.5.

### Added

- **Progressive (SOF2) encode** — `JpegEncoder::set_progressive(true)`
  emits an 8-scan successive-approximation progressive stream (DC
  interleaved first + per-component AC first at `Al=1`, then DC refine +
  per-component AC refine at `Al=0`), covering all four T.81 Annex G
  scan types. Decodable by any conforming progressive decoder, including
  this crate's.
- **EXIF / ICC metadata pass-through** — `set_exif(Option<Vec<u8>>)` and
  `set_icc_profile(Option<Vec<u8>>)` route raw blobs through as APP1 /
  APP2 segments after the JFIF APP0; ICC profiles over ~65 KB are split
  across multiple APP2 segments per the ICC.1 multi-segment convention.

### Performance

- Encoder hot path, all backends: an unsafe `BitWriter::drain_high32`
  that skips `Vec::push` bounds checks, and a fused AC code+magnitude
  `write_bits` that collapses two emits into one.
- Encoder hot path, NEON only: a `vqtbl4q` zig-zag scatter, and a
  one-shot SIMD precompute of the JPEG magnitude category per
  coefficient.
- Encoder hot path, AVX2 only: a 3-byte RGB→YCbCr deinterleave kernel.
  3-byte RGB input previously fell back to scalar colour conversion on
  x86_64; it now uses the AVX2 colour path, bringing `encode_rgb` to
  parity with `encode_rgba` (Cascade Lake 4K 4:2:0 RGB-input encode
  ~95.8 → ~63.4 ms).
- Net: encode vs the `image` crate at 4:2:0 rises to ~4.5× on Apple
  Silicon (NEON) and ~5.5× on Intel Cascade Lake (AVX2), from ~2.9× /
  ~3.3× in 0.7.5.

### Changed

- **Source tree reorganized** to mirror the encode/decode symmetry: the
  encoder modules + `JpegEncoder` moved into `src/encode/` (sibling to
  `src/decode/`), leaving `lib.rs` a thin crate root; `color.rs` /
  `tables.rs` / `arch/` remain the shared core. Both benchmark
  entrypoints moved to `benches/` (`pipeline.rs`, was `src/bin/bench.rs`;
  `vs_image.rs`, was `tests/comparison_bench.rs`, now a `harness = false`
  bench). No behaviour change — public API and output bytes are
  identical, verified perf-neutral on NEON and AVX2.
- Benchmark harness rebuilt with descriptive section names, a
  progressive baseline-vs-SOF2 section, and a RGB/RGBA input split;
  `BENCH.md` rebuilt from a single coherent two-host campaign (Apple
  M-series NEON + Intel Cascade Lake AVX2).

## [0.7.5] — 2026-05-28

Decoder cycle close — fixed-overhead and per-block trims.

### Performance

- Fuse dequantize into the entropy-decode block loop, eliminating the
  per-block intermediate coefficient buffer and the separate dequant
  pass.
- AVX2 RGB 3-byte output interleave via PSHUFB, replacing a scalar
  per-byte extract loop (decode side).
- Skip zero-fill on the per-decode sample-plane and output `Vec`
  allocations.
- Combined effect: 4K 4:2:0 natural-content decode moves ahead of
  `image` / `zune-jpeg` on both microarchitectures.

### Added

- `examples/visual_realtime` — a real-time encode/decode visualizer.

### Changed

- `deny.toml` accepts `RUSTSEC-2024-0384` (`instant`, unmaintained),
  reached only transitively through the `minifb` dev-dependency.

## [0.7.0] — 2026-05-27

Decoder entropy and sparse-path refinements.

### Performance

- AVX2 IDCT sparse parity: DC-only and rows/cols-4–7-zero pass-1 /
  pass-2 kernels, matching the NEON sparse fast paths on x86_64.
- Combined AC/DC Huffman LUT — resolves the run/size symbol and the
  magnitude bits in a single table lookup, wired into the baseline scan
  loop and the progressive DC-first / AC-first scans.
- SWAR 32-bit Huffman bit-reader refill — fills the 64-bit accumulator
  four bytes at a time when no `0xFF` stuffing is present (+4–7% on
  natural 4K content across NEON and AVX2).

### Changed

- Extracted a shared canonical Huffman code builder.
- Added a `profiling` Cargo profile for sampling profilers.

## [0.6.0] — 2026-05-25

Decoder SIMD — per-stage kernels on both backends.

### Performance

- NEON IDCT ported from libjpeg-turbo `jidctint`, on an i16 workspace,
  with DC-only and partial-sparse (rows/cols-4–7-zero) fast paths.
- AVX2 `idct_islow` and AVX2 YCbCr→RGB colour conversion.
- NEON YCbCr→RGB colour conversion (BT.601, all eight output layouts).
- NEON + AVX2 fancy (interpolating) chroma upsample kernels
  (`h2_fancy` horizontal, `h2v2_fancy` vertical), dispatched through a
  new `sample` backend module.
- Release profile tightened to `lto = "fat"` + `codegen-units = 1`.
- Net: decode vs `image` on the 4:2:0 path closes from ~0.41× to
  ~0.77×.

## [0.5.1] — 2026-05-25

### Changed

- Version bump and crate-metadata refresh; no functional change.

## [0.5.0] — 2026-05-25

Encoder differentiation features.

### Added

- Two-pass optimized Huffman tables (T.81 K.2/K.3 from per-image symbol
  frequencies) via `set_optimize_huffman`.
- Multi-threaded MCU-row encode via rayon (opt-in `set_threads`),
  bit-identical output across thread counts.
- Restart-interval option (`set_restart_interval` — DRI + RSTn emission).
- Custom quantization-table API (`set_quant_tables`).
- Tolerant `read_marker` on decode — skips stray inter-segment bytes.

### Fixed

- Include K.3-truncated symbols in the optimized-Huffman `HUFFVAL`.

### Changed

- Unified benchmark harness with section dispatch; SAFETY contracts
  documented on the SIMD intrinsic modules.

## [0.4.0] — 2026-05-24

Decoder coverage.

### Added

- Progressive (SOF2) decoder.
- Non-interleaved baseline scan support (multi-SOS streams).
- Fancy chroma upsample on decode (`h2v2` / `h2v1` / `v2`).
- Vendored fixture corpus + comparison harness against `image`.

### Fixed

- Block-count off-by-one for unaligned widths.
- Non-interleaved baseline block count and raster ordering.

### Changed

- SHA-pinned third-party GitHub Actions.

## [0.3.0] — 2026-05-20

### Added

- Baseline JPEG **decoder** (initial).
- Comparison harness against the `image` crate.
- `deny.toml` with a permissive-license allowlist.

### Fixed

- Reject zero and oversized image dimensions in SOF parsing.
- Propagate `UnexpectedEof` from the restart-handling loop.

## [0.2.1] — 2026-05-20

### Changed

- Documentation catch-up for the 0.2.0 surface and the Huffman SIMD
  bitmap.

## [0.2.0] — 2026-05-20

### Added

- 4:2:2 chroma subsampling (scalar reference + NEON / AVX2
  `h2v1_downsample`).
- Eight input pixel layouts: RGB, BGR, RGBA, BGRA, ARGB, ABGR, RGBX,
  BGRX.

### Performance

- Bitmap-driven AC scan: a `u64` nonzero bitmap collapses zero runs into
  a single jump per nonzero, with a NEON nonzero-bitmap kernel and an
  SSE2 kernel (`pcmpeqw + packsswb + pmovmskb`) on x86_64.

## [0.1.0] — 2026-05-06

Initial release — a SIMD-accelerated baseline JPEG **encoder** in pure
Rust, with kernels translated from libjpeg-turbo and bit-exact against a
scalar reference.

### Added

- Baseline (SOF0) encoder: 4:4:4 / 4:2:0 subsampling, quality-driven
  Annex K quantization, scalar Huffman entropy coding.
- Per-architecture kernel backends (`src/arch/`), selected at compile
  time with a runtime AVX2 check on x86_64.
- NEON (aarch64) kernels: RGBA→YCbCr colour, forward DCT, quantize,
  4:2:0 chroma downsample.
- AVX2 (x86_64) kernels: RGBA→YCbCr colour, forward DCT, quantize,
  4:2:0 chroma downsample.
- `SamplingScheme` trait for chroma-subsampling dispatch.
- CI (test matrix including Windows x86_64 / aarch64), rustdoc, and
  project docs (README, CONTRIBUTING, `docs/ARCHITECTURE.md`, NOTICE).
- Rust 2024 edition with unsafe-block enforcement.

[0.8.0]: https://github.com/naoto256/jpeg-rusturbo/releases/tag/v0.8.0
[0.7.5]: https://github.com/naoto256/jpeg-rusturbo/releases/tag/v0.7.5
[0.7.0]: https://github.com/naoto256/jpeg-rusturbo/releases/tag/v0.7.0
[0.6.0]: https://github.com/naoto256/jpeg-rusturbo/releases/tag/v0.6.0
[0.5.1]: https://github.com/naoto256/jpeg-rusturbo/releases/tag/v0.5.1
[0.5.0]: https://github.com/naoto256/jpeg-rusturbo/releases/tag/v0.5.0
[0.4.0]: https://github.com/naoto256/jpeg-rusturbo/releases/tag/v0.4.0
[0.3.0]: https://github.com/naoto256/jpeg-rusturbo/releases/tag/v0.3.0
[0.2.1]: https://github.com/naoto256/jpeg-rusturbo/releases/tag/v0.2.1
[0.2.0]: https://github.com/naoto256/jpeg-rusturbo/releases/tag/v0.2.0
[0.1.0]: https://github.com/naoto256/jpeg-rusturbo/releases/tag/v0.1.0
