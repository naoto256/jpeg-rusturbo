# Architecture

`jpeg-rusturbo` is a baseline + progressive (SOF2) JPEG encoder — with
EXIF / ICC metadata pass-through — and a baseline + progressive decoder.
The public surface stays small (`JpegEncoder` / `ChromaSubsampling` /
`PixelFormat` on the encode side, `Decoder` / `decode` / `PixelFormat` /
`ImageInfo` on the decode side); the rest of the crate is the `encode`
and `decode` pipelines plus per-architecture kernel backends shared
between them. The two pipelines mirror each other: `src/encode/` and
`src/decode/` are sibling module trees, with `color.rs` / `tables.rs`
and `arch/` as the shared core between them.

## File layout

```
src/
├── lib.rs                  — public API (JpegEncoder re-export,
│                             PixelFormat, ChromaSubsampling, the
│                             PixelFormat → PixelLayout bridge)
├── color.rs                — shared: colour constants + PixelLayout;
│                             encode-side block / MCU extraction (edge
│                             replication). Used by encode, decode, arch.
├── tables.rs               — shared: JPEG Annex K standard tables + ZIGZAG
├── encode/
│   ├── mod.rs              — JpegEncoder + encode-side pipeline
│   │                         orchestration, SamplingScheme trait +
│   │                         Yuv444 / Yuv422 / Yuv420 impls
│   ├── quant.rs            — Divisors, build_divisors,
│   │                         compute_reciprocal, zig-zag wrapper
│   ├── huffman.rs          — HuffmanTable, BitWriter, encode_block,
│   │                         magnitude_category (encode side)
│   ├── huffman_optimize.rs — two-pass optimized Huffman table builder
│   ├── progressive.rs      — progressive (SOF2) encode-side pipeline:
│   │                         8-scan successive-approximation plan
│   │                         (DC first / AC first per-component / DC
│   │                         refine / AC refine), four encoder kernels
│   └── markers.rs          — JPEG segment writers (SOI, APP0, APP1
│                             EXIF, APP2 ICC multi-segment, DQT,
│                             SOF0 / SOF2, DHT, SOS / SOS-spectral,
│                             EOI)
├── decode/
│   ├── mod.rs              — public Decoder + decode() entry points,
│   │                         compose_output (plane → RGB)
│   ├── baseline.rs         — baseline Huffman scan + dequant + IDCT
│   ├── progressive.rs      — progressive (multi-scan) decode
│   ├── huffman.rs          — BitReader, HuffmanDecodeTable, combined
│   │                         AC/DC LUTs, SWAR refill
│   ├── markers.rs          — JPEG marker parser (header chain)
│   └── error.rs            — DecodeError / Result
└── arch/
    ├── mod.rs              — cfg dispatch hub, picks the active backend
    ├── scalar.rs           — bit-exact scalar reference for every
    │                         hot kernel (always compiled)
    ├── neon.rs             — AArch64 NEON kernels (compiled on aarch64)
    └── x86_64.rs           — x86_64 AVX2 kernels (compiled on x86_64)

benches/
├── pipeline.rs            — encode + decode micro-bench harness used
│                            to produce BENCH.md
└── vs_image.rs            — cross-crate comparison vs the `image` crate
```

Each `arch::<backend>` exposes five inline modules — `color`, `dct`,
`quant`, `huffman`, `sample` — with bit-exact-equivalent signatures
to `arch::scalar`. `sample` hosts chroma upsample / downsample
kernels (e.g. `h2v2_fancy_vblend`, `h2_fancy_upsample`,
`h2v2_downsample`). The decoder uses `dct::idct_islow` + `color::
ycc_row_to_rgb` + `sample::*` upsample; the encoder uses
`dct::fdct_islow` + `color::rgb_row_to_ycc` + `sample::*` downsample
+ `quant::quantize_natural` + `huffman::group_of_8_is_zero`.

## Encode pipeline

```
RGB(A) bytes
     │ color::extract_block_ycbcr / extract_mcu_420   (orchestration)
     │   └─ arch::backend::color::rgb_row_to_ycc      (per-row hot path)
     ▼
8x8 i16 blocks (level-shifted to centered range)
     │ arch::backend::dct::fdct_islow                  (12-mul integer LL&M DCT)
     ▼
8x8 i16 DCT coefficients (scaled by 8 vs true DCT)
     │ quant::quantize_and_zigzag
     │   └─ arch::backend::quant::quantize_natural    (recip-mul) + scalar zig-zag
     ▼
8x8 i16 zig-zag-ordered quantized coefficients
     │ huffman::encode_block
     │   └─ arch::backend::huffman::group_of_8_is_zero (8-skip in AC RLE)
     ▼
entropy-coded bytes (with 0xFF → 0xFF 0x00 stuffing)
```

Top-level orchestration happens in `JpegEncoder::encode_inner`
(`src/encode/mod.rs`). It validates dimensions, builds the per-component
quant divisors and Huffman tables, writes the marker prologue (SOI /
APP0 / optional APP1 EXIF + APP2 ICC / DQT / SOF0 / DHT / SOS), then
dispatches the entropy-coded segment to the right `SamplingScheme`
impl. After the scan it flushes the bitwriter and writes EOI. When
progressive output is requested the SOF2 multi-scan plan in
`src/encode/progressive.rs` takes over after the shared prologue.

The hot kernels live behind `arch::backend::*` and are addressed
by name: `color::rgb_row_to_ycc`, `color::ycc_row_to_rgb` (decoder
counterpart), `dct::fdct_islow`, `dct::idct_islow` (decoder
counterpart), `quant::quantize_natural`,
`huffman::group_of_8_is_zero`, `sample::h2v2_downsample` /
`sample::h2v2_fancy_vblend` / `sample::h2_fancy_upsample`. Each
backend implementation is
internally stand-alone — no shared trait — but they all expose the
same function signatures, and cross-check tests in `arch::neon::tests`
and `arch::x86_64::tests` assert bit-exact equality with the scalar
reference on a panel of inputs.

## Backend selection

```rust
// arch/mod.rs (sketch)
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
pub use neon as backend;

#[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
pub use x86_64 as backend;

#[cfg(any(
    feature = "force-scalar",
    not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
pub use scalar as backend;
```

`scalar` is always compiled (the cross-check tests reach for it from
the SIMD backends). `neon` / `x86_64` are gated by `target_arch` so
unrelated arches never see the intrinsic-using code.

On x86_64, the public `arch::x86_64::*::fn_name` wrappers gate the
AVX2 fast path with `std::arch::is_x86_feature_detected!("avx2")` and
fall through to the scalar reference if AVX2 is absent. The detection
result is cached on first call.

## Subsampling dispatch

`ChromaSubsampling` is an enum spanning the three baseline-JPEG
sampling layouts the encoder produces (`Yuv444`, `Yuv422`, `Yuv420`);
each variant has a corresponding zero-sized scheme type implementing
the `SamplingScheme` trait:

```rust
trait SamplingScheme {
    const H_V: (u8, u8);            // SOF0 component descriptor (h, v) for Y
    const MCU_W: u32;
    const MCU_H: u32;
    fn encode_one_mcu<W: Write>(...) -> io::Result<()>;
}
```

`encode_scan<S: SamplingScheme>` is generic over the scheme and
collapses what would have been five near-duplicate `encode_scan_NNN`
functions into one MCU loop. Adding a new scheme is:

1. `impl SamplingScheme for Yuv<NNN>Scheme { ... }`
2. `ChromaSubsampling::Yuv<NNN>` variant
3. one match arm in each of the two dispatch sites (SOF0 + scan)
4. an `extract_mcu_NNN` analog of `extract_mcu_420` in `color.rs`

…with no edits to the scan loop itself.

## Adding a new arch backend

1. Create `src/arch/<name>.rs` with five inline modules — `color`,
   `dct`, `quant`, `huffman`, `sample` — each exposing the same
   kernel functions named in `arch::scalar`. Use
   `pub use crate::arch::scalar::<kernel>::*;` for any kernel you
   don't override.
2. Declare the module in `arch/mod.rs` under the appropriate
   `#[cfg(target_arch = "...")]`.
3. Add a `pub use <name> as backend;` cfg arm so it gets selected.
4. Update `benches/pipeline.rs`'s `arch` label to print the right string.
5. Mirror the cross-check tests pattern from `arch::neon::tests` /
   `arch::x86_64::tests` (compare each kernel against scalar on a
   panel of inputs).

## Bit-exact contract

Across `arch::scalar`, `arch::neon`, and `arch::x86_64`, the encoded
JPEG byte stream is identical for the same input. This is enforced by:

- **Cross-check unit tests** at the kernel level
  (`color_neon_matches_scalar_*`, `fdct_avx2_matches_scalar_*`,
  `quant_neon_matches_scalar`, `h2v2_downsample_avx2_matches_scalar_*`,
  …).
- **Roundtrip integration tests** in `tests/roundtrip.rs` that decode
  the produced JPEG via the `image` crate's decoder and compare
  pixel-by-pixel for solid colors, and assert equal byte counts /
  similar pixel magnitudes for content with detail.

The kernels' arithmetic is intentionally identical down to integer
rounding — the same FIX_xxx 13-bit fixed-point constants from the
JPEG spec, the same libjpeg-turbo bias patterns (e.g.
`{1, 2, 1, 2, …}` for 4:2:0 chroma rounding), the same
reciprocal-multiply quantize formulation. `compute_reciprocal` emits
both the `shift` field used by scalar/NEON and the `scale` field used
by AVX2 (which fakes per-lane variable shift via a second `vpmulhuw`);
the two paths are algebraically equivalent.

## Testing layout

```
tests/roundtrip.rs    — integration tests via the `image` decoder
src/<file>::tests     — algorithmic unit tests (in-module)
src/arch/neon.rs      — NEON-vs-scalar cross-check tests (cfg aarch64)
src/arch/x86_64.rs    — AVX2-vs-scalar cross-check tests (cfg x86_64)
```

`cargo test --release` exercises everything at the active arch
backend. `cargo test --release --features force-scalar` exercises the
scalar fallback on the same machine. Together they cover every
backend the host can reach.

## What lives where (quick lookup)

| Want to change … | File |
|---|---|
| Quality scaling, standard tables | `src/tables.rs` |
| Encode MCU iteration, scan-level dispatch | `src/encode/mod.rs` |
| 8x8 / 16x16 block extraction, padding, level shift | `src/color.rs` |
| Quant divisor construction (`compute_reciprocal`) | `src/encode/quant.rs` |
| Encode-side bit-stuffing, byte-stuffing, Huffman emission | `src/encode/huffman.rs` |
| Two-pass optimized Huffman table builder | `src/encode/huffman_optimize.rs` |
| Progressive (SOF2) encode (8-scan SA plan, DC/AC first + refine) | `src/encode/progressive.rs` |
| Decode entry points (`Decoder`, `decode`) | `src/decode/mod.rs` |
| Baseline scan, dequant fusion, IDCT dispatch | `src/decode/baseline.rs` |
| Progressive (multi-scan) decode | `src/decode/progressive.rs` |
| Decode-side BitReader, combined AC/DC LUTs, SWAR refill | `src/decode/huffman.rs` |
| Header chain parser (SOI / DQT / DHT / SOF / SOS / EOI) | `src/decode/markers.rs` |
| Per-arch hot kernels (5 modules: color / dct / quant / huffman / sample) | `src/arch/{scalar,neon,x86_64}.rs` |
| JPEG segment writers (encode side) | `src/encode/markers.rs` |
| Benchmark harness output / labels | `benches/pipeline.rs` |
