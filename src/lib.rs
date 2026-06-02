//! Baseline JPEG encoder + decoder with NEON / AVX2 / SSE2 / scalar
//! SIMD backends. The encoder is a drop-in for
//! [`image::codecs::jpeg::JpegEncoder`]; the decoder is a standalone
//! [`decode::Decoder`] under the `decode` module.
//!
//! [`image::codecs::jpeg::JpegEncoder`]: https://docs.rs/image/latest/image/codecs/jpeg/struct.JpegEncoder.html
//!
//! # Quick start — encode
//!
//! ```
//! use jpeg_rusturbo::JpegEncoder;
//!
//! let pixels = vec![128u8; 16 * 16 * 4]; // 16x16 mid-gray RGBA
//! let mut out: Vec<u8> = Vec::new();
//! let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
//! enc.encode_rgba(&pixels, 16, 16)?;
//! assert_eq!(&out[..2], &[0xFF, 0xD8]); // SOI marker
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! # Quick start — decode
//!
//! ```
//! use jpeg_rusturbo::{JpegEncoder, PixelFormat, decode};
//!
//! # let pixels = vec![128u8; 16 * 16 * 4];
//! # let mut jpeg = Vec::new();
//! # JpegEncoder::new_with_quality(&mut jpeg, 80).encode_rgba(&pixels, 16, 16)?;
//! let rgb = decode::decode(&jpeg, PixelFormat::Rgb)?;
//! assert_eq!(rgb.len(), 16 * 16 * 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Choosing chroma subsampling
//!
//! ```
//! use jpeg_rusturbo::{ChromaSubsampling, JpegEncoder};
//!
//! let mut out: Vec<u8> = Vec::new();
//! let mut enc = JpegEncoder::new_with_quality(&mut out, 90);
//! enc.set_subsampling(ChromaSubsampling::Yuv444); // higher chroma fidelity
//! # let pixels = vec![0u8; 8 * 8 * 4];
//! # enc.encode_rgba(&pixels, 8, 8)?;
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! Default is [`ChromaSubsampling::Yuv420`] (matches what mainstream
//! JPEG encoders ship). [`ChromaSubsampling::Yuv444`] preserves chroma
//! at full resolution, useful for graphic / text content.
//!
//! # Performance
//!
//! This crate exists because a real workload needed a fast JPEG
//! encoder in pure Rust — `image`'s bundled encoder is scalar and
//! 4:2:0-only, so encode-heavy pipelines leave throughput on the
//! table. Per-architecture SIMD kernels (NEON on aarch64, AVX2 +
//! SSE2 on x86_64) are translated from libjpeg-turbo with bit-exact
//! output guarantees against the scalar reference. Versus the
//! `image` crate's scalar encoder, jpeg-rusturbo's encoder is
//! ~4.5× on Apple M and ~5.5× on Cascade Lake at 4K 4:2:0 q=80
//! (up from ~2.9× / ~3.3× in 0.7.5 — the 0.8.0 encoder hot-path
//! pass added unsafe `BitWriter::drain_high32`, NEON
//! `vqtbl4q`-based zig-zag scatter, fused AC code+magnitude
//! `write_bits`, a one-shot SIMD precompute of the JPEG magnitude
//! category, and an AVX2 3-byte RGB→YCbCr deinterleave kernel; the
//! `drain_high32` + AC fusion help both backends, the zig-zag
//! scatter + magnitude precompute are NEON-only, the RGB color
//! kernel is AVX2-only. The Cascade ratio runs higher than Apple's
//! mainly because `image`'s scalar encoder is slower on that CPU).
//! Opt-in [`JpegEncoder::set_threads`] adds another 1.2–2.4× on
//! top via MCU-row parallelism; opt-in
//! [`JpegEncoder::set_optimize_huffman`] trims output size ~5%
//! across subsampling/quality at ~1.7× (AVX2) / ~2.3× (NEON)
//! encode cost. Encode speed is the headline.
//!
//! 0.8.0 also adds two encoder-surface features:
//! [`JpegEncoder::set_progressive`] for SOF2 output (8-scan
//! successive-approximation plan covering all four progressive
//! scan types) and [`JpegEncoder::set_exif`] /
//! [`JpegEncoder::set_icc_profile`] for APP1 / APP2 metadata
//! pass-through. Both are off by default — when neither is called,
//! the encoder's output is bit-identical to a build that doesn't
//! know about them.
//!
//! 0.9.0 closes the non-perf coverage gaps with four additions:
//! [`JpegEncoder::encode_grayscale`] / [`PixelFormat::Gray`] for
//! 1-byte-per-pixel luma-only input and decode-side Y-plane
//! extraction; [`JpegEncoder::encode_cmyk`] / [`PixelFormat::Cmyk`]
//! for 4-byte CMYK pass-through (no CMYK↔RGB conversion in either
//! direction); [`decode::Decoder::exif`] and
//! [`decode::Decoder::icc_profile`] symmetric with the 0.8.0
//! encoder-side pass-through (multi-segment ICC reassembled lazily,
//! EXIF returned zero-copy with the `Exif\0\0` identifier stripped);
//! and [`JpegEncoder::set_optimize_huffman`] composing with
//! [`JpegEncoder::set_progressive`] (counts symbol frequencies
//! per scan, builds per-scan custom Huffman tables that include
//! `EOBn` symbols, packs multi-block end-of-band runs — the
//! resulting progressive output is **smaller** than the baseline
//! SOF0 equivalent rather than ~+45% larger). No kernel changes,
//! no perf regression for the 8 RGB-family layouts; default
//! behaviour is byte-identical to 0.8.0.
//!
//! The decoder is bundled for API symmetry — read your own JPEGs
//! back without reaching for another crate — rather than as a speed
//! play. It gained per-stage SIMD kernels in 0.6.0 (IDCT, YCC → RGB
//! color convert, and fancy chroma upsample in NEON + AVX2). As of
//! 0.7.5 (entropy + dequant fusion, AVX2 PSHUFB RGB interleave,
//! uninit `Vec` allocation) it sits ahead of `image` at 4K on both
//! microarchitectures and both corpora (~1.03–1.10× on synthetic
//! Huffman-heavy content, ~1.18–1.22× on natural-content), while
//! matching coverage — baseline + progressive Huffman, fancy chroma
//! upsample, all ten pixel layouts (the eight RGB-family layouts
//! plus `Gray` and `Cmyk` added in 0.9.0). 0.8.0 doesn't touch the
//! decoder; the above carries over. The IDCT carries DC-only and
//! sparse-row fast paths that fire on smooth regions in natural
//! photographs (+11–19% of total decode time on natural content,
//! noise-level on synthetic input); 0.7.0 ported those fast paths
//! to AVX2 to match NEON. The Huffman entropy decoder is scalar by
//! design — the bit-reader + canonical-table walk has a serial
//! dependency on per-symbol code length that doesn't reshape into
//! vector SIMD — and 0.7.0 lands two scalar bit-ops refinements on
//! top: a combined run/size + magnitude LUT (table-driven path,
//! used by both AC and DC including progressive scans) and a SWAR
//! 32-bit bit-reader refill that fills the `u64` accumulator four
//! bytes at a time when no `0xFF` byte stuffing is present. The
//! SWAR refill delivers +4–7% on natural 4K content across both
//! NEON and AVX2; the combined LUT sits at the noise floor at q=80
//! and is retained as a
//! bit-exact foundation. See [`BENCH.md`] in the repository for
//! detailed numbers.
//!
//! [`BENCH.md`]: https://github.com/naoto256/jpeg-rusturbo/blob/main/BENCH.md
//!
//! On x86_64, AVX2 dispatch is gated by a runtime
//! `is_x86_feature_detected!("avx2")` check; CPUs without AVX2 fall
//! through to the scalar path automatically.
//!
//! The `force-scalar` Cargo feature opts every target out of SIMD
//! (used by the bench harness to compare scalar vs SIMD on the same
//! hardware).
//!
//! # Implementation notes
//!
//! Hot kernels live behind `arch::backend::*` selected at compile
//! time. The encode pipeline:
//!
//! ```text
//!   RGB(A) bytes
//!       │ block / MCU extraction (orchestration)
//!       │   └─ arch::backend::color::rgb_row_to_ycc
//!       ▼
//!   8x8 i16 blocks (level-shifted)
//!       │ arch::backend::dct::fdct_islow      (12-mul integer LL&M DCT)
//!       ▼
//!   8x8 i16 DCT coefficients (scaled by 8)
//!       │ quantize + zig-zag
//!       │   └─ arch::backend::quant::quantize_natural
//!       ▼
//!   8x8 i16 zig-zag coefficients
//!       │ Huffman entropy code (bitmap-driven AC scan)
//!       │   └─ arch::backend::huffman::nonzero_bitmap
//!       ▼
//!   entropy-coded bytes (with 0xFF → 0xFF 0x00 stuffing)
//! ```
//!
//! The decode pipeline mirrors this in reverse: marker parser →
//! Huffman decode (bit reader plus canonical-Huffman LUT) →
//! de-zig-zag and dequantize → `arch::backend::dct::idct_islow` →
//! chroma upsample → YCbCr→RGB.
//!
//! See [`docs/ARCHITECTURE.md`] for the full internal design and the
//! "adding a new arch backend" recipe.
//!
//! [`docs/ARCHITECTURE.md`]: https://github.com/naoto256/jpeg-rusturbo/blob/main/docs/ARCHITECTURE.md
//!
//! # Attribution
//!
//! The SIMD kernels are translations of libjpeg-turbo (BSD-3-Clause +
//! IJG). See `NOTICE.md` in the repository for the full attribution.

mod arch;
mod color;
pub mod decode;
mod encode;
mod tables;

use crate::color::{ABGR, ARGB, BGR, BGRA, BGRX, CMYK, GRAY, PixelLayout, RGB, RGBA, RGBX};

pub use crate::encode::JpegEncoder;

/// Chroma subsampling mode for the encoded JPEG.
///
/// JPEG stores Y separately from Cb / Cr. Subsampling reduces chroma
/// resolution because the human visual system is much more sensitive
/// to luma than chroma; trading chroma detail for smaller files is
/// usually invisible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaSubsampling {
    /// 4:4:4 — Y, Cb, Cr all at full resolution.
    ///
    /// No chroma loss. Bigger files. Right choice for synthetic
    /// content (text, line art, screenshots) where chroma edges
    /// matter.
    Yuv444,
    /// 4:2:2 — Y at full resolution, Cb / Cr at half horizontal
    /// resolution (one chroma sample per 2×1 luma pair). Preserves
    /// vertical chroma fidelity; common in video and broadcast
    /// pipelines.
    Yuv422,
    /// 4:2:0 — Y at full resolution, Cb / Cr at half resolution in
    /// both axes (one chroma sample per 2×2 luma quad).
    ///
    /// Default. What most cameras and image software produce. Roughly
    /// 1.5× smaller than 4:4:4 at the same quality knob, with no
    /// visible loss on natural-scene photographs.
    Yuv420,
}

/// Pixel format used at both the encode-input and decode-output
/// boundary.
///
/// For the 3- / 4-byte color formats, JPEG stores Y/Cb/Cr internally,
/// so the alpha or pad byte in 4-byte formats is read and then
/// discarded by the encoder (and synthesized as `0xFF` opaque by the
/// decoder, when requested).
///
/// [`PixelFormat::Gray`] is single-byte (1 bpp): the byte *is* Y. As
/// an encoder input it produces a single-component (luma-only) JPEG —
/// see [`JpegEncoder::encode_grayscale`]. As a decoder output it
/// returns the Y plane verbatim, regardless of whether the source
/// JPEG was 1-component grayscale or 3-component color (in the color
/// case, chroma is discarded — a fast Y-extraction shortcut).
///
/// [`PixelFormat::Cmyk`] is 4 bytes/pixel (C, M, Y, K) and is a
/// **pass-through**: the encoder takes raw CMYK bytes and emits a
/// 4-component baseline JPEG without any CMYK↔RGB conversion (no
/// per-channel transform of any kind); the decoder reads such a JPEG
/// back into the same C/M/Y/K byte order. Decoding a 4-component
/// (CMYK) source into any other [`PixelFormat`] is rejected with
/// `Unsupported` — this crate does not perform CMYK→RGB conversion.
/// Adobe-flavoured YCCK (signalled via APP14) is intentionally out of
/// scope and treated as plain CMYK regardless of the APP14 transform
/// byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// 3 bytes per pixel, in order (R, G, B).
    Rgb,
    /// 3 bytes per pixel, in order (B, G, R).
    Bgr,
    /// 4 bytes per pixel, in order (R, G, B, A). Alpha dropped.
    Rgba,
    /// 4 bytes per pixel, in order (B, G, R, A). Alpha dropped.
    Bgra,
    /// 4 bytes per pixel, in order (A, R, G, B). Alpha dropped.
    Argb,
    /// 4 bytes per pixel, in order (A, B, G, R). Alpha dropped.
    Abgr,
    /// 4 bytes per pixel, in order (R, G, B, X). Pad byte ignored.
    Rgbx,
    /// 4 bytes per pixel, in order (B, G, R, X). Pad byte ignored.
    Bgrx,
    /// 1 byte per pixel — the byte is Y (luma) directly.
    ///
    /// On the encode side, see [`JpegEncoder::encode_grayscale`] —
    /// produces a 1-component (luma-only) JPEG, no chroma DQT / DHT /
    /// SOF / SOS overhead.
    ///
    /// On the decode side, returns the Y plane verbatim with no
    /// chroma upsample and no color convert — works for both
    /// 1-component grayscale source JPEGs and 3-component color
    /// sources (in the color case, Cb/Cr are simply discarded).
    Gray,
    /// 4 bytes per pixel — raw (C, M, Y, K), pass-through.
    ///
    /// On the encode side, see [`JpegEncoder::encode_cmyk`] —
    /// produces a 4-component baseline JPEG with no CMYK↔RGB
    /// conversion, no APP14 marker, and no Adobe YCCK transform.
    ///
    /// On the decode side, accepted only against a 4-component
    /// (CMYK) source JPEG; decoding a CMYK source into any non-CMYK
    /// `PixelFormat` returns `DecodeError::Unsupported`, and
    /// decoding a 3-component source into `PixelFormat::Cmyk`
    /// likewise returns `Unsupported`. APP14-flagged YCCK files are
    /// read as plain CMYK (the YCCK transform is intentionally not
    /// applied).
    Cmyk,
}

impl From<PixelFormat> for PixelLayout {
    fn from(f: PixelFormat) -> Self {
        match f {
            PixelFormat::Rgb => RGB,
            PixelFormat::Bgr => BGR,
            PixelFormat::Rgba => RGBA,
            PixelFormat::Bgra => BGRA,
            PixelFormat::Argb => ARGB,
            PixelFormat::Abgr => ABGR,
            PixelFormat::Rgbx => RGBX,
            PixelFormat::Bgrx => BGRX,
            PixelFormat::Gray => GRAY,
            PixelFormat::Cmyk => CMYK,
        }
    }
}
