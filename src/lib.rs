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
//! Per-architecture SIMD kernels (NEON on aarch64, AVX2 + SSE2 on
//! x86_64) are translated from libjpeg-turbo with bit-exact output
//! guarantees against the scalar reference. Encoder whole-pipeline
//! speedup vs scalar is ~1.5× on Apple Silicon and ~2.0× on Intel
//! Ice Lake at 1080p / 4K, q=80. Versus the `image` crate's
//! scalar encoder, jpeg-rusturbo's encoder is ~2.5× / ~3.9× faster
//! (Apple M / Ice Lake). The decoder is scalar by design — it lags
//! `image`'s SIMD decoder by ~2.5× but matches its coverage
//! (baseline + progressive Huffman, fancy chroma upsample, all eight
//! pixel layouts). Decoder SIMD is scheduled for 0.6.0. See
//! [`BENCH.md`] in the repository for detailed numbers.
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
mod huffman;
mod markers;
mod quant;
mod tables;

use std::io::{self, Write};

use crate::color::{ABGR, ARGB, BGR, BGRA, BGRX, PixelLayout, RGB, RGBA, RGBX};
use crate::huffman::{BitWriter, HuffmanTable, encode_block};
use crate::quant::Divisors;
use crate::tables::{
    STD_CHROMA_AC, STD_CHROMA_DC, STD_CHROMA_QUANT, STD_LUMA_AC, STD_LUMA_DC, STD_LUMA_QUANT,
    scale_quant_table,
};

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

/// Source pixel format for the generic [`JpegEncoder::encode`] entry
/// point.
///
/// JPEG stores Y/Cb/Cr internally, so the alpha or pad byte in 4-byte
/// formats is read and then discarded by the encoder.
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
        }
    }
}

/// JPEG encoder over an arbitrary [`Write`] sink.
///
/// Public API mirrors [`image::codecs::jpeg::JpegEncoder`] so call
/// sites can be ported by changing the `use` line.
///
/// [`image::codecs::jpeg::JpegEncoder`]: https://docs.rs/image/latest/image/codecs/jpeg/struct.JpegEncoder.html
///
/// # Examples
///
/// ```
/// use jpeg_rusturbo::JpegEncoder;
///
/// let mut out: Vec<u8> = Vec::new();
/// let mut enc = JpegEncoder::new_with_quality(&mut out, 75);
/// # let pixels = vec![0u8; 8 * 8 * 3];
/// enc.encode_rgb(&pixels, 8, 8)?;
/// # Ok::<(), std::io::Error>(())
/// ```
///
/// Each `encode_*` call produces a complete, self-contained JPEG
/// stream (SOI … EOI). Calling `encode_*` more than once on the same
/// encoder will produce concatenated streams in the sink, which is
/// almost certainly not what you want — construct a fresh
/// `JpegEncoder` per image.
/// One pair of (luma, chroma) quantization tables in natural order.
/// Boxed so the encoder struct stays cheap to move around.
type QuantPair = (Box<[u8; 64]>, Box<[u8; 64]>);

pub struct JpegEncoder<W: Write> {
    out: W,
    quality: u8,
    subsampling: ChromaSubsampling,
    restart_interval: u16,
    custom_quant: Option<QuantPair>,
}

impl<W: Write> JpegEncoder<W> {
    /// Create an encoder at the given quality, writing to `out`.
    ///
    /// `quality` is the conventional JPEG quality knob: 1..=100,
    /// clamped to that range. 1 is the smallest / lowest fidelity, 100
    /// is the largest / highest. Most workflows use 70-90; defaults to
    /// 80 are common.
    ///
    /// Subsampling defaults to [`ChromaSubsampling::Yuv420`]; override
    /// with [`set_subsampling`](Self::set_subsampling) before calling
    /// any `encode_*` method.
    pub fn new_with_quality(out: W, quality: u8) -> Self {
        Self {
            out,
            quality: quality.clamp(1, 100),
            subsampling: ChromaSubsampling::Yuv420,
            restart_interval: 0,
            custom_quant: None,
        }
    }

    /// Override the chroma subsampling mode for this encoder. Must be
    /// called before any `encode_*`; the value is read once at the
    /// start of encoding.
    pub fn set_subsampling(&mut self, s: ChromaSubsampling) {
        self.subsampling = s;
    }

    /// Override the per-component quantization tables. Values are
    /// `u8` (8-bit precision, the only one we emit) in **natural row-
    /// major order** — index 0 = DC, index 63 = highest-frequency AC.
    /// The encoder writes them out in zig-zag order in the DQT segment
    /// automatically.
    ///
    /// When set, `set_quality()` is bypassed entirely — the supplied
    /// tables go through verbatim. To recover the default behavior
    /// (Annex K + quality scaling) call [`clear_quant_tables`].
    ///
    /// Each entry must be in `1..=255` (a zero quantizer divides DCT
    /// coefficients by zero and is rejected by every conforming
    /// decoder). The encoder doesn't validate the range — pass values
    /// you actually want emitted.
    ///
    /// Intended for advanced workflows: ML-driven RDO, mozjpeg /
    /// libjpeg-turbo table replication, per-image perceptual tuning.
    pub fn set_quant_tables(&mut self, luma: [u8; 64], chroma: [u8; 64]) {
        self.custom_quant = Some((Box::new(luma), Box::new(chroma)));
    }

    /// Clear any custom quantization tables previously installed via
    /// [`set_quant_tables`]; subsequent encodes use the Annex K +
    /// quality-scaled defaults again.
    pub fn clear_quant_tables(&mut self) {
        self.custom_quant = None;
    }

    /// Emit an `RSTn` restart marker every `interval` MCUs. Restart
    /// markers let downstream tools resync the entropy stream at known
    /// byte-aligned positions — they're how parallel JPEG decoders
    /// partition the work across threads. `0` (the default) disables
    /// restart markers and skips the DRI segment in the output.
    ///
    /// Cost: a 2-byte RSTn marker + DC-predictor reset every `interval`
    /// MCUs. Typical values are 1-256 (per-row or smaller). Setting
    /// this higher than the total MCU count produces a single RSTn at
    /// the end of the scan, effectively a no-op.
    pub fn set_restart_interval(&mut self, interval: u16) {
        self.restart_interval = interval;
    }

    /// Encode an RGB pixel buffer (3 bytes/pixel) as a complete JPEG
    /// stream into the sink.
    ///
    /// `pixels` is treated as `width * height` packed RGB triples in
    /// row-major order. Trailing bytes past `width * height * 3` are
    /// ignored.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] if `width` or `height`
    /// is zero, if `width * height * 3` overflows `usize`, or if the
    /// pixel buffer is shorter than `width * height * 3`. Any I/O
    /// error from the sink is propagated as-is.
    pub fn encode_rgb(&mut self, pixels: &[u8], width: u32, height: u32) -> io::Result<()> {
        self.encode_inner(pixels, width, height, RGB)
    }

    /// Encode an RGBA pixel buffer (4 bytes/pixel) as a complete JPEG
    /// stream into the sink. The alpha channel is dropped (JPEG has
    /// no alpha).
    ///
    /// Compared to `encode_rgb`, this avoids the caller having to
    /// repack RGBA → RGB before calling, saving one full-frame copy
    /// on the common case where image data arrives as RGBA.
    ///
    /// # Errors
    ///
    /// Same as [`encode_rgb`](Self::encode_rgb) but with the size
    /// requirement `width * height * 4`.
    pub fn encode_rgba(&mut self, pixels: &[u8], width: u32, height: u32) -> io::Result<()> {
        self.encode_inner(pixels, width, height, RGBA)
    }

    /// Encode an arbitrary [`PixelFormat`] pixel buffer. Generic entry
    /// point covering all eight supported byte layouts (RGB / BGR /
    /// RGBA / BGRA / ARGB / ABGR / RGBX / BGRX).
    ///
    /// # Errors
    /// Same shape as [`encode_rgb`](Self::encode_rgb) / [`encode_rgba`](Self::encode_rgba),
    /// scaled by `format`'s bytes-per-pixel.
    pub fn encode(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> io::Result<()> {
        self.encode_inner(pixels, width, height, format.into())
    }

    fn encode_inner(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
    ) -> io::Result<()> {
        if width == 0 || height == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "image dimensions must be non-zero",
            ));
        }
        // `width * height * bpp` must fit in `usize`. On 64-bit hosts
        // `u32 * u32 ≤ u64::MAX` so the check is mostly belt-and-braces;
        // on 32-bit (e.g. wasm32) it's load-bearing — without it a wrapped
        // small `needed` could allow oversized inputs through and OOB the
        // pixel buffer downstream.
        let needed = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(layout.bpp))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "image dimensions overflow usize (width * height * bpp)",
                )
            })?;
        if pixels.len() < needed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "pixel buffer too small: {} < {} ({}x{}*{})",
                    pixels.len(),
                    needed,
                    width,
                    height,
                    layout.bpp
                ),
            ));
        }

        // Quant tables (8-bit, scaled by quality, OR user-supplied via
        // `set_quant_tables`). Index 0 = luma, 1 = chroma.
        let (luma_q, chroma_q) = match &self.custom_quant {
            Some((l, c)) => (**l, **c),
            None => (
                scale_quant_table(&STD_LUMA_QUANT, self.quality),
                scale_quant_table(&STD_CHROMA_QUANT, self.quality),
            ),
        };
        let div_luma = quant::build_divisors(&luma_q);
        let div_chroma = quant::build_divisors(&chroma_q);

        // Expand the standard Huffman tables into encoder lookups.
        let dc_luma = HuffmanTable::from_std(&STD_LUMA_DC);
        let ac_luma = HuffmanTable::from_std(&STD_LUMA_AC);
        let dc_chroma = HuffmanTable::from_std(&STD_CHROMA_DC);
        let ac_chroma = HuffmanTable::from_std(&STD_CHROMA_AC);

        // ---- Header ----
        markers::write_soi(&mut self.out)?;
        markers::write_app0_jfif(&mut self.out)?;
        markers::write_dqt(&mut self.out, 0, &luma_q)?;
        markers::write_dqt(&mut self.out, 1, &chroma_q)?;

        let (h_y, v_y) = match self.subsampling {
            ChromaSubsampling::Yuv444 => Yuv444Scheme::H_V,
            ChromaSubsampling::Yuv422 => Yuv422Scheme::H_V,
            ChromaSubsampling::Yuv420 => Yuv420Scheme::H_V,
        };
        markers::write_sof0(
            &mut self.out,
            width as u16,
            height as u16,
            &[
                (1, h_y, v_y, 0), // Y
                (2, 1, 1, 1),     // Cb
                (3, 1, 1, 1),     // Cr
            ],
        )?;

        markers::write_dht(&mut self.out, 0, 0, &STD_LUMA_DC)?;
        markers::write_dht(&mut self.out, 1, 0, &STD_LUMA_AC)?;
        markers::write_dht(&mut self.out, 0, 1, &STD_CHROMA_DC)?;
        markers::write_dht(&mut self.out, 1, 1, &STD_CHROMA_AC)?;

        if self.restart_interval > 0 {
            markers::write_dri(&mut self.out, self.restart_interval)?;
        }

        markers::write_sos(
            &mut self.out,
            &[
                (1, 0, 0), // Y → DC0/AC0
                (2, 1, 1), // Cb → DC1/AC1
                (3, 1, 1), // Cr → DC1/AC1
            ],
        )?;

        // ---- Entropy-coded segment ----
        let mut bw = BitWriter::new(&mut self.out);
        // Crude upper bound: ~1 byte per pixel for q=80 typical content;
        // the bitwriter resizes as needed but starting close avoids the
        // first few reallocations on large frames. Reuse `needed` so we
        // benefit from the same overflow check as the input validation
        // above.
        bw.reserve(needed);
        let restart_interval = self.restart_interval;
        match self.subsampling {
            ChromaSubsampling::Yuv444 => encode_scan::<Yuv444Scheme, _>(
                pixels,
                width,
                height,
                layout,
                &mut bw,
                &div_luma,
                &div_chroma,
                &dc_luma,
                &ac_luma,
                &dc_chroma,
                &ac_chroma,
                restart_interval,
            )?,
            ChromaSubsampling::Yuv422 => encode_scan::<Yuv422Scheme, _>(
                pixels,
                width,
                height,
                layout,
                &mut bw,
                &div_luma,
                &div_chroma,
                &dc_luma,
                &ac_luma,
                &dc_chroma,
                &ac_chroma,
                restart_interval,
            )?,
            ChromaSubsampling::Yuv420 => encode_scan::<Yuv420Scheme, _>(
                pixels,
                width,
                height,
                layout,
                &mut bw,
                &div_luma,
                &div_chroma,
                &dc_luma,
                &ac_luma,
                &dc_chroma,
                &ac_chroma,
                restart_interval,
            )?,
        }
        bw.flush_to_byte_boundary()?;

        // ---- Trailer ----
        markers::write_eoi(&mut self.out)?;
        Ok(())
    }
}

/// Per-MCU running DC predictor for the three components. Difference-coded
/// across MCUs within a scan (F.1.2.1).
#[derive(Default)]
struct DcPredictors {
    y: i16,
    cb: i16,
    cr: i16,
}

/// One chroma-subsampling scheme (4:4:4, 4:2:0, future: 4:2:2 …).
///
/// Each impl owns its MCU geometry and the per-MCU encode work — the
/// generic `encode_scan` below just iterates MCUs and forwards. Adding a
/// new scheme is "impl this trait + add a variant + add two match arms",
/// no scan-level surgery required (Rule of Three: prep work at 2 instances).
trait SamplingScheme {
    /// `(h_factor, v_factor)` for the Y component in SOF0. Cb/Cr are
    /// always (1, 1) in the schemes we support.
    const H_V: (u8, u8);
    /// One MCU's pixel footprint `(width, height)`.
    const MCU_W: u32;
    const MCU_H: u32;

    #[allow(clippy::too_many_arguments)]
    fn encode_one_mcu<W: Write>(
        bw: &mut BitWriter<W>,
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        mx: u32,
        my: u32,
        prev_dc: &mut DcPredictors,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        dc_luma: &HuffmanTable,
        ac_luma: &HuffmanTable,
        dc_chroma: &HuffmanTable,
        ac_chroma: &HuffmanTable,
    ) -> io::Result<()>;
}

struct Yuv444Scheme;
impl SamplingScheme for Yuv444Scheme {
    const H_V: (u8, u8) = (1, 1);
    const MCU_W: u32 = 8;
    const MCU_H: u32 = 8;

    fn encode_one_mcu<W: Write>(
        bw: &mut BitWriter<W>,
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        mx: u32,
        my: u32,
        prev_dc: &mut DcPredictors,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        dc_luma: &HuffmanTable,
        ac_luma: &HuffmanTable,
        dc_chroma: &HuffmanTable,
        ac_chroma: &HuffmanTable,
    ) -> io::Result<()> {
        let mut y_blk = [0i16; 64];
        let mut cb_blk = [0i16; 64];
        let mut cr_blk = [0i16; 64];
        color::extract_block_ycbcr(
            pixels,
            width,
            height,
            layout,
            mx * Self::MCU_W,
            my * Self::MCU_H,
            &mut y_blk,
            &mut cb_blk,
            &mut cr_blk,
        );
        prev_dc.y = encode_one_block(bw, &mut y_blk, div_luma, prev_dc.y, dc_luma, ac_luma)?;
        prev_dc.cb = encode_one_block(
            bw,
            &mut cb_blk,
            div_chroma,
            prev_dc.cb,
            dc_chroma,
            ac_chroma,
        )?;
        prev_dc.cr = encode_one_block(
            bw,
            &mut cr_blk,
            div_chroma,
            prev_dc.cr,
            dc_chroma,
            ac_chroma,
        )?;
        Ok(())
    }
}

struct Yuv420Scheme;
impl SamplingScheme for Yuv420Scheme {
    const H_V: (u8, u8) = (2, 2);
    const MCU_W: u32 = 16;
    const MCU_H: u32 = 16;

    fn encode_one_mcu<W: Write>(
        bw: &mut BitWriter<W>,
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        mx: u32,
        my: u32,
        prev_dc: &mut DcPredictors,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        dc_luma: &HuffmanTable,
        ac_luma: &HuffmanTable,
        dc_chroma: &HuffmanTable,
        ac_chroma: &HuffmanTable,
    ) -> io::Result<()> {
        let mut y_blocks = [[0i16; 64]; 4];
        let mut cb_blk = [0i16; 64];
        let mut cr_blk = [0i16; 64];
        color::extract_mcu_420(
            pixels,
            width,
            height,
            layout,
            mx * Self::MCU_W,
            my * Self::MCU_H,
            &mut y_blocks,
            &mut cb_blk,
            &mut cr_blk,
        );
        for blk in y_blocks.iter_mut() {
            prev_dc.y = encode_one_block(bw, blk, div_luma, prev_dc.y, dc_luma, ac_luma)?;
        }
        prev_dc.cb = encode_one_block(
            bw,
            &mut cb_blk,
            div_chroma,
            prev_dc.cb,
            dc_chroma,
            ac_chroma,
        )?;
        prev_dc.cr = encode_one_block(
            bw,
            &mut cr_blk,
            div_chroma,
            prev_dc.cr,
            dc_chroma,
            ac_chroma,
        )?;
        Ok(())
    }
}

struct Yuv422Scheme;
impl SamplingScheme for Yuv422Scheme {
    const H_V: (u8, u8) = (2, 1);
    const MCU_W: u32 = 16;
    const MCU_H: u32 = 8;

    fn encode_one_mcu<W: Write>(
        bw: &mut BitWriter<W>,
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        mx: u32,
        my: u32,
        prev_dc: &mut DcPredictors,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        dc_luma: &HuffmanTable,
        ac_luma: &HuffmanTable,
        dc_chroma: &HuffmanTable,
        ac_chroma: &HuffmanTable,
    ) -> io::Result<()> {
        let mut y_blocks = [[0i16; 64]; 2];
        let mut cb_blk = [0i16; 64];
        let mut cr_blk = [0i16; 64];
        color::extract_mcu_422(
            pixels,
            width,
            height,
            layout,
            mx * Self::MCU_W,
            my * Self::MCU_H,
            &mut y_blocks,
            &mut cb_blk,
            &mut cr_blk,
        );
        for blk in y_blocks.iter_mut() {
            prev_dc.y = encode_one_block(bw, blk, div_luma, prev_dc.y, dc_luma, ac_luma)?;
        }
        prev_dc.cb = encode_one_block(
            bw,
            &mut cb_blk,
            div_chroma,
            prev_dc.cb,
            dc_chroma,
            ac_chroma,
        )?;
        prev_dc.cr = encode_one_block(
            bw,
            &mut cr_blk,
            div_chroma,
            prev_dc.cr,
            dc_chroma,
            ac_chroma,
        )?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_scan<S: SamplingScheme, W: Write>(
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    bw: &mut BitWriter<W>,
    div_luma: &Divisors,
    div_chroma: &Divisors,
    dc_luma: &HuffmanTable,
    ac_luma: &HuffmanTable,
    dc_chroma: &HuffmanTable,
    ac_chroma: &HuffmanTable,
    restart_interval: u16,
) -> io::Result<()> {
    let mcus_x = width.div_ceil(S::MCU_W);
    let mcus_y = height.div_ceil(S::MCU_H);
    let mut prev_dc = DcPredictors::default();
    // Restart bookkeeping: every `restart_interval` MCUs we flush the
    // entropy bits to a byte boundary, emit an RSTn (n cycles 0..=7),
    // and reset the DC predictors. Disabled when `restart_interval == 0`.
    let restart_interval = restart_interval as u32;
    let mut mcus_since_rst: u32 = 0;
    let mut next_rst: u8 = 0;
    let mut mcu_count: u64 = 0;
    let total_mcus = mcus_x as u64 * mcus_y as u64;
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            if restart_interval > 0
                && mcus_since_rst == restart_interval
                && mcu_count < total_mcus
            {
                bw.write_restart(next_rst)?;
                next_rst = (next_rst + 1) & 7;
                mcus_since_rst = 0;
                prev_dc = DcPredictors::default();
            }
            S::encode_one_mcu(
                bw,
                pixels,
                width,
                height,
                layout,
                mx,
                my,
                &mut prev_dc,
                div_luma,
                div_chroma,
                dc_luma,
                ac_luma,
                dc_chroma,
                ac_chroma,
            )?;
            mcus_since_rst += 1;
            mcu_count += 1;
        }
    }
    Ok(())
}

/// Run one block end-to-end: DCT → quantize+zigzag → entropy-code.
fn encode_one_block<W: Write>(
    bw: &mut BitWriter<W>,
    block: &mut [i16; 64],
    div: &Divisors,
    prev_dc: i16,
    dc_tab: &HuffmanTable,
    ac_tab: &HuffmanTable,
) -> io::Result<i16> {
    arch::backend::dct::fdct_islow(block);
    let zz = quant::quantize_and_zigzag(block, div);
    encode_block(bw, &zz, prev_dc, dc_tab, ac_tab)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: SOI + EOI bracket a non-empty payload.
    #[test]
    fn produces_soi_eoi_markers() {
        let w = 16;
        let h = 16;
        let mut rgba = Vec::with_capacity(w * h * 4);
        for _ in 0..(w * h) {
            rgba.extend_from_slice(&[200, 100, 50, 255]);
        }
        let mut out = Vec::new();
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.encode_rgba(&rgba, w as u32, h as u32).unwrap();
        assert_eq!(&out[..2], &[0xFF, 0xD8], "missing SOI");
        assert_eq!(&out[out.len() - 2..], &[0xFF, 0xD9], "missing EOI");
        assert!(out.len() > 100, "stream too short to be plausible");
    }

    #[test]
    fn rejects_short_buffer() {
        let mut out = Vec::new();
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        let pixels = vec![0u8; 4 * 4 * 2]; // half what 4x4 RGBA needs
        let err = enc.encode_rgba(&pixels, 4, 4).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn rejects_zero_dimensions() {
        let mut out = Vec::new();
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        let err = enc.encode_rgb(&[], 0, 8).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }
}
