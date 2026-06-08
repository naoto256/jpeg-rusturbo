//! Baseline + progressive JPEG encoder.
//!
//! The [`JpegEncoder`] public type and its `encode_*` entry points live
//! here, alongside the [`SamplingScheme`] trait and its per-subsampling
//! MCU implementations. The crate root re-exports [`JpegEncoder`] so the
//! public path stays `jpeg_rusturbo::JpegEncoder`.

pub(crate) mod huffman;
mod huffman_optimize;
mod markers;
mod progressive;
pub(crate) mod quant;

use std::io::{self, Write};

use rayon::prelude::*;

use crate::color::{self, PixelClass, PixelLayout, RGB, RGBA};
use crate::tables::{
    STD_CHROMA_AC, STD_CHROMA_DC, STD_CHROMA_QUANT, STD_LUMA_AC, STD_LUMA_DC, STD_LUMA_QUANT,
    scale_quant_table,
};
use crate::{ChromaSubsampling, PixelFormat};

use huffman::{BitWriter, HuffmanTable, encode_block};

use crate::arch;
use crate::tables::{Divisors, build_divisors};

/// Run-time `ChromaSubsampling` enum → static `SamplingScheme` type
/// dispatch. The trait is consumed by generic functions
/// (`encode_scan<S>`, `encode_optimized<S>`, `parallel_quantize_rows<S>`
/// etc.) which monomorphize per scheme; this macro routes the runtime
/// enum to the right monomorphization in a single place per call site.
///
/// `$S` is bound as a type alias to the matched scheme inside the body,
/// so callers write `S::H_V` / `encode_optimized::<S, _>(...)` once and
/// the macro expands the match. Adding a new variant requires editing
/// only this macro plus the `enum ChromaSubsampling` declaration —
/// every other call site keeps working without modification.
macro_rules! dispatch_scheme {
    ($subsampling:expr, $S:ident => $body:expr) => {
        match $subsampling {
            ChromaSubsampling::Yuv444 => {
                type $S = Yuv444Scheme;
                $body
            }
            ChromaSubsampling::Yuv422 => {
                type $S = Yuv422Scheme;
                $body
            }
            ChromaSubsampling::Yuv420 => {
                type $S = Yuv420Scheme;
                $body
            }
        }
    };
}

#[allow(clippy::too_many_arguments)]
fn extract_block_ycbcr_rgb(
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    mx: u32,
    my: u32,
    y_blk: &mut [i16; 64],
    cb_blk: &mut [i16; 64],
    cr_blk: &mut [i16; 64],
) {
    let x0 = mx * Yuv444Scheme::MCU_W;
    let y0 = my * Yuv444Scheme::MCU_H;
    if layout == RGB && x0 + Yuv444Scheme::MCU_W <= width && y0 + Yuv444Scheme::MCU_H <= height {
        color::extract_block_ycbcr_rgb_full(pixels, width, x0, y0, y_blk, cb_blk, cr_blk);
    } else {
        color::extract_block_ycbcr(pixels, width, height, layout, x0, y0, y_blk, cb_blk, cr_blk);
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
    optimize_huffman: bool,
    progressive: bool,
    threads: u32,
    exif: Option<Vec<u8>>,
    icc_profile: Option<Vec<u8>>,
}

fn validate_custom_quant_tables(custom_quant: &Option<QuantPair>) -> io::Result<()> {
    let Some((luma, chroma)) = custom_quant else {
        return Ok(());
    };
    if luma.contains(&0) || chroma.contains(&0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "custom quantization table contains zero (must be in 1..=255)",
        ));
    }
    Ok(())
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
            optimize_huffman: false,
            progressive: false,
            threads: 1,
            exif: None,
            icc_profile: None,
        }
    }

    /// Set the parallelism budget for the DCT + quantize stage.
    ///
    /// - `1` (the default): fully serial — uses the same single-thread
    ///   path as before and the output is byte-for-byte identical to a
    ///   build that doesn't know about this knob at all.
    /// - `0`: automatic — uses the ambient rayon thread pool, which by
    ///   default sizes to the number of logical CPUs.
    /// - `n > 1`: build a private rayon pool with `n` worker threads
    ///   for this encode. The pool is constructed once per `encode_*`
    ///   call and dropped on return, so the caller's global pool (if
    ///   any) is left undisturbed.
    ///
    /// Only the per-MCU pixel-fetch / color-convert / forward-DCT /
    /// quantize / zigzag work runs in parallel. The Huffman entropy
    /// emit stays sequential because the DC predictor chain and the
    /// bit-stream itself are MCU-ordered. The encoded JPEG bytes are
    /// therefore identical regardless of how many threads run the
    /// front half.
    pub fn set_threads(&mut self, n: u32) {
        self.threads = n;
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
    /// (Annex K + quality scaling) call [`Self::clear_quant_tables`].
    ///
    /// Each entry must be in `1..=255` (a zero quantizer divides DCT
    /// coefficients by zero and is rejected by every conforming
    /// decoder). The encoder rejects zero entries at encode time with
    /// [`io::ErrorKind::InvalidInput`].
    ///
    /// Intended for advanced workflows: ML-driven RDO, mozjpeg /
    /// libjpeg-turbo table replication, per-image perceptual tuning.
    pub fn set_quant_tables(&mut self, luma: [u8; 64], chroma: [u8; 64]) {
        self.custom_quant = Some((Box::new(luma), Box::new(chroma)));
    }

    /// Clear any custom quantization tables previously installed via
    /// [`Self::set_quant_tables`]; subsequent encodes use the Annex K +
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
    /// this higher than the total MCU count emits no RSTn markers at
    /// all (the interval is never reached), effectively a no-op aside
    /// from the DRI segment overhead.
    pub fn set_restart_interval(&mut self, interval: u16) {
        self.restart_interval = interval;
    }

    /// Enable two-pass optimized Huffman tables (libjpeg-turbo
    /// `-optimize`-style). Pass 1 counts the actual symbol frequencies
    /// on the quantized coefficients of this image; pass 2 builds the
    /// per-image optimal canonical Huffman tables (T.81 K.2 algorithm
    /// with K.3 length limiting) and re-emits the scan using them.
    ///
    /// Typical savings on photographic content at q=80 are 4-10% in
    /// file size at identical reconstructed PSNR; cost is roughly one
    /// extra entropy-pass worth of CPU plus a buffer holding the
    /// quantized coefficients between passes.
    ///
    /// Default is `false` — when off, the encoder's output is
    /// byte-identical to a build without this setter.
    pub fn set_optimize_huffman(&mut self, on: bool) {
        self.optimize_huffman = on;
    }

    /// Attach an EXIF metadata blob, emitted as an APP1 segment
    /// immediately after the JFIF header. `exif` should be the raw
    /// EXIF payload (typically `TIFF header + IFD0 + ...`) — the
    /// encoder prepends the standard `"Exif\0\0"` identifier itself.
    ///
    /// Pass-through use case: read an existing JPEG with a separate
    /// EXIF parser, hand the bytes back here, re-encode without
    /// losing camera / orientation metadata. Image-processing
    /// pipelines that strip EXIF by default can use this to restore
    /// it.
    ///
    /// EXIF is allowed exactly one APP1 segment, so the payload is
    /// capped at ~65 KB. Setting `None` (or never calling this)
    /// produces a JPEG with no APP1 — bit-identical to a build
    /// without this setter.
    pub fn set_exif(&mut self, exif: Option<Vec<u8>>) {
        self.exif = exif;
    }

    /// Attach an ICC color profile, emitted as one or more APP2
    /// segments per the ICC.1 embedding convention. `icc` is the raw
    /// `.icc` / `.icm` profile bytes — the encoder writes the
    /// `"ICC_PROFILE\0"` identifier and the multi-segment chunking
    /// itself.
    ///
    /// Pass-through use case: same as `set_exif` but for color
    /// management. Photo-display pipelines on macOS / Windows /
    /// browsers respect APP2 ICC when present and gracefully fall
    /// back to sRGB when absent, so this is purely additive.
    ///
    /// Multi-segment chunking handles profiles up to ~16.7 MB
    /// (255 segments × ~65 KB); realistic ICC profiles are 1-5 KB and
    /// fit in a single segment. Setting `None` emits no APP2 —
    /// bit-identical to a build without this setter.
    pub fn set_icc_profile(&mut self, icc: Option<Vec<u8>>) {
        self.icc_profile = icc;
    }

    /// Emit a **progressive** JPEG (SOF2) instead of the default
    /// baseline (SOF0). Progressive splits the entropy-coded segment
    /// into multiple scans that each carry a sub-band of every
    /// block's coefficients — a conforming decoder can render a
    /// coarse "DC-only" preview as soon as the first scan finishes,
    /// then refine as later scans arrive. This is the standard JPEG
    /// shape for slow-network photo delivery (browsers, image
    /// galleries).
    ///
    /// This ships the full eight-scan **successive-approximation**
    /// plan: DC of all three components interleaved first, then
    /// per-component AC at the first approximation (`Al=1`), followed
    /// by the DC interleaved refinement and per-component AC
    /// refinement passes (`Al=0`). That exercises all four T.81 Annex
    /// G scan types (DC first, AC first, DC refine, AC refine), so the
    /// output is decodable by every conforming progressive decoder
    /// (including this crate's).
    ///
    /// Composes with [`set_optimize_huffman`](Self::set_optimize_huffman):
    /// when both are set, the encoder runs a count-then-emit pass per
    /// scan, builds per-scan optimal Huffman tables (including the
    /// `EOBn` symbols absent from the Annex K reference tables), and
    /// emits multi-block end-of-band runs. That collapses the file-
    /// size cost progressive normally carries vs baseline-SOF0 on
    /// natural content. Default is `false` (= baseline SOF0, byte-
    /// identical to a build that doesn't know about this setter).
    ///
    /// Does **not** compose with
    /// [`encode_grayscale`](Self::encode_grayscale) — setting both
    /// returns [`io::ErrorKind::Unsupported`] at encode time
    /// (progressive grayscale is not currently implemented).
    pub fn set_progressive(&mut self, on: bool) {
        self.progressive = on;
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
    /// is zero, if either dimension exceeds JPEG's `u16::MAX` SOF
    /// limit, if `width * height * 3` overflows `usize`, or if the
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
    /// point covering all eight color byte layouts (RGB / BGR / RGBA /
    /// BGRA / ARGB / ABGR / RGBX / BGRX) plus single-byte grayscale
    /// ([`PixelFormat::Gray`]). For `Gray` this dispatches to the same
    /// single-component path as [`encode_grayscale`](Self::encode_grayscale).
    ///
    /// # Errors
    /// Same shape as [`encode_rgb`](Self::encode_rgb) /
    /// [`encode_rgba`](Self::encode_rgba), scaled by `format`'s
    /// bytes-per-pixel.
    pub fn encode(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> io::Result<()> {
        self.encode_inner(pixels, width, height, format.into())
    }

    /// Encode a grayscale (single-byte-per-pixel) buffer as a
    /// **1-component (luma-only) JPEG**. The byte in `pixels` is
    /// treated as Y directly — no RGB→YCbCr conversion, no chroma
    /// planes, no chroma DQT / DHT / SOF / SOS overhead. Output is
    /// roughly **a third of the size** of a Y-channel-only re-encode
    /// of the same content through the 4:2:0 RGB path.
    ///
    /// `pixels` is `width * height` bytes in row-major order. Trailing
    /// bytes past `width * height` are ignored.
    ///
    /// Composes with [`set_optimize_huffman`](Self::set_optimize_huffman),
    /// [`set_quality`-equivalent quant settings](Self::set_quant_tables),
    /// [`set_restart_interval`](Self::set_restart_interval),
    /// [`set_exif`](Self::set_exif) / [`set_icc_profile`](Self::set_icc_profile).
    ///
    /// **Does not** compose with [`set_progressive`](Self::set_progressive)
    /// — calling that with `true` and then `encode_grayscale` returns
    /// [`io::ErrorKind::Unsupported`].
    ///
    /// [`set_subsampling`](Self::set_subsampling) is silently ignored
    /// — there is no chroma to subsample on a 1-component image.
    ///
    /// [`set_threads`](Self::set_threads) is silently treated as 1 —
    /// the grayscale path is currently serial-only. Bytes are
    /// identical regardless of the configured thread count.
    ///
    /// # Errors
    ///
    /// Same shape as [`encode_rgb`](Self::encode_rgb), but with the
    /// size requirement `width * height`. Returns
    /// [`io::ErrorKind::Unsupported`] if `set_progressive(true)` was
    /// previously called.
    ///
    /// # Example
    ///
    /// ```
    /// use jpeg_rusturbo::JpegEncoder;
    ///
    /// let pixels = vec![200u8; 16 * 16]; // 16x16 light-gray
    /// let mut out: Vec<u8> = Vec::new();
    /// let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
    /// enc.encode_grayscale(&pixels, 16, 16)?;
    /// assert_eq!(&out[..2], &[0xFF, 0xD8]); // SOI marker
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn encode_grayscale(&mut self, pixels: &[u8], width: u32, height: u32) -> io::Result<()> {
        self.encode_inner(pixels, width, height, color::GRAY)
    }

    /// Encode a CMYK pixel buffer (4 bytes/pixel: C, M, Y, K) as a
    /// **4-component baseline JPEG**, pass-through.
    ///
    /// Each of the four channels becomes an independent JPEG
    /// component (sampling factor 1:1:1:1, all four sharing the luma
    /// quantization table and one luma DC + one luma AC Huffman
    /// table). No CMYK↔RGB conversion of any kind is performed; the
    /// bytes go through the standard DCT / quantize / entropy chain
    /// one channel at a time. The output carries no APP14 Adobe
    /// marker — it is a plain (non-YCCK) CMYK stream that any
    /// conforming JPEG decoder reads back as four raw components.
    ///
    /// `pixels` is `width * height * 4` bytes in row-major order
    /// (`C, M, Y, K, C, M, Y, K, …`). Trailing bytes past
    /// `width * height * 4` are ignored.
    ///
    /// Composes with [`set_optimize_huffman`](Self::set_optimize_huffman)
    /// — the two-pass machinery counts all four components' symbols
    /// into one luma-DC + one luma-AC frequency table and emits one
    /// optimal DC table + one optimal AC table shared across the
    /// scan. Also composes with [`set_restart_interval`](Self::set_restart_interval),
    /// [`set_exif`](Self::set_exif) / [`set_icc_profile`](Self::set_icc_profile)
    /// (ICC is especially useful for print pipelines).
    ///
    /// [`set_subsampling`](Self::set_subsampling) is silently ignored
    /// — CMYK encode is fixed at 1:1:1:1, with no chroma analog.
    ///
    /// [`set_threads`](Self::set_threads) is silently treated as 1 —
    /// the CMYK path is currently serial-only.
    ///
    /// [`set_quant_tables`](Self::set_quant_tables): only the luma
    /// table is consulted on the CMYK path (a single DQT segment is
    /// emitted, shared across all four components). The chroma table
    /// is silently ignored.
    ///
    /// **Does not** compose with [`set_progressive`](Self::set_progressive)
    /// — combining the two returns [`io::ErrorKind::Unsupported`].
    ///
    /// # Errors
    ///
    /// Same shape as [`encode_rgb`](Self::encode_rgb), but with the
    /// size requirement `width * height * 4`. Returns
    /// [`io::ErrorKind::Unsupported`] if `set_progressive(true)` was
    /// previously called.
    ///
    /// # Example
    ///
    /// ```
    /// use jpeg_rusturbo::JpegEncoder;
    ///
    /// // 16x16 of pure cyan ink (C=255, M=Y=K=0).
    /// let mut cmyk = Vec::with_capacity(16 * 16 * 4);
    /// for _ in 0..(16 * 16) {
    ///     cmyk.extend_from_slice(&[255, 0, 0, 0]);
    /// }
    /// let mut out: Vec<u8> = Vec::new();
    /// let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
    /// enc.encode_cmyk(&cmyk, 16, 16)?;
    /// assert_eq!(&out[..2], &[0xFF, 0xD8]); // SOI marker
    /// # Ok::<(), std::io::Error>(())
    /// ```
    pub fn encode_cmyk(&mut self, pixels: &[u8], width: u32, height: u32) -> io::Result<()> {
        self.encode_inner(pixels, width, height, color::CMYK)
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
        if width > u16::MAX as u32 || height > u16::MAX as u32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "image dimensions exceed JPEG SOF u16 limit: {}x{} (max {}x{})",
                    width,
                    height,
                    u16::MAX,
                    u16::MAX
                ),
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
        validate_custom_quant_tables(&self.custom_quant)?;

        // Dispatch on the layout category. Non-RGB categories use
        // dedicated single-or-non-RGB-component pipelines that skip the
        // chroma DQT / DHT / SOF / SOS overhead and the RGB→YCbCr
        // conversion entirely. The RGB arm falls through to the
        // 3-component baseline pipeline below.
        match layout.class() {
            PixelClass::Gray => return self.encode_grayscale_inner(pixels, width, height),
            PixelClass::Cmyk => return self.encode_cmyk_inner(pixels, width, height),
            PixelClass::Rgb => {}
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
        let div_luma = build_divisors(&luma_q);
        let div_chroma = build_divisors(&chroma_q);

        // Progressive (SOF2) takes precedence — it ships its own
        // SOI..EOI shape with multi-scan entropy. The progressive
        // path internally honors `optimize_huffman` to switch to its
        // per-scan EOBn-aware two-pass plan.
        if self.progressive {
            return progressive::encode_progressive_inner(
                self,
                pixels,
                width,
                height,
                layout,
                &div_luma,
                &div_chroma,
            );
        }

        // Optimized Huffman: run a separate two-pass entry point.
        // Returns once it has produced a complete stream (SOI..EOI).
        if self.optimize_huffman {
            return self.encode_inner_optimize(
                pixels,
                width,
                height,
                layout,
                &div_luma,
                &div_chroma,
            );
        }

        // Expand the standard Huffman tables into encoder lookups.
        let dc_luma = HuffmanTable::from_std(&STD_LUMA_DC);
        let ac_luma = HuffmanTable::from_std(&STD_LUMA_AC);
        let dc_chroma = HuffmanTable::from_std(&STD_CHROMA_DC);
        let ac_chroma = HuffmanTable::from_std(&STD_CHROMA_AC);

        // ---- Header ----
        markers::write_soi(&mut self.out)?;
        markers::write_app0_jfif(&mut self.out)?;
        // EXIF / ICC pass-through metadata. APP1 / APP2 segments are
        // written immediately after the JFIF APP0 — that's the
        // conventional libjpeg / exiftool placement and what every
        // decoder we tested expects. They're absent (= bit-identical
        // to a no-metadata build) when the user never called
        // `set_exif` / `set_icc_profile`.
        if let Some(exif) = self.exif.as_deref() {
            markers::write_app1_exif(&mut self.out, exif)?;
        }
        if let Some(icc) = self.icc_profile.as_deref() {
            markers::write_app2_icc(&mut self.out, icc)?;
        }
        markers::write_dqt(&mut self.out, 0, &luma_q)?;
        markers::write_dqt(&mut self.out, 1, &chroma_q)?;

        let (h_y, v_y) = dispatch_scheme!(self.subsampling, S => S::H_V);
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
        let threads = self.threads;
        dispatch_scan(
            threads,
            self.subsampling,
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
        )?;
        bw.flush_to_byte_boundary()?;

        // ---- Trailer ----
        markers::write_eoi(&mut self.out)?;
        Ok(())
    }

    /// Two-pass optimized-Huffman entry point. Pass 1: DCT + quantize
    /// every block into per-component buffers and count the symbol
    /// frequencies the standard encoder would have emitted. Pass 2:
    /// build optimal canonical Huffman tables from those frequencies,
    /// write the DHT segments, and entropy-code the buffered blocks.
    fn encode_inner_optimize(
        &mut self,
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        div_luma: &Divisors,
        div_chroma: &Divisors,
    ) -> io::Result<()> {
        // Re-derive quant tables for header emission (cheap; matches
        // exactly what the caller computed).
        let (luma_q, chroma_q) = match &self.custom_quant {
            Some((l, c)) => (**l, **c),
            None => (
                scale_quant_table(&STD_LUMA_QUANT, self.quality),
                scale_quant_table(&STD_CHROMA_QUANT, self.quality),
            ),
        };
        dispatch_scheme!(self.subsampling, S => encode_optimized::<S, _>(
            self, pixels, width, height, layout, &luma_q, &chroma_q, div_luma, div_chroma,
        ))
    }

    /// Single-component (grayscale, T.81 1-component frame) encode
    /// path. Branches off `encode_inner` before any chroma-aware
    /// dispatch — emits SOI/APP0[/APP1][/APP2]/DQT(luma)/SOF0(1 comp)/
    /// DHT(luma DC + AC)/[DRI]/SOS(1 comp)/entropy/EOI.
    ///
    /// `set_subsampling` and `set_threads` are ignored (no-ops for a
    /// 1-component image); `set_progressive(true)` returns
    /// `Unsupported`. Other knobs (custom quant, optimize-Huffman,
    /// restart interval, EXIF, ICC) compose normally.
    fn encode_grayscale_inner(&mut self, pixels: &[u8], width: u32, height: u32) -> io::Result<()> {
        if self.progressive {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "progressive grayscale encode is not implemented; \
                 use baseline by calling set_progressive(false), \
                 or open an issue",
            ));
        }

        let luma_q = match &self.custom_quant {
            Some((l, _)) => **l,
            None => scale_quant_table(&STD_LUMA_QUANT, self.quality),
        };
        let div_luma = build_divisors(&luma_q);

        let mcus_x = width.div_ceil(8);
        let mcus_y = height.div_ceil(8);
        let total_mcus = (mcus_x as usize) * (mcus_y as usize);

        if self.optimize_huffman {
            // Pass 1: DCT + quantize every 8×8 block into a buffer and
            // count the symbol frequencies the standard tables would
            // have produced (restart-interval-aware DC reset).
            let mut blocks: Vec<[i16; 64]> = Vec::with_capacity(total_mcus);
            for my in 0..mcus_y {
                for mx in 0..mcus_x {
                    let mut blk = [0i16; 64];
                    color::extract_block_gray(pixels, width, height, mx * 8, my * 8, &mut blk);
                    arch::backend::dct::fdct_islow(&mut blk);
                    blocks.push(quant::quantize_and_zigzag(&blk, &div_luma));
                }
            }

            let mut dc_freq = [0u32; 257];
            let mut ac_freq = [0u32; 257];
            {
                let restart = self.restart_interval as u32;
                let mut prev_dc: i16 = 0;
                let mut mcus_since_rst: u32 = 0;
                for blk in blocks.iter() {
                    if restart > 0 && mcus_since_rst == restart {
                        prev_dc = 0;
                        mcus_since_rst = 0;
                    }
                    prev_dc =
                        huffman_optimize::count_block(blk, prev_dc, &mut dc_freq, &mut ac_freq);
                    mcus_since_rst += 1;
                }
            }

            let opt_dc = huffman_optimize::build_optimal_huffman(
                &dc_freq,
                &STD_LUMA_DC.bits,
                STD_LUMA_DC.values,
            );
            let opt_ac = huffman_optimize::build_optimal_huffman(
                &ac_freq,
                &STD_LUMA_AC.bits,
                STD_LUMA_AC.values,
            );
            let dc_tab = HuffmanTable::from_bits_values(&opt_dc.bits, &opt_dc.values);
            let ac_tab = HuffmanTable::from_bits_values(&opt_ac.bits, &opt_ac.values);

            // Header.
            markers::write_soi(&mut self.out)?;
            markers::write_app0_jfif(&mut self.out)?;
            if let Some(exif) = self.exif.as_deref() {
                markers::write_app1_exif(&mut self.out, exif)?;
            }
            if let Some(icc) = self.icc_profile.as_deref() {
                markers::write_app2_icc(&mut self.out, icc)?;
            }
            markers::write_dqt(&mut self.out, 0, &luma_q)?;
            markers::write_sof0(&mut self.out, width as u16, height as u16, &[(1, 1, 1, 0)])?;
            markers::write_dht_bits_values(&mut self.out, 0, 0, &opt_dc.bits, &opt_dc.values)?;
            markers::write_dht_bits_values(&mut self.out, 1, 0, &opt_ac.bits, &opt_ac.values)?;
            if self.restart_interval > 0 {
                markers::write_dri(&mut self.out, self.restart_interval)?;
            }
            markers::write_sos(&mut self.out, &[(1, 0, 0)])?;

            // Pass 2: entropy-code the buffered blocks.
            let mut bw = BitWriter::new(&mut self.out);
            bw.reserve(total_mcus * 32);
            let restart = self.restart_interval as u32;
            let mut prev_dc: i16 = 0;
            let mut mcus_since_rst: u32 = 0;
            let mut next_rst: u8 = 0;
            for (idx, blk) in blocks.iter().enumerate() {
                if restart > 0 && mcus_since_rst == restart && idx < total_mcus {
                    bw.write_restart(next_rst)?;
                    next_rst = (next_rst + 1) & 7;
                    mcus_since_rst = 0;
                    prev_dc = 0;
                }
                prev_dc = encode_block(&mut bw, blk, prev_dc, &dc_tab, &ac_tab)?;
                mcus_since_rst += 1;
            }
            bw.flush_to_byte_boundary()?;
            markers::write_eoi(&mut self.out)?;
            return Ok(());
        }

        // ---- Standard-tables path ----
        let dc_tab = HuffmanTable::from_std(&STD_LUMA_DC);
        let ac_tab = HuffmanTable::from_std(&STD_LUMA_AC);

        markers::write_soi(&mut self.out)?;
        markers::write_app0_jfif(&mut self.out)?;
        if let Some(exif) = self.exif.as_deref() {
            markers::write_app1_exif(&mut self.out, exif)?;
        }
        if let Some(icc) = self.icc_profile.as_deref() {
            markers::write_app2_icc(&mut self.out, icc)?;
        }
        markers::write_dqt(&mut self.out, 0, &luma_q)?;
        markers::write_sof0(&mut self.out, width as u16, height as u16, &[(1, 1, 1, 0)])?;
        markers::write_dht(&mut self.out, 0, 0, &STD_LUMA_DC)?;
        markers::write_dht(&mut self.out, 1, 0, &STD_LUMA_AC)?;
        if self.restart_interval > 0 {
            markers::write_dri(&mut self.out, self.restart_interval)?;
        }
        markers::write_sos(&mut self.out, &[(1, 0, 0)])?;

        let mut bw = BitWriter::new(&mut self.out);
        bw.reserve((width as usize) * (height as usize));
        let restart = self.restart_interval as u32;
        let mut prev_dc: i16 = 0;
        let mut mcus_since_rst: u32 = 0;
        let mut next_rst: u8 = 0;
        let mut mcu_idx: usize = 0;
        for my in 0..mcus_y {
            for mx in 0..mcus_x {
                if restart > 0 && mcus_since_rst == restart && mcu_idx < total_mcus {
                    bw.write_restart(next_rst)?;
                    next_rst = (next_rst + 1) & 7;
                    mcus_since_rst = 0;
                    prev_dc = 0;
                }
                let mut blk = [0i16; 64];
                color::extract_block_gray(pixels, width, height, mx * 8, my * 8, &mut blk);
                prev_dc =
                    encode_one_block(&mut bw, &mut blk, &div_luma, prev_dc, &dc_tab, &ac_tab)?;
                mcus_since_rst += 1;
                mcu_idx += 1;
            }
        }
        bw.flush_to_byte_boundary()?;
        markers::write_eoi(&mut self.out)?;
        Ok(())
    }

    /// Four-component (CMYK) baseline encode path. Treats each of the
    /// four input channels as an independent JPEG component, sampling
    /// factor 1:1:1:1, all four sharing the luma quant table and one
    /// luma-DC + one luma-AC Huffman table. No APP14 marker is
    /// emitted — output is plain (non-YCCK) CMYK. Mirrors the shape
    /// of `encode_grayscale_inner` (single DQT, single DHT pair,
    /// optimize-Huffman composes via shared frequency tables).
    fn encode_cmyk_inner(&mut self, pixels: &[u8], width: u32, height: u32) -> io::Result<()> {
        if self.progressive {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "progressive CMYK encode is not implemented; \
                 use baseline by calling set_progressive(false), \
                 or open an issue",
            ));
        }

        // Only the luma quant table participates on the CMYK path.
        // A user-supplied chroma table (via `set_quant_tables`) is
        // silently ignored — there is no chroma to quantize.
        let luma_q = match &self.custom_quant {
            Some((l, _)) => **l,
            None => scale_quant_table(&STD_LUMA_QUANT, self.quality),
        };
        let div_luma = build_divisors(&luma_q);

        let mcus_x = width.div_ceil(8);
        let mcus_y = height.div_ceil(8);
        let total_mcus = (mcus_x as usize) * (mcus_y as usize);

        if self.optimize_huffman {
            // Pass 1: DCT + quantize all 4 channels of every MCU into
            // a flat per-MCU buffer of 4 zigzag blocks (C, M, Y, K
            // order). Per-component DC predictors are tracked across
            // restart boundaries the same way the chroma path does
            // for Y/Cb/Cr.
            let mut blocks: Vec<[i16; 64]> = Vec::with_capacity(total_mcus * 4);
            for my in 0..mcus_y {
                for mx in 0..mcus_x {
                    for ch in 0..4 {
                        let mut blk = [0i16; 64];
                        color::extract_block_cmyk(
                            pixels,
                            width,
                            height,
                            mx * 8,
                            my * 8,
                            ch,
                            &mut blk,
                        );
                        arch::backend::dct::fdct_islow(&mut blk);
                        blocks.push(quant::quantize_and_zigzag(&blk, &div_luma));
                    }
                }
            }

            // Count all 4 components into one luma DC + one luma AC
            // frequency histogram (shared tables on emit).
            let mut dc_freq = [0u32; 257];
            let mut ac_freq = [0u32; 257];
            {
                let restart = self.restart_interval as u32;
                let mut prev_dc = [0i16; 4];
                let mut mcus_since_rst: u32 = 0;
                for mcu in blocks.chunks_exact(4) {
                    if restart > 0 && mcus_since_rst == restart {
                        prev_dc = [0; 4];
                        mcus_since_rst = 0;
                    }
                    for (ch, blk) in mcu.iter().enumerate() {
                        prev_dc[ch] = huffman_optimize::count_block(
                            blk,
                            prev_dc[ch],
                            &mut dc_freq,
                            &mut ac_freq,
                        );
                    }
                    mcus_since_rst += 1;
                }
            }

            let opt_dc = huffman_optimize::build_optimal_huffman(
                &dc_freq,
                &STD_LUMA_DC.bits,
                STD_LUMA_DC.values,
            );
            let opt_ac = huffman_optimize::build_optimal_huffman(
                &ac_freq,
                &STD_LUMA_AC.bits,
                STD_LUMA_AC.values,
            );
            let dc_tab = HuffmanTable::from_bits_values(&opt_dc.bits, &opt_dc.values);
            let ac_tab = HuffmanTable::from_bits_values(&opt_ac.bits, &opt_ac.values);

            // Header.
            markers::write_soi(&mut self.out)?;
            markers::write_app0_jfif(&mut self.out)?;
            if let Some(exif) = self.exif.as_deref() {
                markers::write_app1_exif(&mut self.out, exif)?;
            }
            if let Some(icc) = self.icc_profile.as_deref() {
                markers::write_app2_icc(&mut self.out, icc)?;
            }
            markers::write_dqt(&mut self.out, 0, &luma_q)?;
            markers::write_sof0(
                &mut self.out,
                width as u16,
                height as u16,
                &[
                    (1, 1, 1, 0), // C
                    (2, 1, 1, 0), // M
                    (3, 1, 1, 0), // Y
                    (4, 1, 1, 0), // K
                ],
            )?;
            markers::write_dht_bits_values(&mut self.out, 0, 0, &opt_dc.bits, &opt_dc.values)?;
            markers::write_dht_bits_values(&mut self.out, 1, 0, &opt_ac.bits, &opt_ac.values)?;
            if self.restart_interval > 0 {
                markers::write_dri(&mut self.out, self.restart_interval)?;
            }
            markers::write_sos(
                &mut self.out,
                &[
                    (1, 0, 0), // C → DC0/AC0
                    (2, 0, 0), // M → DC0/AC0
                    (3, 0, 0), // Y → DC0/AC0
                    (4, 0, 0), // K → DC0/AC0
                ],
            )?;

            // Pass 2: entropy-code the buffered blocks.
            let mut bw = BitWriter::new(&mut self.out);
            bw.reserve(total_mcus * 64);
            let restart = self.restart_interval as u32;
            let mut prev_dc = [0i16; 4];
            let mut mcus_since_rst: u32 = 0;
            let mut next_rst: u8 = 0;
            for (mcu_idx, mcu) in blocks.chunks_exact(4).enumerate() {
                if restart > 0 && mcus_since_rst == restart && mcu_idx < total_mcus {
                    bw.write_restart(next_rst)?;
                    next_rst = (next_rst + 1) & 7;
                    mcus_since_rst = 0;
                    prev_dc = [0; 4];
                }
                for (ch, blk) in mcu.iter().enumerate() {
                    prev_dc[ch] = encode_block(&mut bw, blk, prev_dc[ch], &dc_tab, &ac_tab)?;
                }
                mcus_since_rst += 1;
            }
            bw.flush_to_byte_boundary()?;
            markers::write_eoi(&mut self.out)?;
            return Ok(());
        }

        // ---- Standard-tables path ----
        let dc_tab = HuffmanTable::from_std(&STD_LUMA_DC);
        let ac_tab = HuffmanTable::from_std(&STD_LUMA_AC);

        markers::write_soi(&mut self.out)?;
        markers::write_app0_jfif(&mut self.out)?;
        if let Some(exif) = self.exif.as_deref() {
            markers::write_app1_exif(&mut self.out, exif)?;
        }
        if let Some(icc) = self.icc_profile.as_deref() {
            markers::write_app2_icc(&mut self.out, icc)?;
        }
        markers::write_dqt(&mut self.out, 0, &luma_q)?;
        markers::write_sof0(
            &mut self.out,
            width as u16,
            height as u16,
            &[(1, 1, 1, 0), (2, 1, 1, 0), (3, 1, 1, 0), (4, 1, 1, 0)],
        )?;
        markers::write_dht(&mut self.out, 0, 0, &STD_LUMA_DC)?;
        markers::write_dht(&mut self.out, 1, 0, &STD_LUMA_AC)?;
        if self.restart_interval > 0 {
            markers::write_dri(&mut self.out, self.restart_interval)?;
        }
        markers::write_sos(&mut self.out, &[(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)])?;

        let mut bw = BitWriter::new(&mut self.out);
        bw.reserve((width as usize) * (height as usize) * 4);
        let restart = self.restart_interval as u32;
        let mut prev_dc = [0i16; 4];
        let mut mcus_since_rst: u32 = 0;
        let mut next_rst: u8 = 0;
        let mut mcu_idx: usize = 0;
        for my in 0..mcus_y {
            for mx in 0..mcus_x {
                if restart > 0 && mcus_since_rst == restart && mcu_idx < total_mcus {
                    bw.write_restart(next_rst)?;
                    next_rst = (next_rst + 1) & 7;
                    mcus_since_rst = 0;
                    prev_dc = [0; 4];
                }
                for (ch, dc) in prev_dc.iter_mut().enumerate() {
                    let mut blk = [0i16; 64];
                    color::extract_block_cmyk(pixels, width, height, mx * 8, my * 8, ch, &mut blk);
                    *dc = encode_one_block(&mut bw, &mut blk, &div_luma, *dc, &dc_tab, &ac_tab)?;
                }
                mcus_since_rst += 1;
                mcu_idx += 1;
            }
        }
        bw.flush_to_byte_boundary()?;
        markers::write_eoi(&mut self.out)?;
        Ok(())
    }
}

/// Shared body of the optimized-Huffman path, parameterized over the
/// sampling scheme. Kept as a free function so the per-scheme const
/// `Y_BLOCKS_PER_MCU` is monomorphized into the hot loops.
#[allow(clippy::too_many_arguments)]
fn encode_optimized<S: SamplingScheme, W: Write>(
    enc: &mut JpegEncoder<W>,
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    luma_q: &[u8; 64],
    chroma_q: &[u8; 64],
    div_luma: &Divisors,
    div_chroma: &Divisors,
) -> io::Result<()> {
    let mcus_x = width.div_ceil(S::MCU_W);
    let mcus_y = height.div_ceil(S::MCU_H);
    let total_mcus = (mcus_x as usize) * (mcus_y as usize);

    // ---- Pass 1: DCT + quantize all blocks; collect into per-component
    // buffers. Frequencies are counted in a second walk that mirrors
    // pass 2's emit order (matters when restart_interval is non-zero —
    // DC predictors reset and the count must reflect that).
    let mut y_blocks: Vec<[i16; 64]> = Vec::with_capacity(total_mcus * S::Y_BLOCKS_PER_MCU);
    let mut cb_blocks: Vec<[i16; 64]> = Vec::with_capacity(total_mcus);
    let mut cr_blocks: Vec<[i16; 64]> = Vec::with_capacity(total_mcus);
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            S::quantize_one_mcu_per_comp(
                pixels,
                width,
                height,
                layout,
                mx,
                my,
                div_luma,
                div_chroma,
                &mut y_blocks,
                &mut cb_blocks,
                &mut cr_blocks,
            );
        }
    }

    // ---- Frequency counting (scan order, with restart-interval-aware
    // DC predictor reset so pass 2 sees the same DC diffs).
    let mut dc_luma_freq = [0u32; 257];
    let mut ac_luma_freq = [0u32; 257];
    let mut dc_chroma_freq = [0u32; 257];
    let mut ac_chroma_freq = [0u32; 257];
    {
        let restart = enc.restart_interval as u32;
        let mut prev_dc = DcPredictors::default();
        let mut mcus_since_rst: u32 = 0;
        let y_mcus = y_blocks.chunks_exact(S::Y_BLOCKS_PER_MCU);
        for (y_chunk, (cb, cr)) in y_mcus.zip(cb_blocks.iter().zip(cr_blocks.iter())) {
            if restart > 0 && mcus_since_rst == restart {
                prev_dc = DcPredictors::default();
                mcus_since_rst = 0;
            }
            for y in y_chunk {
                prev_dc.y = huffman_optimize::count_block(
                    y,
                    prev_dc.y,
                    &mut dc_luma_freq,
                    &mut ac_luma_freq,
                );
            }
            prev_dc.cb = huffman_optimize::count_block(
                cb,
                prev_dc.cb,
                &mut dc_chroma_freq,
                &mut ac_chroma_freq,
            );
            prev_dc.cr = huffman_optimize::count_block(
                cr,
                prev_dc.cr,
                &mut dc_chroma_freq,
                &mut ac_chroma_freq,
            );
            mcus_since_rst += 1;
        }
    }

    // ---- Build optimal tables.
    let opt_dc_luma = huffman_optimize::build_optimal_huffman(
        &dc_luma_freq,
        &STD_LUMA_DC.bits,
        STD_LUMA_DC.values,
    );
    let opt_ac_luma = huffman_optimize::build_optimal_huffman(
        &ac_luma_freq,
        &STD_LUMA_AC.bits,
        STD_LUMA_AC.values,
    );
    let opt_dc_chroma = huffman_optimize::build_optimal_huffman(
        &dc_chroma_freq,
        &STD_CHROMA_DC.bits,
        STD_CHROMA_DC.values,
    );
    let opt_ac_chroma = huffman_optimize::build_optimal_huffman(
        &ac_chroma_freq,
        &STD_CHROMA_AC.bits,
        STD_CHROMA_AC.values,
    );

    let dc_luma_tab = HuffmanTable::from_bits_values(&opt_dc_luma.bits, &opt_dc_luma.values);
    let ac_luma_tab = HuffmanTable::from_bits_values(&opt_ac_luma.bits, &opt_ac_luma.values);
    let dc_chroma_tab = HuffmanTable::from_bits_values(&opt_dc_chroma.bits, &opt_dc_chroma.values);
    let ac_chroma_tab = HuffmanTable::from_bits_values(&opt_ac_chroma.bits, &opt_ac_chroma.values);

    // ---- Header emission.
    markers::write_soi(&mut enc.out)?;
    markers::write_app0_jfif(&mut enc.out)?;
    // EXIF / ICC pass-through — mirror of the baseline path above.
    if let Some(exif) = enc.exif.as_deref() {
        markers::write_app1_exif(&mut enc.out, exif)?;
    }
    if let Some(icc) = enc.icc_profile.as_deref() {
        markers::write_app2_icc(&mut enc.out, icc)?;
    }
    markers::write_dqt(&mut enc.out, 0, luma_q)?;
    markers::write_dqt(&mut enc.out, 1, chroma_q)?;
    let (h_y, v_y) = S::H_V;
    markers::write_sof0(
        &mut enc.out,
        width as u16,
        height as u16,
        &[(1, h_y, v_y, 0), (2, 1, 1, 1), (3, 1, 1, 1)],
    )?;
    markers::write_dht_bits_values(&mut enc.out, 0, 0, &opt_dc_luma.bits, &opt_dc_luma.values)?;
    markers::write_dht_bits_values(&mut enc.out, 1, 0, &opt_ac_luma.bits, &opt_ac_luma.values)?;
    markers::write_dht_bits_values(
        &mut enc.out,
        0,
        1,
        &opt_dc_chroma.bits,
        &opt_dc_chroma.values,
    )?;
    markers::write_dht_bits_values(
        &mut enc.out,
        1,
        1,
        &opt_ac_chroma.bits,
        &opt_ac_chroma.values,
    )?;
    if enc.restart_interval > 0 {
        markers::write_dri(&mut enc.out, enc.restart_interval)?;
    }
    markers::write_sos(&mut enc.out, &[(1, 0, 0), (2, 1, 1), (3, 1, 1)])?;

    // ---- Pass 2: entropy-code from the buffered blocks using the
    // optimal tables. Mirrors the pass-1 walk exactly so DC predictors
    // and restart markers line up.
    let mut bw = BitWriter::new(&mut enc.out);
    bw.reserve(y_blocks.len() * 32);
    let restart = enc.restart_interval as u32;
    let mut prev_dc = DcPredictors::default();
    let mut mcus_since_rst: u32 = 0;
    let mut next_rst: u8 = 0;
    let y_mcus = y_blocks.chunks_exact(S::Y_BLOCKS_PER_MCU);
    for (mcu_idx, (y_chunk, (cb, cr))) in y_mcus
        .zip(cb_blocks.iter().zip(cr_blocks.iter()))
        .enumerate()
    {
        if restart > 0 && mcus_since_rst == restart && mcu_idx < total_mcus {
            bw.write_restart(next_rst)?;
            next_rst = (next_rst + 1) & 7;
            mcus_since_rst = 0;
            prev_dc = DcPredictors::default();
        }
        for y in y_chunk {
            prev_dc.y = encode_block(&mut bw, y, prev_dc.y, &dc_luma_tab, &ac_luma_tab)?;
        }
        prev_dc.cb = encode_block(&mut bw, cb, prev_dc.cb, &dc_chroma_tab, &ac_chroma_tab)?;
        prev_dc.cr = encode_block(&mut bw, cr, prev_dc.cr, &dc_chroma_tab, &ac_chroma_tab)?;
        mcus_since_rst += 1;
    }
    bw.flush_to_byte_boundary()?;

    markers::write_eoi(&mut enc.out)?;
    Ok(())
}

/// Per-MCU running DC predictor for the three components. Difference-coded
/// across MCUs within a scan (F.1.2.1).
#[derive(Default)]
pub(crate) struct DcPredictors {
    y: i16,
    cb: i16,
    cr: i16,
}

/// One chroma-subsampling scheme (4:4:4, 4:2:2, 4:2:0).
///
/// Each impl owns its MCU geometry and the per-MCU encode work — the
/// generic `encode_scan` below just iterates MCUs and forwards. Adding a
/// new scheme is "impl this trait + add a variant + register here",
/// no scan-level surgery required (Rule of Three: prep work at 2 instances).
pub(crate) trait SamplingScheme {
    /// `(h_factor, v_factor)` for the Y component in SOF0. Cb/Cr are
    /// always (1, 1) in the schemes we support.
    const H_V: (u8, u8);
    /// One MCU's pixel footprint `(width, height)`.
    const MCU_W: u32;
    const MCU_H: u32;
    /// Number of 8×8 blocks emitted per MCU: Y blocks followed by one
    /// Cb and one Cr. 444 = 3, 422 = 4, 420 = 6. Used by the threaded
    /// scan to size its per-MCU block buffer.
    const BLOCKS_PER_MCU: usize;
    /// Of the [`BLOCKS_PER_MCU`] blocks, how many are luma. The rest
    /// are one Cb followed by one Cr.
    ///
    /// Also used by the optimized-Huffman two-pass path to size per-
    /// component coefficient buffers.
    ///
    /// [`BLOCKS_PER_MCU`]: SamplingScheme::BLOCKS_PER_MCU
    const Y_BLOCKS_PER_MCU: usize;

    /// Optimized-Huffman pass-1 companion to [`encode_one_mcu`]: extract +
    /// DCT + quantize one MCU, appending the resulting zig-zagged blocks
    /// into per-component growing buffers instead of entropy-coding them.
    /// Pass 2 of that pipeline then walks the buffers and calls
    /// `encode_block` directly with the optimal Huffman tables.
    ///
    /// [`encode_one_mcu`]: SamplingScheme::encode_one_mcu
    #[allow(clippy::too_many_arguments)]
    fn quantize_one_mcu_per_comp(
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        mx: u32,
        my: u32,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        y_blocks: &mut Vec<[i16; 64]>,
        cb_blocks: &mut Vec<[i16; 64]>,
        cr_blocks: &mut Vec<[i16; 64]>,
    );

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

    /// Threaded-path front half: pixel fetch + color convert + forward
    /// DCT + quantize + zigzag for one MCU. Output is written to
    /// `out`, which must have length [`BLOCKS_PER_MCU`] — first the
    /// luma blocks, then Cb, then Cr, each already in zig-zag order.
    ///
    /// [`BLOCKS_PER_MCU`]: SamplingScheme::BLOCKS_PER_MCU
    #[allow(clippy::too_many_arguments)]
    fn quantize_one_mcu(
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        mx: u32,
        my: u32,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        out: &mut [[i16; 64]],
    );
}

pub(crate) struct Yuv444Scheme;
impl SamplingScheme for Yuv444Scheme {
    const H_V: (u8, u8) = (1, 1);
    const MCU_W: u32 = 8;
    const MCU_H: u32 = 8;
    const BLOCKS_PER_MCU: usize = 3;
    const Y_BLOCKS_PER_MCU: usize = 1;

    fn quantize_one_mcu_per_comp(
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        mx: u32,
        my: u32,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        y_blocks: &mut Vec<[i16; 64]>,
        cb_blocks: &mut Vec<[i16; 64]>,
        cr_blocks: &mut Vec<[i16; 64]>,
    ) {
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        {
            let mut blocks = [[0i16; 64]; 3];
            quantize_mcu_444_rgb(
                pixels,
                width,
                height,
                layout,
                mx,
                my,
                div_luma,
                div_chroma,
                &mut blocks,
            );
            y_blocks.push(blocks[0]);
            cb_blocks.push(blocks[1]);
            cr_blocks.push(blocks[2]);
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut y = [0i16; 64];
            let mut cb = [0i16; 64];
            let mut cr = [0i16; 64];
            extract_block_ycbcr_rgb(
                pixels, width, height, layout, mx, my, &mut y, &mut cb, &mut cr,
            );
            arch::backend::dct::fdct_islow(&mut y);
            y_blocks.push(quant::quantize_and_zigzag(&y, div_luma));
            arch::backend::dct::fdct_islow(&mut cb);
            cb_blocks.push(quant::quantize_and_zigzag(&cb, div_chroma));
            arch::backend::dct::fdct_islow(&mut cr);
            cr_blocks.push(quant::quantize_and_zigzag(&cr, div_chroma));
        }
    }

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
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        {
            let mut blocks = [[0i16; 64]; 3];
            quantize_mcu_444_rgb(
                pixels,
                width,
                height,
                layout,
                mx,
                my,
                div_luma,
                div_chroma,
                &mut blocks,
            );
            prev_dc.y = encode_block(bw, &blocks[0], prev_dc.y, dc_luma, ac_luma)?;
            prev_dc.cb = encode_block(bw, &blocks[1], prev_dc.cb, dc_chroma, ac_chroma)?;
            prev_dc.cr = encode_block(bw, &blocks[2], prev_dc.cr, dc_chroma, ac_chroma)?;
            Ok(())
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut y_blk = [0i16; 64];
            let mut cb_blk = [0i16; 64];
            let mut cr_blk = [0i16; 64];
            extract_block_ycbcr_rgb(
                pixels,
                width,
                height,
                layout,
                mx,
                my,
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

    fn quantize_one_mcu(
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        mx: u32,
        my: u32,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        out: &mut [[i16; 64]],
    ) {
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        {
            quantize_mcu_444_rgb(
                pixels, width, height, layout, mx, my, div_luma, div_chroma, out,
            );
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let mut y_blk = [0i16; 64];
            let mut cb_blk = [0i16; 64];
            let mut cr_blk = [0i16; 64];
            extract_block_ycbcr_rgb(
                pixels,
                width,
                height,
                layout,
                mx,
                my,
                &mut y_blk,
                &mut cb_blk,
                &mut cr_blk,
            );
            out[0] = quantize_block(&mut y_blk, div_luma);
            out[1] = quantize_block(&mut cb_blk, div_chroma);
            out[2] = quantize_block(&mut cr_blk, div_chroma);
        }
    }
}

pub(crate) struct Yuv420Scheme;
impl SamplingScheme for Yuv420Scheme {
    const H_V: (u8, u8) = (2, 2);
    const MCU_W: u32 = 16;
    const MCU_H: u32 = 16;
    const BLOCKS_PER_MCU: usize = 6;
    const Y_BLOCKS_PER_MCU: usize = 4;

    fn quantize_one_mcu_per_comp(
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        mx: u32,
        my: u32,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        y_blocks: &mut Vec<[i16; 64]>,
        cb_blocks: &mut Vec<[i16; 64]>,
        cr_blocks: &mut Vec<[i16; 64]>,
    ) {
        let mut ys = [[0i16; 64]; 4];
        let mut cb = [0i16; 64];
        let mut cr = [0i16; 64];
        color::extract_mcu_420(
            pixels,
            width,
            height,
            layout,
            mx * Self::MCU_W,
            my * Self::MCU_H,
            &mut ys,
            &mut cb,
            &mut cr,
        );
        for y in ys.iter_mut() {
            arch::backend::dct::fdct_islow(y);
            y_blocks.push(quant::quantize_and_zigzag(y, div_luma));
        }
        arch::backend::dct::fdct_islow(&mut cb);
        cb_blocks.push(quant::quantize_and_zigzag(&cb, div_chroma));
        arch::backend::dct::fdct_islow(&mut cr);
        cr_blocks.push(quant::quantize_and_zigzag(&cr, div_chroma));
    }

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

    fn quantize_one_mcu(
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        mx: u32,
        my: u32,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        out: &mut [[i16; 64]],
    ) {
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
        for (i, blk) in y_blocks.iter_mut().enumerate() {
            out[i] = quantize_block(blk, div_luma);
        }
        out[4] = quantize_block(&mut cb_blk, div_chroma);
        out[5] = quantize_block(&mut cr_blk, div_chroma);
    }
}

pub(crate) struct Yuv422Scheme;
impl SamplingScheme for Yuv422Scheme {
    const H_V: (u8, u8) = (2, 1);
    const MCU_W: u32 = 16;
    const MCU_H: u32 = 8;
    const BLOCKS_PER_MCU: usize = 4;
    const Y_BLOCKS_PER_MCU: usize = 2;

    fn quantize_one_mcu_per_comp(
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        mx: u32,
        my: u32,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        y_blocks: &mut Vec<[i16; 64]>,
        cb_blocks: &mut Vec<[i16; 64]>,
        cr_blocks: &mut Vec<[i16; 64]>,
    ) {
        let mut blocks = [[0i16; 64]; 4];
        quantize_mcu_422_rgb(
            pixels,
            width,
            height,
            layout,
            mx,
            my,
            div_luma,
            div_chroma,
            &mut blocks,
        );
        y_blocks.push(blocks[0]);
        y_blocks.push(blocks[1]);
        cb_blocks.push(blocks[2]);
        cr_blocks.push(blocks[3]);
    }

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
        let mut blocks = [[0i16; 64]; 4];
        quantize_mcu_422_rgb(
            pixels,
            width,
            height,
            layout,
            mx,
            my,
            div_luma,
            div_chroma,
            &mut blocks,
        );
        for blk in &blocks[..2] {
            prev_dc.y = encode_block(bw, blk, prev_dc.y, dc_luma, ac_luma)?;
        }
        prev_dc.cb = encode_block(bw, &blocks[2], prev_dc.cb, dc_chroma, ac_chroma)?;
        prev_dc.cr = encode_block(bw, &blocks[3], prev_dc.cr, dc_chroma, ac_chroma)?;
        Ok(())
    }

    fn quantize_one_mcu(
        pixels: &[u8],
        width: u32,
        height: u32,
        layout: PixelLayout,
        mx: u32,
        my: u32,
        div_luma: &Divisors,
        div_chroma: &Divisors,
        out: &mut [[i16; 64]],
    ) {
        quantize_mcu_422_rgb(
            pixels, width, height, layout, mx, my, div_luma, div_chroma, out,
        );
    }
}

/// Pick the serial vs threaded path based on the `threads` knob, then
/// monomorphize the scheme. `threads == 1` keeps the original
/// single-thread call (= byte-exact regression-safe). `threads == 0`
/// runs the threaded path on the ambient rayon pool. `threads > 1`
/// builds a private pool sized to exactly that many workers, runs the
/// scan inside `install`, and drops the pool on return so callers'
/// global pools are left untouched.
#[allow(clippy::too_many_arguments)]
fn dispatch_scan<W: Write>(
    threads: u32,
    subsampling: ChromaSubsampling,
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
    if matches!(subsampling, ChromaSubsampling::Yuv420) && layout == RGB {
        if threads == 1 {
            return encode_scan_420_rgb(
                pixels,
                width,
                height,
                bw,
                div_luma,
                div_chroma,
                dc_luma,
                ac_luma,
                dc_chroma,
                ac_chroma,
                restart_interval,
            );
        }

        let mcus_x = width.div_ceil(Yuv420Scheme::MCU_W);
        let rows = run_on_pool(threads, || {
            parallel_quantize_rows_420_rgb(pixels, width, height, div_luma, div_chroma)
        })?;

        return serial_emit_rows::<Yuv420Scheme, _>(
            &rows,
            mcus_x,
            bw,
            dc_luma,
            ac_luma,
            dc_chroma,
            ac_chroma,
            restart_interval,
        );
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
    if threads == 1 && matches!(subsampling, ChromaSubsampling::Yuv444) && layout == RGB {
        return encode_scan_444_rgb_pair(
            pixels,
            width,
            height,
            bw,
            div_luma,
            div_chroma,
            dc_luma,
            ac_luma,
            dc_chroma,
            ac_chroma,
            restart_interval,
        );
    }

    if threads == 1 {
        return dispatch_scheme!(subsampling, S => encode_scan::<S, _>(
            pixels,
            width,
            height,
            layout,
            bw,
            div_luma,
            div_chroma,
            dc_luma,
            ac_luma,
            dc_chroma,
            ac_chroma,
            restart_interval,
        ));
    }

    #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
    if matches!(subsampling, ChromaSubsampling::Yuv444) && layout == RGB {
        let mcus_x = width.div_ceil(Yuv444Scheme::MCU_W);
        let rows = run_on_pool(threads, || {
            parallel_quantize_rows_444_rgb_pair(pixels, width, height, div_luma, div_chroma)
        })?;

        return serial_emit_rows::<Yuv444Scheme, _>(
            &rows,
            mcus_x,
            bw,
            dc_luma,
            ac_luma,
            dc_chroma,
            ac_chroma,
            restart_interval,
        );
    }

    // Front half on the chosen pool, back half always on the caller's
    // thread. This keeps the parallel work pool-scoped without forcing
    // `W: Send` on the bit writer.
    let (mcus_x, rows) = dispatch_scheme!(subsampling, S => (
        width.div_ceil(S::MCU_W),
        run_on_pool(threads, || {
            parallel_quantize_rows::<S>(pixels, width, height, layout, div_luma, div_chroma)
        })?,
    ));

    dispatch_scheme!(subsampling, S => serial_emit_rows::<S, _>(
        &rows,
        mcus_x,
        bw,
        dc_luma,
        ac_luma,
        dc_chroma,
        ac_chroma,
        restart_interval,
    ))
}

/// Run `f` on the ambient rayon pool (`threads == 0`) or on a freshly
/// constructed local pool of `threads` workers (`threads > 1`). The
/// local pool is dropped on return so the caller's global pool is left
/// undisturbed.
fn run_on_pool<F, R>(threads: u32, f: F) -> io::Result<R>
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    if threads == 0 {
        Ok(f())
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads as usize)
            .build()
            .map_err(io::Error::other)?;
        Ok(pool.install(f))
    }
}

#[allow(clippy::too_many_arguments)]
fn quantize_mcu_420_rgb(
    pixels: &[u8],
    width: u32,
    height: u32,
    mx: u32,
    my: u32,
    div_luma: &Divisors,
    div_chroma: &Divisors,
    out: &mut [[i16; 64]],
) {
    let x0 = mx * Yuv420Scheme::MCU_W;
    let y0 = my * Yuv420Scheme::MCU_H;
    if x0 + Yuv420Scheme::MCU_W <= width && y0 + Yuv420Scheme::MCU_H <= height {
        arch::backend::encode::quantize_mcu_420_rgb_full(
            pixels, width, x0, y0, div_luma, div_chroma, out,
        );
    } else {
        let mut y_blocks = [[0i16; 64]; 4];
        let mut cb_blk = [0i16; 64];
        let mut cr_blk = [0i16; 64];
        color::extract_mcu_420(
            pixels,
            width,
            height,
            RGB,
            x0,
            y0,
            &mut y_blocks,
            &mut cb_blk,
            &mut cr_blk,
        );
        for (i, blk) in y_blocks.iter_mut().enumerate() {
            out[i] = quantize_block(blk, div_luma);
        }
        out[4] = quantize_block(&mut cb_blk, div_chroma);
        out[5] = quantize_block(&mut cr_blk, div_chroma);
    }
}

#[allow(clippy::too_many_arguments)]
fn quantize_mcu_422_rgb(
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    mx: u32,
    my: u32,
    div_luma: &Divisors,
    div_chroma: &Divisors,
    out: &mut [[i16; 64]],
) {
    let x0 = mx * Yuv422Scheme::MCU_W;
    let y0 = my * Yuv422Scheme::MCU_H;
    if layout == RGB && x0 + Yuv422Scheme::MCU_W <= width && y0 + Yuv422Scheme::MCU_H <= height {
        arch::backend::encode::quantize_mcu_422_rgb_full(
            pixels, width, x0, y0, div_luma, div_chroma, out,
        );
    } else {
        let mut y_blocks = [[0i16; 64]; 2];
        let mut cb_blk = [0i16; 64];
        let mut cr_blk = [0i16; 64];
        extract_mcu_422_rgb(
            pixels,
            width,
            height,
            layout,
            mx,
            my,
            &mut y_blocks,
            &mut cb_blk,
            &mut cr_blk,
        );
        for (i, blk) in y_blocks.iter_mut().enumerate() {
            out[i] = quantize_block(blk, div_luma);
        }
        out[2] = quantize_block(&mut cb_blk, div_chroma);
        out[3] = quantize_block(&mut cr_blk, div_chroma);
    }
}

#[allow(clippy::too_many_arguments)]
fn extract_mcu_422_rgb(
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    mx: u32,
    my: u32,
    y_blocks: &mut [[i16; 64]; 2],
    cb_blk: &mut [i16; 64],
    cr_blk: &mut [i16; 64],
) {
    let x0 = mx * Yuv422Scheme::MCU_W;
    let y0 = my * Yuv422Scheme::MCU_H;
    if layout == RGB && x0 + Yuv422Scheme::MCU_W <= width && y0 + Yuv422Scheme::MCU_H <= height {
        color::extract_mcu_422_rgb_full(pixels, width, x0, y0, y_blocks, cb_blk, cr_blk);
    } else {
        color::extract_mcu_422(
            pixels, width, height, layout, x0, y0, y_blocks, cb_blk, cr_blk,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_one_mcu_420_rgb<W: Write>(
    bw: &mut BitWriter<W>,
    pixels: &[u8],
    width: u32,
    height: u32,
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
    let mut blocks = [[0i16; 64]; Yuv420Scheme::BLOCKS_PER_MCU];
    quantize_mcu_420_rgb(
        pixels,
        width,
        height,
        mx,
        my,
        div_luma,
        div_chroma,
        &mut blocks,
    );
    for blk in &blocks[..Yuv420Scheme::Y_BLOCKS_PER_MCU] {
        prev_dc.y = encode_block(bw, blk, prev_dc.y, dc_luma, ac_luma)?;
    }
    prev_dc.cb = encode_block(bw, &blocks[4], prev_dc.cb, dc_chroma, ac_chroma)?;
    prev_dc.cr = encode_block(bw, &blocks[5], prev_dc.cr, dc_chroma, ac_chroma)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_scan_420_rgb<W: Write>(
    pixels: &[u8],
    width: u32,
    height: u32,
    bw: &mut BitWriter<W>,
    div_luma: &Divisors,
    div_chroma: &Divisors,
    dc_luma: &HuffmanTable,
    ac_luma: &HuffmanTable,
    dc_chroma: &HuffmanTable,
    ac_chroma: &HuffmanTable,
    restart_interval: u16,
) -> io::Result<()> {
    let mcus_x = width.div_ceil(Yuv420Scheme::MCU_W);
    let mcus_y = height.div_ceil(Yuv420Scheme::MCU_H);
    let mut prev_dc = DcPredictors::default();
    let restart_interval = restart_interval as u32;
    let mut mcus_since_rst: u32 = 0;
    let mut next_rst: u8 = 0;
    let mut mcu_count: u64 = 0;
    let total_mcus = mcus_x as u64 * mcus_y as u64;
    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            if restart_interval > 0 && mcus_since_rst == restart_interval && mcu_count < total_mcus
            {
                bw.write_restart(next_rst)?;
                next_rst = (next_rst + 1) & 7;
                mcus_since_rst = 0;
                prev_dc = DcPredictors::default();
            }
            encode_one_mcu_420_rgb(
                bw,
                pixels,
                width,
                height,
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
            if restart_interval > 0 && mcus_since_rst == restart_interval && mcu_count < total_mcus
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

#[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
#[allow(clippy::too_many_arguments)]
fn encode_scan_444_rgb_pair<W: Write>(
    pixels: &[u8],
    width: u32,
    height: u32,
    bw: &mut BitWriter<W>,
    div_luma: &Divisors,
    div_chroma: &Divisors,
    dc_luma: &HuffmanTable,
    ac_luma: &HuffmanTable,
    dc_chroma: &HuffmanTable,
    ac_chroma: &HuffmanTable,
    restart_interval: u16,
) -> io::Result<()> {
    let mcus_x = width.div_ceil(Yuv444Scheme::MCU_W);
    let mcus_y = height.div_ceil(Yuv444Scheme::MCU_H);
    let mut prev_dc = DcPredictors::default();
    let restart_interval = restart_interval as u32;
    let mut mcus_since_rst: u32 = 0;
    let mut next_rst: u8 = 0;
    let mut mcu_count: u64 = 0;
    let total_mcus = mcus_x as u64 * mcus_y as u64;

    for my in 0..mcus_y {
        let y0 = my * Yuv444Scheme::MCU_H;
        let mut mx = 0;
        while mx + 1 < mcus_x {
            let x0 = mx * Yuv444Scheme::MCU_W;
            if x0 + Yuv444Scheme::MCU_W * 2 <= width && y0 + Yuv444Scheme::MCU_H <= height {
                let mut blocks = [[0i16; 64]; 6];
                arch::backend::encode::quantize_mcu_444_rgb_pair_full(
                    pixels,
                    width,
                    x0,
                    y0,
                    div_luma,
                    div_chroma,
                    &mut blocks,
                );

                write_restart_if_needed(
                    bw,
                    restart_interval,
                    &mut mcus_since_rst,
                    &mut next_rst,
                    &mut prev_dc,
                    mcu_count,
                    total_mcus,
                )?;
                emit_444_blocks(
                    bw,
                    &blocks[..3],
                    &mut prev_dc,
                    dc_luma,
                    ac_luma,
                    dc_chroma,
                    ac_chroma,
                )?;
                mcus_since_rst += 1;
                mcu_count += 1;

                write_restart_if_needed(
                    bw,
                    restart_interval,
                    &mut mcus_since_rst,
                    &mut next_rst,
                    &mut prev_dc,
                    mcu_count,
                    total_mcus,
                )?;
                emit_444_blocks(
                    bw,
                    &blocks[3..6],
                    &mut prev_dc,
                    dc_luma,
                    ac_luma,
                    dc_chroma,
                    ac_chroma,
                )?;
                mcus_since_rst += 1;
                mcu_count += 1;
                mx += 2;
            } else {
                write_restart_if_needed(
                    bw,
                    restart_interval,
                    &mut mcus_since_rst,
                    &mut next_rst,
                    &mut prev_dc,
                    mcu_count,
                    total_mcus,
                )?;
                Yuv444Scheme::encode_one_mcu(
                    bw,
                    pixels,
                    width,
                    height,
                    RGB,
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
                mx += 1;
            }
        }

        while mx < mcus_x {
            write_restart_if_needed(
                bw,
                restart_interval,
                &mut mcus_since_rst,
                &mut next_rst,
                &mut prev_dc,
                mcu_count,
                total_mcus,
            )?;
            Yuv444Scheme::encode_one_mcu(
                bw,
                pixels,
                width,
                height,
                RGB,
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
            mx += 1;
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
#[allow(clippy::too_many_arguments)]
fn write_restart_if_needed<W: Write>(
    bw: &mut BitWriter<W>,
    restart_interval: u32,
    mcus_since_rst: &mut u32,
    next_rst: &mut u8,
    prev_dc: &mut DcPredictors,
    mcu_count: u64,
    total_mcus: u64,
) -> io::Result<()> {
    if restart_interval > 0 && *mcus_since_rst == restart_interval && mcu_count < total_mcus {
        bw.write_restart(*next_rst)?;
        *next_rst = (*next_rst + 1) & 7;
        *mcus_since_rst = 0;
        *prev_dc = DcPredictors::default();
    }
    Ok(())
}

#[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
#[allow(clippy::too_many_arguments)]
fn emit_444_blocks<W: Write>(
    bw: &mut BitWriter<W>,
    blocks: &[[i16; 64]],
    prev_dc: &mut DcPredictors,
    dc_luma: &HuffmanTable,
    ac_luma: &HuffmanTable,
    dc_chroma: &HuffmanTable,
    ac_chroma: &HuffmanTable,
) -> io::Result<()> {
    prev_dc.y = encode_block(bw, &blocks[0], prev_dc.y, dc_luma, ac_luma)?;
    prev_dc.cb = encode_block(bw, &blocks[1], prev_dc.cb, dc_chroma, ac_chroma)?;
    prev_dc.cr = encode_block(bw, &blocks[2], prev_dc.cr, dc_chroma, ac_chroma)?;
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

/// Forward DCT + quantize + zigzag, returning the zig-zag-ordered
/// coefficient block. Split from the entropy step so the threaded scan
/// can run this half in parallel and emit serially.
fn quantize_block(block: &mut [i16; 64], div: &Divisors) -> [i16; 64] {
    arch::backend::dct::fdct_islow(block);
    quant::quantize_and_zigzag(block, div)
}

#[allow(clippy::too_many_arguments)]
fn quantize_one_mcu_420_rgb(
    pixels: &[u8],
    width: u32,
    height: u32,
    mx: u32,
    my: u32,
    div_luma: &Divisors,
    div_chroma: &Divisors,
    out: &mut [[i16; 64]],
) {
    quantize_mcu_420_rgb(pixels, width, height, mx, my, div_luma, div_chroma, out);
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
fn quantize_mcu_444_rgb(
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    mx: u32,
    my: u32,
    div_luma: &Divisors,
    div_chroma: &Divisors,
    out: &mut [[i16; 64]],
) {
    let x0 = mx * Yuv444Scheme::MCU_W;
    let y0 = my * Yuv444Scheme::MCU_H;
    if layout == RGB && x0 + Yuv444Scheme::MCU_W <= width && y0 + Yuv444Scheme::MCU_H <= height {
        arch::backend::encode::quantize_mcu_444_rgb_full(
            pixels, width, x0, y0, div_luma, div_chroma, out,
        );
    } else {
        let mut y_blk = [0i16; 64];
        let mut cb_blk = [0i16; 64];
        let mut cr_blk = [0i16; 64];
        extract_block_ycbcr_rgb(
            pixels,
            width,
            height,
            layout,
            mx,
            my,
            &mut y_blk,
            &mut cb_blk,
            &mut cr_blk,
        );
        out[0] = quantize_block(&mut y_blk, div_luma);
        out[1] = quantize_block(&mut cb_blk, div_chroma);
        out[2] = quantize_block(&mut cr_blk, div_chroma);
    }
}

/// Parallel front half of the threaded scan: for each MCU row, color
/// convert + forward DCT + quantize + zigzag every MCU and return the
/// row's quantized blocks. Pure data in / pure data out — no
/// references to the bit writer, so this is the only piece that needs
/// to be Send-friendly.
fn parallel_quantize_rows<S: SamplingScheme>(
    pixels: &[u8],
    width: u32,
    height: u32,
    layout: PixelLayout,
    div_luma: &Divisors,
    div_chroma: &Divisors,
) -> Vec<Vec<[i16; 64]>> {
    let mcus_x = width.div_ceil(S::MCU_W);
    let mcus_y = height.div_ceil(S::MCU_H);
    let blocks_per_mcu = S::BLOCKS_PER_MCU;
    let blocks_per_row = (mcus_x as usize) * blocks_per_mcu;

    (0..mcus_y)
        .into_par_iter()
        .map(|my| {
            let mut row: Vec<[i16; 64]> = vec![[0i16; 64]; blocks_per_row];
            for mx in 0..mcus_x {
                let start = (mx as usize) * blocks_per_mcu;
                let slot = &mut row[start..start + blocks_per_mcu];
                S::quantize_one_mcu(
                    pixels, width, height, layout, mx, my, div_luma, div_chroma, slot,
                );
            }
            row
        })
        .collect()
}

#[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
fn parallel_quantize_rows_444_rgb_pair(
    pixels: &[u8],
    width: u32,
    height: u32,
    div_luma: &Divisors,
    div_chroma: &Divisors,
) -> Vec<Vec<[i16; 64]>> {
    let mcus_x = width.div_ceil(Yuv444Scheme::MCU_W);
    let mcus_y = height.div_ceil(Yuv444Scheme::MCU_H);
    let blocks_per_mcu = Yuv444Scheme::BLOCKS_PER_MCU;
    let blocks_per_row = (mcus_x as usize) * blocks_per_mcu;

    (0..mcus_y)
        .into_par_iter()
        .map(|my| {
            let mut row: Vec<[i16; 64]> = vec![[0i16; 64]; blocks_per_row];
            let y0 = my * Yuv444Scheme::MCU_H;
            let mut mx = 0;
            while mx + 1 < mcus_x {
                let x0 = mx * Yuv444Scheme::MCU_W;
                let start = (mx as usize) * blocks_per_mcu;
                let slot = &mut row[start..start + blocks_per_mcu * 2];
                if x0 + Yuv444Scheme::MCU_W * 2 <= width && y0 + Yuv444Scheme::MCU_H <= height {
                    arch::backend::encode::quantize_mcu_444_rgb_pair_full(
                        pixels, width, x0, y0, div_luma, div_chroma, slot,
                    );
                    mx += 2;
                } else {
                    Yuv444Scheme::quantize_one_mcu(
                        pixels,
                        width,
                        height,
                        RGB,
                        mx,
                        my,
                        div_luma,
                        div_chroma,
                        &mut slot[..3],
                    );
                    mx += 1;
                }
            }

            if mx < mcus_x {
                let start = (mx as usize) * blocks_per_mcu;
                let slot = &mut row[start..start + blocks_per_mcu];
                Yuv444Scheme::quantize_one_mcu(
                    pixels, width, height, RGB, mx, my, div_luma, div_chroma, slot,
                );
            }

            row
        })
        .collect()
}

fn parallel_quantize_rows_420_rgb(
    pixels: &[u8],
    width: u32,
    height: u32,
    div_luma: &Divisors,
    div_chroma: &Divisors,
) -> Vec<Vec<[i16; 64]>> {
    let mcus_x = width.div_ceil(Yuv420Scheme::MCU_W);
    let mcus_y = height.div_ceil(Yuv420Scheme::MCU_H);
    let blocks_per_mcu = Yuv420Scheme::BLOCKS_PER_MCU;
    let blocks_per_row = (mcus_x as usize) * blocks_per_mcu;

    (0..mcus_y)
        .into_par_iter()
        .map(|my| {
            let mut row: Vec<[i16; 64]> = vec![[0i16; 64]; blocks_per_row];
            for mx in 0..mcus_x {
                let start = (mx as usize) * blocks_per_mcu;
                let slot = &mut row[start..start + blocks_per_mcu];
                quantize_one_mcu_420_rgb(pixels, width, height, mx, my, div_luma, div_chroma, slot);
            }
            row
        })
        .collect()
}

/// Serial back half: walk the parallel front half's output in raster
/// order and emit Huffman bits. DC predictor chain and RSTn bookkeeping
/// mirror [`encode_scan`] exactly so the byte stream is identical.
///
/// `rows.len()` is the number of MCU rows; `mcus_x` is MCUs per row.
/// `total_mcus` (used for the "no trailing RSTn" check) is therefore
/// `mcus_x * rows.len()`.
#[allow(clippy::too_many_arguments)]
fn serial_emit_rows<S: SamplingScheme, W: Write>(
    rows: &[Vec<[i16; 64]>],
    mcus_x: u32,
    bw: &mut BitWriter<W>,
    dc_luma: &HuffmanTable,
    ac_luma: &HuffmanTable,
    dc_chroma: &HuffmanTable,
    ac_chroma: &HuffmanTable,
    restart_interval: u16,
) -> io::Result<()> {
    let blocks_per_mcu = S::BLOCKS_PER_MCU;
    let y_blocks = S::Y_BLOCKS_PER_MCU;
    let mut prev_dc = DcPredictors::default();
    let restart_interval = restart_interval as u32;
    let mut mcus_since_rst: u32 = 0;
    let mut next_rst: u8 = 0;
    let mut mcu_count: u64 = 0;
    let total_mcus = mcus_x as u64 * rows.len() as u64;
    for row in rows {
        for mx in 0..mcus_x as usize {
            if restart_interval > 0 && mcus_since_rst == restart_interval && mcu_count < total_mcus
            {
                bw.write_restart(next_rst)?;
                next_rst = (next_rst + 1) & 7;
                mcus_since_rst = 0;
                prev_dc = DcPredictors::default();
            }
            let start = mx * blocks_per_mcu;
            let blocks = &row[start..start + blocks_per_mcu];
            for blk in &blocks[..y_blocks] {
                prev_dc.y = encode_block(bw, blk, prev_dc.y, dc_luma, ac_luma)?;
            }
            prev_dc.cb = encode_block(bw, &blocks[y_blocks], prev_dc.cb, dc_chroma, ac_chroma)?;
            prev_dc.cr = encode_block(bw, &blocks[y_blocks + 1], prev_dc.cr, dc_chroma, ac_chroma)?;
            mcus_since_rst += 1;
            mcu_count += 1;
        }
    }
    Ok(())
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

    #[test]
    fn rejects_dimensions_above_jpeg_sof_limit() {
        let mut out = Vec::new();
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        let pixels = vec![0u8; (u16::MAX as usize + 1) * 3];

        let err = enc.encode_rgb(&pixels, u16::MAX as u32 + 1, 1).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

        let err = enc.encode_rgb(&pixels, 1, u16::MAX as u32 + 1).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn max_sof_dimensions_still_use_buffer_validation() {
        let mut out = Vec::new();
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        let err = enc
            .encode_rgb(&[], u16::MAX as u32, u16::MAX as u32)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("pixel buffer too small"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_zero_custom_quant_entries() {
        let pixels = vec![0u8; 8 * 8 * 3];

        let mut out = Vec::new();
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        let mut luma = [1u8; 64];
        luma[5] = 0;
        enc.set_quant_tables(luma, [1u8; 64]);
        let err = enc.encode_rgb(&pixels, 8, 8).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);

        let mut out = Vec::new();
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        let mut chroma = [1u8; 64];
        chroma[20] = 0;
        enc.set_quant_tables([1u8; 64], chroma);
        let err = enc.encode_rgb(&pixels, 8, 8).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn accepts_nonzero_custom_quant_entries() {
        let pixels = vec![0u8; 8 * 8 * 3];
        let mut out = Vec::new();
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.set_quant_tables([1u8; 64], [1u8; 64]);
        enc.encode_rgb(&pixels, 8, 8).unwrap();
    }
}
