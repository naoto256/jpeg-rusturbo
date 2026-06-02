//! JPEG decoder — baseline + progressive Huffman scans
//! translated from libjpeg-turbo, sharing the
//! `arch::backend` SIMD kernels with the encoder where they overlap.
//!
//! Top-level entry point: [`Decoder::new`] parses the header chain
//! (SOI / DQT / DHT / SOF / SOS) eagerly; [`Decoder::decode`] runs
//! the scan(s) and emits interleaved pixel bytes in the requested
//! [`PixelFormat`].

mod baseline;
mod error;
mod huffman;
mod markers;
mod progressive;

pub use error::{DecodeError, Result};

use std::cell::OnceCell;

use crate::PixelFormat;
use crate::arch;
use crate::color::{PixelClass, PixelLayout};

use baseline::{DecodedPlane, DecodedPlanes};
use markers::{DecoderHeaders, MarkerReader, ScanHeader};

/// Basic image descriptor returned by [`Decoder::info`].
#[derive(Clone, Debug)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub components: u8,
    pub progressive: bool,
}

/// Stateful JPEG decoder. Construct with [`Decoder::new`], inspect
/// dimensions via [`Decoder::info`], then call [`Decoder::decode`]
/// (which consumes the decoder) for the interleaved pixel output.
pub struct Decoder<'a> {
    src: &'a [u8],
    headers: DecoderHeaders,
    first_scan: ScanHeader,
    entropy_start: usize,
    /// Lazily-assembled ICC profile (concatenated APP2 chunks in seq
    /// order). Populated on first call to [`Decoder::icc_profile`].
    /// `Some(Vec)` = valid reassembly, `Some(Vec::new())` is reserved
    /// for "header said empty"; `None` inside the cell after init
    /// means the metadata is malformed (gap / mismatch) and the
    /// accessor returns `None`. The outer `OnceCell` distinguishes
    /// "not yet computed" from "computed".
    icc_cache: OnceCell<Option<Vec<u8>>>,
}

impl<'a> Decoder<'a> {
    /// Parse the JPEG header chain. Returns an error if the stream is
    /// malformed or uses an unsupported feature (e.g. arithmetic
    /// coding, hierarchical mode, 12-bit precision).
    pub fn new(src: &'a [u8]) -> Result<Self> {
        let mut reader = MarkerReader::new(src);
        let (headers, first_scan) = reader.read_to_scan()?;
        let entropy_start = reader.pos();
        Ok(Self {
            src,
            headers,
            first_scan,
            entropy_start,
            icc_cache: OnceCell::new(),
        })
    }

    pub fn info(&self) -> ImageInfo {
        ImageInfo {
            width: self.headers.frame.width as u32,
            height: self.headers.frame.height as u32,
            components: self.headers.frame.components.len() as u8,
            progressive: self.headers.frame.progressive,
        }
    }

    /// Retained EXIF payload, with the 6-byte `Exif\0\0` identifier
    /// stripped. Returns `None` if the stream carried no APP1 segment
    /// with the `Exif\0\0` identifier. Per the EXIF spec a JPEG holds
    /// at most one such segment; if multiple are present this returns
    /// the FIRST one. Other APP1 flavours (Adobe XMP etc.) are
    /// intentionally out of scope for this accessor — `None` here
    /// does NOT prove the file has no XMP.
    ///
    /// The returned slice borrows directly from the source buffer
    /// passed to [`Decoder::new`], so retrieval is zero-copy. The bytes
    /// can be passed straight back to the encoder via
    /// `JpegEncoder::set_exif` to close a decode → operate → re-encode
    /// loop:
    ///
    /// ```no_run
    /// # use jpeg_rusturbo::{decode::Decoder, JpegEncoder, PixelFormat};
    /// # fn run(src: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    /// let decoder = Decoder::new(src)?;
    /// let exif = decoder.exif().map(|b| b.to_vec());
    /// let info = decoder.info();
    /// let pixels = decoder.decode(PixelFormat::Rgb)?;
    /// let mut out = Vec::new();
    /// let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
    /// enc.set_exif(exif);
    /// enc.encode_rgb(&pixels, info.width, info.height)?;
    /// # Ok(()) }
    /// ```
    pub fn exif(&self) -> Option<&[u8]> {
        let range = self.headers.metadata.exif.as_ref()?;
        Some(&self.src[range.clone()])
    }

    /// Reassembled ICC color profile, with all per-segment APP2 chunk
    /// headers (`ICC_PROFILE\0` + seq + total) stripped and the
    /// chunks concatenated in `seq_num` order (1..=total). Returns
    /// `None` if the stream carried no APP2 `ICC_PROFILE\0` segment.
    ///
    /// Malformed metadata also yields `None`:
    /// - a `seq_num` of 0 or greater than `total_segs`,
    /// - a gap in `1..=total_segs` (missing segment),
    /// - inconsistent `total_segs` across segments.
    ///
    /// Duplicate `seq_num` values: the first segment seen wins,
    /// subsequent duplicates are ignored.
    ///
    /// The returned slice borrows from a Vec assembled and cached on
    /// the first call. Subsequent calls are O(1). The bytes can be
    /// passed straight back to the encoder via
    /// `JpegEncoder::set_icc_profile` to close the round-trip loop —
    /// the encoder re-chunks across one or more APP2 segments per the
    /// ICC.1 convention.
    ///
    /// ```no_run
    /// # use jpeg_rusturbo::{decode::Decoder, JpegEncoder, PixelFormat};
    /// # fn run(src: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    /// let decoder = Decoder::new(src)?;
    /// let icc = decoder.icc_profile().map(|b| b.to_vec());
    /// let info = decoder.info();
    /// let pixels = decoder.decode(PixelFormat::Rgb)?;
    /// let mut out = Vec::new();
    /// let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
    /// enc.set_icc_profile(icc);
    /// enc.encode_rgb(&pixels, info.width, info.height)?;
    /// # Ok(()) }
    /// ```
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.icc_cache
            .get_or_init(|| self.assemble_icc())
            .as_deref()
    }

    /// Reassemble the per-segment ICC chunks into a contiguous Vec.
    /// Validates the seq_num / total contract; returns None on any
    /// malformed-metadata condition documented on `icc_profile`.
    fn assemble_icc(&self) -> Option<Vec<u8>> {
        let meta = &self.headers.metadata;
        let total = meta.icc_total?;
        if total == 0 {
            return None;
        }
        if meta.icc_chunks.len() != total as usize {
            // Missing or excess segments (the latter happens if a
            // sender sent a seq > total — we'd have stored it, since
            // we don't validate at parse time to keep the marker walk
            // simple).
            return None;
        }
        let mut out = Vec::new();
        // BTreeMap iterates in seq_num order — that's the assembly
        // order the ICC.1 spec mandates.
        for (i, (seq, range)) in meta.icc_chunks.iter().enumerate() {
            let expected = (i + 1) as u8;
            if *seq != expected {
                // Either seq_num == 0 (sorts before 1) or a gap
                // (e.g. {1, 3} on total=3) — malformed.
                return None;
            }
            if *seq > total {
                return None;
            }
            out.extend_from_slice(&self.src[range.clone()]);
        }
        Some(out)
    }

    /// Decode the stream into a tightly-packed pixel buffer at the
    /// requested [`PixelFormat`]. The buffer is `width * height *
    /// bytes_per_pixel` long, in row-major order.
    pub fn decode(self, format: PixelFormat) -> Result<Vec<u8>> {
        let layout: PixelLayout = format.into();

        let planes = if self.headers.frame.progressive {
            progressive::decode_progressive(
                self.src,
                self.entropy_start,
                &self.headers,
                self.first_scan,
            )?
        } else {
            baseline::decode_baseline_multi(
                self.src,
                self.entropy_start,
                &self.headers,
                self.first_scan,
            )?
        };

        compose_output(&planes, &self.headers, layout)
    }
}

/// One-shot convenience: parse + decode in a single call.
pub fn decode(src: &[u8], format: PixelFormat) -> Result<Vec<u8>> {
    Decoder::new(src)?.decode(format)
}

/// Convert the per-component sample planes into an interleaved
/// `PixelFormat` buffer. Handles 4:4:4 / 4:2:2 / 4:2:0 / 4:1:1 /
/// 4:4:0 chroma layouts via the separable fancy upsample helper, with
/// box-replication fallback for wider sampling factors.
fn compose_output(
    planes: &DecodedPlanes,
    headers: &DecoderHeaders,
    layout: PixelLayout,
) -> Result<Vec<u8>> {
    let width = planes.width as usize;
    let height = planes.height as usize;
    let bpp = layout.bpp;
    // Skip per-decode zero-fill of the output buffer. compose_output
    // writes every byte (`width * bpp` per row × `height` rows) before
    // returning, so the initial contents are never observed.
    // Safety: `u8` has no validity invariants; `set_len` on a freshly-
    // allocated Vec<u8> is sound. The "fully written before read"
    // contract is upheld by the loop bodies below — for 4K RGB this
    // saves ~24 MB of zero-fill page-fault cost.
    let mut out: Vec<u8> = Vec::with_capacity(width * height * bpp);
    #[allow(clippy::uninit_vec)]
    unsafe {
        out.set_len(width * height * bpp);
    }

    let frame = &headers.frame;
    let h_max = frame.h_max() as usize;
    let v_max = frame.v_max() as usize;

    // Dispatch on the layout category. The Cmyk and Gray arms handle
    // their own output shape and return; Rgb falls through to the
    // chroma-upsample + color-convert path below. The per-arch color
    // kernels are only reachable from the Rgb arm.
    match layout.class() {
        PixelClass::Cmyk => {
            // CMYK pass-through output (4-byte C/M/Y/K). Requires a
            // 4-component source — decoding a 3-component (YCbCr)
            // source into CMYK is not in scope.
            if planes.components.len() != 4 {
                return Err(DecodeError::Unsupported(
                    "PixelFormat::Cmyk requires a 4-component source JPEG",
                ));
            }
            for j in 0..height {
                let dst_off = j * width * 4;
                for ch in 0..4 {
                    let plane = &planes.components[ch];
                    let sj = j.min(plane.plane_height.saturating_sub(1));
                    let src_off = sj * plane.stride;
                    let take = width.min(plane.plane_width);
                    // Stride one channel byte per output pixel; row by
                    // row, channel by channel. CMYK is fixed at H=V=1
                    // in our encoder so plane_width == width in the
                    // common case; the .min(plane_width) guard handles
                    // unusual encoders that wrote sampling factors > 1.
                    for i in 0..take {
                        out[dst_off + i * 4 + ch] = plane.samples[src_off + i];
                    }
                    if take < width {
                        let last = if take == 0 {
                            0
                        } else {
                            out[dst_off + (take - 1) * 4 + ch]
                        };
                        for i in take..width {
                            out[dst_off + i * 4 + ch] = last;
                        }
                    }
                }
            }
            return Ok(out);
        }
        PixelClass::Gray | PixelClass::Rgb => {}
    }

    // Reject non-CMYK PixelFormats on a 4-component (CMYK) source:
    // this crate does not perform CMYK→RGB conversion. Callers that
    // need RGB out of a CMYK JPEG should request `PixelFormat::Cmyk`
    // and convert downstream (e.g. via the `image` crate).
    if planes.components.len() == 4 {
        return Err(DecodeError::Unsupported(
            "CMYK source can only be decoded as PixelFormat::Cmyk; \
             this crate does not perform CMYK→RGB conversion",
        ));
    }

    // Single-byte Y output: copy the luma plane directly with no
    // chroma upsample and no color convert. Works for 1-component
    // (grayscale) sources AND 3-component color sources (in the color
    // case, Cb/Cr are discarded — a fast Y-extraction shortcut).
    // Branches *before* the kernel dispatch because the per-arch color
    // kernels are not built to write 1-byte-per-pixel output.
    if layout.class() == PixelClass::Gray {
        if planes.components.is_empty() {
            return Err(DecodeError::Unsupported("no components in frame"));
        }
        // Y is always component[0]: ITU-T T.81 places Y at
        // declaration index 0 for both 1-comp and 3-comp JPEGs, and
        // the encoder side enforces id=1 → index 0.
        let y_plane = &planes.components[0];
        for j in 0..height {
            let sj = j.min(y_plane.plane_height.saturating_sub(1));
            let src_off = sj * y_plane.stride;
            let take = width.min(y_plane.plane_width);
            let dst_off = j * width;
            out[dst_off..dst_off + take].copy_from_slice(&y_plane.samples[src_off..src_off + take]);
            if take < width {
                // Right-edge fill: replicate the last available column
                // — matches the encoder's edge-replication contract.
                let last = if take == 0 {
                    0
                } else {
                    out[dst_off + take - 1]
                };
                out[dst_off + take..dst_off + width].fill(last);
            }
        }
        return Ok(out);
    }

    match planes.components.len() {
        1 => {
            // Single-component (grayscale) image: replicate Y across
            // R/G/B (no chroma planes to consume).
            let y_plane = &planes.components[0];
            let mut y_row = vec![0u8; width];
            let cb = vec![128u8; width];
            let cr = vec![128u8; width];
            for j in 0..height {
                let src_off = j * y_plane.stride;
                y_row.copy_from_slice(&y_plane.samples[src_off..src_off + width]);
                let dst_off = j * width * bpp;
                crate::arch::backend::color::ycc_row_to_rgb(
                    &y_row,
                    &cb,
                    &cr,
                    &mut out[dst_off..dst_off + width * bpp],
                    width,
                    layout,
                );
            }
        }
        3 => {
            // Three-component (Y, Cb, Cr) image. Upsample chroma rows
            // on the fly into per-row scratch buffers; then convert.
            let y_plane = &planes.components[0];
            let cb_plane = &planes.components[1];
            let cr_plane = &planes.components[2];

            // Per-component vertical step: how many y-rows share one
            // chroma row. Equals v_max / Vi.
            let cb_v_step = v_max / (cb_plane.component.v as usize);
            let cr_v_step = v_max / (cr_plane.component.v as usize);
            // Horizontal step: how many output columns share one
            // chroma column. Equals h_max / Hi.
            let cb_h_step = h_max / (cb_plane.component.h as usize);
            let cr_h_step = h_max / (cr_plane.component.h as usize);

            let mut y_row = vec![0u8; width];
            let mut cb_row = vec![0u8; width];
            let mut cr_row = vec![0u8; width];
            // Scratch buffers for the vertically-blended chroma plane
            // row that feeds the horizontal upsample step. Sized to
            // the chroma plane's `plane_width` (one sample per chroma
            // column, not per output column).
            let mut cb_vblend = vec![0u8; cb_plane.plane_width];
            let mut cr_vblend = vec![0u8; cr_plane.plane_width];

            for j in 0..height {
                copy_plane_row(y_plane, j, &mut y_row, width);
                upsample_chroma_row(
                    cb_plane,
                    j,
                    cb_v_step,
                    cb_h_step,
                    &mut cb_vblend,
                    &mut cb_row,
                    width,
                );
                upsample_chroma_row(
                    cr_plane,
                    j,
                    cr_v_step,
                    cr_h_step,
                    &mut cr_vblend,
                    &mut cr_row,
                    width,
                );
                let dst_off = j * width * bpp;
                crate::arch::backend::color::ycc_row_to_rgb(
                    &y_row,
                    &cb_row,
                    &cr_row,
                    &mut out[dst_off..dst_off + width * bpp],
                    width,
                    layout,
                );
            }
        }
        _ => {
            // 4-component (CMYK) is handled by the `PixelClass::Cmyk`
            // / mismatched-pixelformat guards above. Anything else
            // (2-component, 5+) is not a real-world JPEG shape.
            return Err(DecodeError::Unsupported("unsupported component count"));
        }
    }

    Ok(out)
}

/// Copy `width` bytes from `plane.samples` row `j` into `dst`.
/// Out-of-range `j` (e.g. when the component has been vertically
/// downsampled below the output height — non-conventional sampling
/// layouts where Y is not the highest-sampled component) is clamped to
/// the last available row; fancy (interpolating) upsample would address
/// this properly when added.
fn copy_plane_row(plane: &DecodedPlane, j: usize, dst: &mut [u8], width: usize) {
    let j = j.min(plane.plane_height.saturating_sub(1));
    let off = j * plane.stride;
    let take = width.min(plane.plane_width);
    dst[..take].copy_from_slice(&plane.samples[off..off + take]);
    if take < width {
        let last = dst[take - 1];
        dst[take..width].fill(last);
    }
}

/// Upsample one row of chroma into `dst` (output width = `width`),
/// using a separable fancy 2-tap filter for the common `h_step ∈ {1,
/// 2}` × `v_step ∈ {1, 2}` cases and falling back to box replication
/// for the rarer wider sampling factors.
///
/// The vertical-blend buffer `vblend` is the chroma-plane-width scratch
/// where row blending happens before horizontal interpolation; the
/// caller owns it (allocated once per decode) to keep this fn
/// allocation-free in the hot loop.
fn upsample_chroma_row(
    plane: &DecodedPlane,
    j_out: usize,
    v_step: usize,
    h_step: usize,
    vblend: &mut [u8],
    dst: &mut [u8],
    width: usize,
) {
    let plane_w = plane.plane_width;
    let plane_h = plane.plane_height;

    // ---- 1. Vertical blend → vblend (chroma-plane-width row) ----
    if v_step <= 1 {
        // No vertical interpolation needed. Just read the single row.
        let j = j_out.min(plane_h.saturating_sub(1));
        let off = j * plane.stride;
        vblend[..plane_w].copy_from_slice(&plane.samples[off..off + plane_w]);
    } else if v_step == 2 {
        // libjpeg-turbo `h2v2_fancy` vertical pass — see the kernel
        // contract in `arch::backend::sample::h2v2_fancy_vblend`.
        // We pick the neighbor row here (clamped at the top / bottom
        // plane edges) so the kernel itself stays branch-free.
        //
        //   - cur = j_out / 2 (the chroma row this output sits inside)
        //   - phase = j_out & 1 → 0 = upper of pair, neighbor is cur-1
        //                       → 1 = lower of pair, neighbor is cur+1
        let cur = (j_out / 2).min(plane_h.saturating_sub(1));
        let phase = j_out & 1;
        let neighbor = if phase == 0 {
            cur.saturating_sub(1)
        } else {
            (cur + 1).min(plane_h.saturating_sub(1))
        };
        let cur_row = &plane.samples[cur * plane.stride..cur * plane.stride + plane_w];
        let nbr_row = &plane.samples[neighbor * plane.stride..neighbor * plane.stride + plane_w];
        arch::backend::sample::h2v2_fancy_vblend(cur_row, nbr_row, vblend, plane_w);
    } else {
        // v_step > 2 (unusual vertical 3:1 / 4:1 subsampling — no
        // standard layout lands here; 4:4:0 is v_step == 2 above,
        // 4:1:1 is v_step == 1): box-replicate verbatim.
        let cy = (j_out / v_step).min(plane_h.saturating_sub(1));
        let off = cy * plane.stride;
        vblend[..plane_w].copy_from_slice(&plane.samples[off..off + plane_w]);
    }

    // ---- 2. Horizontal upsample of vblend → dst ----
    if h_step <= 1 {
        let take = width.min(plane_w);
        dst[..take].copy_from_slice(&vblend[..take]);
        if take < width {
            let last = if take == 0 { 0 } else { dst[take - 1] };
            dst[take..width].fill(last);
        }
        return;
    }
    if h_step == 2 {
        // libjpeg-turbo `h2_fancy` horizontal pass — see the kernel
        // contract in `arch::backend::sample::h2_fancy_upsample`.
        // The kernel produces exactly `2 * plane_w` bytes; we truncate
        // / pad here to match the requested output `width`.
        if plane_w == 0 {
            dst[..width].fill(0);
            return;
        }
        let full = 2 * plane_w;
        if width >= full {
            arch::backend::sample::h2_fancy_upsample(vblend, dst, plane_w);
            if width > full {
                // Replicate the last produced sample over trailing
                // output columns (= rare: requested output width
                // exceeds 2 * chroma-plane-width).
                let last = dst[full - 1];
                dst[full..width].fill(last);
            }
        } else {
            // Output narrower than the kernel's natural output (=
            // partial right-edge row). Produce the full `2 * plane_w`
            // into a stack scratch and copy the prefix.
            let mut scratch = vec![0u8; full];
            arch::backend::sample::h2_fancy_upsample(vblend, &mut scratch, plane_w);
            dst[..width].copy_from_slice(&scratch[..width]);
        }
        return;
    }
    // h_step > 2: box-replicate each chroma sample `h_step` times.
    let mut x_out = 0usize;
    for &s in vblend.iter().take(plane_w) {
        for _ in 0..h_step {
            if x_out >= width {
                break;
            }
            dst[x_out] = s;
            x_out += 1;
        }
        if x_out >= width {
            break;
        }
    }
    while x_out < width {
        dst[x_out] = dst[x_out - 1];
        x_out += 1;
    }
}
