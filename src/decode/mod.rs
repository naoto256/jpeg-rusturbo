//! JPEG decoder — baseline (and, on the 0.4.0 roadmap, progressive)
//! Huffman scans translated from libjpeg-turbo, sharing the
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

use crate::PixelFormat;
use crate::arch;
use crate::color::PixelLayout;

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
    let mut out = vec![0u8; width * height * bpp];

    let frame = &headers.frame;
    let h_max = frame.h_max() as usize;
    let v_max = frame.v_max() as usize;

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
        n => {
            return Err(DecodeError::Unsupported(if n == 4 {
                "4-component (CMYK) decode"
            } else {
                "unsupported component count"
            }));
        }
    }

    Ok(out)
}

/// Copy `width` bytes from `plane.samples` row `j` into `dst`.
/// Out-of-range `j` (e.g. when the component has been vertically
/// downsampled below the output height — non-conventional sampling
/// layouts where Y is not the highest-sampled component) is clamped to
/// the last available row; fancy upsample will address this properly in
/// the 0.4.0 chroma-upsample refactor.
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
        // v_step > 2 (4:1:1 / 4:4:0 / unusual): box-replicate verbatim.
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
