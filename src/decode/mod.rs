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
/// 4:4:0 chroma layouts via per-row upsample replication.
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

            for j in 0..height {
                copy_plane_row(y_plane, j, &mut y_row, width);
                expand_chroma_row(cb_plane, j / cb_v_step, &mut cb_row, width, cb_h_step);
                expand_chroma_row(cr_plane, j / cr_v_step, &mut cr_row, width, cr_h_step);
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

/// Expand chroma plane row `j` into `dst` at output width `width`, with
/// each chroma sample replicated `h_step` times (box upsample). This
/// covers 4:4:4 / 4:2:2 / 4:2:0 / 4:1:1 / 4:4:0 just by varying h_step.
fn expand_chroma_row(plane: &DecodedPlane, j: usize, dst: &mut [u8], width: usize, h_step: usize) {
    let j = j.min(plane.plane_height.saturating_sub(1));
    let off = j * plane.stride;
    if h_step == 1 {
        let take = width.min(plane.plane_width);
        dst[..take].copy_from_slice(&plane.samples[off..off + take]);
        if take < width {
            let last = dst[take - 1];
            dst[take..width].fill(last);
        }
        return;
    }
    // Box-replicate each chroma sample `h_step` times. The plane width
    // is `ceil(width / h_step)` in samples; replication regenerates
    // the per-pixel chroma signal.
    let mut x_out = 0usize;
    let plane_w = plane.plane_width;
    for xi in 0..plane_w {
        let s = plane.samples[off + xi];
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
    // Pad trailing (e.g. width not a multiple of h_step).
    while x_out < width {
        dst[x_out] = dst[x_out - 1];
        x_out += 1;
    }
}
