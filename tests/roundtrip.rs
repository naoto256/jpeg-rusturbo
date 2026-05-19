//! Round-trip correctness tests.
//!
//! Encode a known RGB pattern, decode it back with the `image` crate,
//! and check that the decoded pixels are reasonably close to the
//! input (PSNR floor — JPEG is lossy, so bit-exact comparison would
//! be wrong here; byte-identical output against libjpeg-turbo would
//! become a separate goal once we match its coefficient precision).

use image::{ImageFormat, ImageReader};
use jpeg_rusturbo::{ChromaSubsampling, JpegEncoder, PixelFormat};
use std::io::Cursor;

/// Build a smoothly-varying RGB image. Smooth content compresses well
/// AND is faithful through 4:2:0 (sharp chroma edges are where 4:2:0
/// bleeds, and we'd need a tighter PSNR threshold).
fn gradient_rgb(w: u32, h: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = (((x + y) * 255) / (w + h).max(1)) as u8;
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    buf
}

fn rgb_to_rgba(rgb: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 3 * 4);
    for chunk in rgb.chunks_exact(3) {
        out.extend_from_slice(chunk);
        out.push(255);
    }
    out
}

/// Mean PSNR between two RGB buffers, ignoring alpha if present.
fn psnr_rgb(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "psnr buffers must be same length");
    let mut sse: u64 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as i32) - (*y as i32);
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / a.len() as f64;
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

fn encode_and_decode(
    rgb: &[u8],
    w: u32,
    h: u32,
    quality: u8,
    subsampling: ChromaSubsampling,
) -> Vec<u8> {
    let mut jpeg = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut jpeg, quality);
    enc.set_subsampling(subsampling);
    enc.encode_rgb(rgb, w, h).expect("encode");

    // SOI/EOI sanity.
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "SOI");
    assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "EOI");

    let img = ImageReader::with_format(Cursor::new(&jpeg), ImageFormat::Jpeg)
        .decode()
        .expect("image::decode")
        .to_rgb8();
    assert_eq!(img.width(), w, "decoded width mismatch");
    assert_eq!(img.height(), h, "decoded height mismatch");
    img.into_raw()
}

fn run_case(w: u32, h: u32, quality: u8, subsampling: ChromaSubsampling, min_psnr: f64) {
    let rgb = gradient_rgb(w, h);
    let decoded = encode_and_decode(&rgb, w, h, quality, subsampling);
    let psnr = psnr_rgb(&rgb, &decoded);
    assert!(
        psnr >= min_psnr,
        "PSNR {psnr:.2} dB below floor {min_psnr:.2} dB ({}x{} q={} {:?})",
        w,
        h,
        quality,
        subsampling
    );
}

#[test]
fn roundtrip_16x16_q80_420() {
    run_case(16, 16, 80, ChromaSubsampling::Yuv420, 28.0);
}

// 17x17 is the meaningful edge case: it forces both luma 8x8 padding
// and the 4:2:0 16x16 MCU to step over the image edge.
#[test]
fn roundtrip_17x17_q80_420() {
    run_case(17, 17, 80, ChromaSubsampling::Yuv420, 28.0);
}

#[test]
fn roundtrip_17x17_q80_444() {
    run_case(17, 17, 80, ChromaSubsampling::Yuv444, 32.0);
}

#[test]
fn roundtrip_16x8_q80_422() {
    // Smallest MCU-exact 4:2:2 case.
    run_case(16, 8, 80, ChromaSubsampling::Yuv422, 30.0);
}

// 17x9 forces both 8x8 luma padding and 4:2:2 16x8 MCU edge stepping
// (right edge and bottom edge both partial).
#[test]
fn roundtrip_17x9_q80_422() {
    run_case(17, 9, 80, ChromaSubsampling::Yuv422, 28.0);
}

#[test]
fn roundtrip_1080p_q80_422() {
    run_case(1920, 1080, 80, ChromaSubsampling::Yuv422, 32.0);
}

#[test]
fn roundtrip_session_size_q70_420() {
    // The actual 1592x1124 size we see in deployed sessions.
    run_case(1592, 1124, 70, ChromaSubsampling::Yuv420, 30.0);
}

#[test]
fn roundtrip_1080p_q80_420() {
    run_case(1920, 1080, 80, ChromaSubsampling::Yuv420, 32.0);
}

#[test]
fn rgba_input_matches_rgb_input() {
    // RGBA path is the win for our caller (skips a packing pass);
    // verify it produces a valid stream and decodes to ~the same
    // bytes as RGB at the same quality.
    let w = 64;
    let h = 32;
    let rgb = gradient_rgb(w, h);
    let rgba = rgb_to_rgba(&rgb);

    let mut jpeg_a = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg_a, 80)
        .encode_rgb(&rgb, w, h)
        .unwrap();
    let mut jpeg_b = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg_b, 80)
        .encode_rgba(&rgba, w, h)
        .unwrap();

    // Streams are byte-equal: alpha is dropped before any arithmetic,
    // so the YCbCr planes match.
    assert_eq!(
        jpeg_a, jpeg_b,
        "RGB and RGBA at the same quality should match"
    );
}

/// Rearrange RGB bytes into the requested `PixelFormat`. For 4-byte
/// formats, the alpha/pad byte is filled with 0xAA so it's clearly
/// non-zero (a buggy encoder that read it as a color channel would
/// produce visibly wrong colors).
fn rgb_to_format(rgb: &[u8], format: PixelFormat) -> Vec<u8> {
    let pixels = rgb.len() / 3;
    let bpp = match format {
        PixelFormat::Rgb | PixelFormat::Bgr => 3,
        _ => 4,
    };
    let mut out = vec![0u8; pixels * bpp];
    for i in 0..pixels {
        let r = rgb[i * 3];
        let g = rgb[i * 3 + 1];
        let b = rgb[i * 3 + 2];
        let p = i * bpp;
        match format {
            PixelFormat::Rgb => {
                out[p] = r;
                out[p + 1] = g;
                out[p + 2] = b;
            }
            PixelFormat::Bgr => {
                out[p] = b;
                out[p + 1] = g;
                out[p + 2] = r;
            }
            PixelFormat::Rgba => {
                out[p] = r;
                out[p + 1] = g;
                out[p + 2] = b;
                out[p + 3] = 0xAA;
            }
            PixelFormat::Bgra => {
                out[p] = b;
                out[p + 1] = g;
                out[p + 2] = r;
                out[p + 3] = 0xAA;
            }
            PixelFormat::Argb => {
                out[p] = 0xAA;
                out[p + 1] = r;
                out[p + 2] = g;
                out[p + 3] = b;
            }
            PixelFormat::Abgr => {
                out[p] = 0xAA;
                out[p + 1] = b;
                out[p + 2] = g;
                out[p + 3] = r;
            }
            PixelFormat::Rgbx => {
                out[p] = r;
                out[p + 1] = g;
                out[p + 2] = b;
                out[p + 3] = 0xAA;
            }
            PixelFormat::Bgrx => {
                out[p] = b;
                out[p + 1] = g;
                out[p + 2] = r;
                out[p + 3] = 0xAA;
            }
        }
    }
    out
}

#[test]
fn all_pixel_formats_match_rgb_bytes() {
    // Same content rendered through each PixelFormat must produce a
    // byte-identical JPEG stream (the encoder converts to YCbCr from
    // the same R/G/B values regardless of byte ordering).
    let w = 64u32;
    let h = 32u32;
    let rgb = gradient_rgb(w, h);

    let mut baseline = Vec::new();
    JpegEncoder::new_with_quality(&mut baseline, 80)
        .encode(&rgb, w, h, PixelFormat::Rgb)
        .unwrap();

    for fmt in [
        PixelFormat::Rgb,
        PixelFormat::Bgr,
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Argb,
        PixelFormat::Abgr,
        PixelFormat::Rgbx,
        PixelFormat::Bgrx,
    ] {
        let buf = rgb_to_format(&rgb, fmt);
        let mut out = Vec::new();
        JpegEncoder::new_with_quality(&mut out, 80)
            .encode(&buf, w, h, fmt)
            .unwrap();
        assert_eq!(
            out, baseline,
            "{fmt:?} produced different bytes than RGB baseline",
        );
    }
}

#[test]
fn all_pixel_formats_roundtrip_1080p() {
    let w = 1920u32;
    let h = 1080u32;
    let rgb = gradient_rgb(w, h);

    for fmt in [
        PixelFormat::Bgr,
        PixelFormat::Bgra,
        PixelFormat::Argb,
        PixelFormat::Abgr,
        PixelFormat::Rgbx,
        PixelFormat::Bgrx,
    ] {
        let buf = rgb_to_format(&rgb, fmt);
        let mut jpeg = Vec::new();
        JpegEncoder::new_with_quality(&mut jpeg, 80)
            .encode(&buf, w, h, fmt)
            .unwrap();
        let img = ImageReader::with_format(Cursor::new(&jpeg), ImageFormat::Jpeg)
            .decode()
            .unwrap()
            .to_rgb8();
        let psnr = psnr_rgb(&rgb, img.as_raw());
        assert!(
            psnr >= 32.0,
            "{fmt:?} PSNR {psnr:.2} dB below floor (1080p q=80)",
        );
    }
}

// Constant-color image: every pixel of every block decodes to (very
// close to) the input. Catches the kind of scaling/transform mistake
// that produces a recognizable JPEG shape but garbled colors per-block.
#[test]
fn roundtrip_solid_color_exact() {
    let w = 16u32;
    let h = 16u32;
    let mut rgb = Vec::new();
    for _ in 0..(w * h) {
        rgb.extend_from_slice(&[200, 100, 50]);
    }
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, 95)
        .encode_rgb(&rgb, w, h)
        .unwrap();
    let img = ImageReader::with_format(Cursor::new(&jpeg), ImageFormat::Jpeg)
        .decode()
        .unwrap()
        .to_rgb8();
    let p = img.as_raw();
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            for (c, exp) in [(p[i], 200i32), (p[i + 1], 100), (p[i + 2], 50)] {
                let diff = (c as i32 - exp).abs();
                assert!(
                    diff <= 6,
                    "pixel ({x},{y}) channel diff {diff} too large: got {c}, expected {exp}"
                );
            }
        }
    }
}
