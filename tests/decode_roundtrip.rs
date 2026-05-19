//! Self-roundtrip tests for the decoder: encode with our encoder,
//! decode with our decoder, verify PSNR against the input. The
//! companion `roundtrip.rs` already exercises encode→image-crate-
//! decode → input direction; this file exercises the symmetric
//! encode→ours-decode → input direction so both halves of the
//! pipeline are independently anchored.

use jpeg_rusturbo::decode::Decoder;
use jpeg_rusturbo::{ChromaSubsampling, JpegEncoder, PixelFormat};

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

fn psnr_rgb(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
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

fn encode_then_decode(
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
    let decoder = Decoder::new(&jpeg).expect("parse headers");
    let info = decoder.info();
    assert_eq!(info.width, w);
    assert_eq!(info.height, h);
    decoder.decode(PixelFormat::Rgb).expect("decode")
}

fn run_self_roundtrip(w: u32, h: u32, q: u8, sub: ChromaSubsampling, min_psnr: f64) {
    let rgb = gradient_rgb(w, h);
    let decoded = encode_then_decode(&rgb, w, h, q, sub);
    assert_eq!(decoded.len(), (w * h * 3) as usize);
    let psnr = psnr_rgb(&rgb, &decoded);
    assert!(
        psnr >= min_psnr,
        "self-roundtrip PSNR {psnr:.2} dB below floor {min_psnr:.2} dB ({w}x{h} q={q} {sub:?})",
    );
}

#[test]
fn self_roundtrip_16x16_q80_420() {
    run_self_roundtrip(16, 16, 80, ChromaSubsampling::Yuv420, 28.0);
}

#[test]
fn self_roundtrip_17x17_q80_420() {
    run_self_roundtrip(17, 17, 80, ChromaSubsampling::Yuv420, 28.0);
}

#[test]
fn self_roundtrip_17x17_q80_444() {
    run_self_roundtrip(17, 17, 80, ChromaSubsampling::Yuv444, 32.0);
}

#[test]
fn self_roundtrip_16x8_q80_422() {
    run_self_roundtrip(16, 8, 80, ChromaSubsampling::Yuv422, 30.0);
}

#[test]
fn self_roundtrip_1080p_q80_420() {
    run_self_roundtrip(1920, 1080, 80, ChromaSubsampling::Yuv420, 32.0);
}

#[test]
fn self_roundtrip_1080p_q80_422() {
    run_self_roundtrip(1920, 1080, 80, ChromaSubsampling::Yuv422, 32.0);
}

#[test]
fn self_roundtrip_matches_image_crate_pixels() {
    // The "true" decoder reference is libjpeg-turbo, but since we
    // don't want to take that dependency, use `image`'s decoder as a
    // proxy and assert our decoded pixels match its decoded pixels
    // exactly. (Both should produce the same JPEG-conforming output
    // for our encoder's bytes.)
    use image::{ImageFormat, ImageReader};
    use std::io::Cursor;

    let w = 320;
    let h = 240;
    let rgb = gradient_rgb(w, h);
    let mut jpeg = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut jpeg, 80);
    enc.encode_rgb(&rgb, w, h).unwrap();

    let ours = Decoder::new(&jpeg)
        .unwrap()
        .decode(PixelFormat::Rgb)
        .unwrap();

    let image_dec = ImageReader::with_format(Cursor::new(&jpeg), ImageFormat::Jpeg)
        .decode()
        .unwrap()
        .to_rgb8();
    let theirs = image_dec.into_raw();

    assert_eq!(ours.len(), theirs.len());
    // Tolerate small per-pixel drift (rounding choices may differ).
    let mut max_diff = 0i32;
    let mut sum_sq: u64 = 0;
    for (a, b) in ours.iter().zip(theirs.iter()) {
        let d = (*a as i32 - *b as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        sum_sq += (d * d) as u64;
    }
    let mse = sum_sq as f64 / ours.len() as f64;
    let psnr = if sum_sq == 0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    };
    assert!(
        max_diff <= 3,
        "our decoder differs from image's by more than 3 per channel (max diff = {max_diff})",
    );
    assert!(
        psnr >= 40.0,
        "PSNR vs image's decoder too low: {psnr:.2} dB"
    );
}

#[test]
fn self_roundtrip_rgba_output() {
    let w = 64;
    let h = 32;
    let rgb = gradient_rgb(w, h);
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, 80)
        .encode_rgb(&rgb, w, h)
        .unwrap();
    let decoded = Decoder::new(&jpeg)
        .unwrap()
        .decode(PixelFormat::Rgba)
        .unwrap();
    assert_eq!(decoded.len(), (w * h * 4) as usize);
    // Alpha must be 0xFF.
    for px in decoded.chunks_exact(4) {
        assert_eq!(px[3], 0xFF, "alpha byte should be 0xFF");
    }
}
