//! Self-roundtrip tests for the decoder: encode with our encoder,
//! decode with our decoder, verify PSNR against the input. The
//! companion `roundtrip.rs` already exercises encode→image-crate-
//! decode → input direction; this file exercises the symmetric
//! encode→ours-decode → input direction so both halves of the
//! pipeline are independently anchored.

use jpeg_rusturbo::decode::{DecodeError, Decoder};
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

/// Locate `0xFF 0xC0` (SOF0) by walking length-prefixed segments
/// starting after SOI. Returns the byte offset of `0xFF`.
fn find_sof0_offset(jpeg: &[u8]) -> Option<usize> {
    let mut i = 2usize; // skip SOI (0xFF D8)
    while i + 1 < jpeg.len() {
        if jpeg[i] != 0xFF {
            return None;
        }
        let id = jpeg[i + 1];
        if id == 0xC0 {
            return Some(i);
        }
        // All other segments here are length-prefixed (we're between
        // SOI and SOF0, no standalone markers in that range).
        if i + 3 >= jpeg.len() {
            return None;
        }
        let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        i += 2 + len;
    }
    None
}

/// Patch a JPEG's SOF0 height/width fields in-place. Layout of SOF0
/// payload (after the 2-byte marker + 2-byte length): precision(1) +
/// height(2) + width(2) + Nf(1) + per-component(3 × Nf).
fn patch_sof_dims(jpeg: &mut [u8], width: u16, height: u16) {
    let sof_off = find_sof0_offset(jpeg).expect("SOF0 marker not found");
    let height_bytes = height.to_be_bytes();
    let width_bytes = width.to_be_bytes();
    // 0xFF C0 | len(2) | precision(1) | height(2) | width(2) | …
    jpeg[sof_off + 5] = height_bytes[0];
    jpeg[sof_off + 6] = height_bytes[1];
    jpeg[sof_off + 7] = width_bytes[0];
    jpeg[sof_off + 8] = width_bytes[1];
}

fn encode_minimal_jpeg() -> Vec<u8> {
    let w = 16u32;
    let h = 16u32;
    let rgb = gradient_rgb(w, h);
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, 80)
        .encode_rgb(&rgb, w, h)
        .unwrap();
    jpeg
}

#[test]
fn decode_rejects_zero_height() {
    let mut jpeg = encode_minimal_jpeg();
    patch_sof_dims(&mut jpeg, 16, 0);
    let Err(err) = Decoder::new(&jpeg) else {
        panic!("zero height should be rejected");
    };
    assert!(
        matches!(err, DecodeError::InvalidDimensions(_)),
        "got {err:?}"
    );
}

#[test]
fn decode_rejects_zero_width() {
    let mut jpeg = encode_minimal_jpeg();
    patch_sof_dims(&mut jpeg, 0, 16);
    let Err(err) = Decoder::new(&jpeg) else {
        panic!("zero width should be rejected");
    };
    assert!(
        matches!(err, DecodeError::InvalidDimensions(_)),
        "got {err:?}"
    );
}

#[test]
fn decode_rejects_oversized_dimensions() {
    // 32768 > MAX_DIMENSION (16384) — caps the per-component plane
    // allocation at a sane ceiling so a 50-byte malformed header
    // can't demand multi-gigabyte allocations.
    let mut jpeg = encode_minimal_jpeg();
    patch_sof_dims(&mut jpeg, 32768, 32768);
    let Err(err) = Decoder::new(&jpeg) else {
        panic!("oversized dims should be rejected");
    };
    assert!(
        matches!(err, DecodeError::InvalidDimensions(_)),
        "got {err:?}"
    );
}

#[test]
fn decode_accepts_max_supported_dimension() {
    // 16384 is the cap — equal-or-under should pass header parse.
    // (We don't actually decode a 16k×16k image in this test — too
    // slow; the encoder produces a 16x16 JPEG and we patch the
    // declared dims so Decoder::new sees 16384×16384 but the entropy
    // data is for a 16x16 image. `Decoder::new` only validates the
    // header, so it should succeed here even though `.decode()`
    // would later fail on the truncated stream.)
    let mut jpeg = encode_minimal_jpeg();
    patch_sof_dims(&mut jpeg, 16384, 16384);
    if Decoder::new(&jpeg).is_err() {
        panic!("16384 is at the cap; header parse should succeed");
    }
}

/// Round-trip through a stream that carries restart markers: encoder
/// emits `RSTn` every N MCUs, decoder follows them via the RST
/// handling path. Asserts both the high-level pixel fidelity *and*
/// that the encoded stream actually contains restart markers (so the
/// next encoder refactor can't silently drop them).
fn run_restart_roundtrip(w: u32, h: u32, interval: u16, sub: ChromaSubsampling, min_psnr: f64) {
    let rgb = gradient_rgb(w, h);
    let mut jpeg = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut jpeg, 80);
    enc.set_subsampling(sub);
    enc.set_restart_interval(interval);
    enc.encode_rgb(&rgb, w, h).expect("encode");

    // Spot-check DRI + at least one RSTn in the byte stream.
    let has_dri = jpeg.windows(2).any(|p| p == [0xFF, 0xDD]);
    let has_rst = jpeg
        .windows(2)
        .any(|p| matches!(p, [0xFF, b] if (0xD0..=0xD7).contains(b)));
    assert!(has_dri, "encoded stream missing DRI segment");
    assert!(has_rst, "encoded stream missing RSTn marker");

    let decoded = Decoder::new(&jpeg)
        .unwrap()
        .decode(PixelFormat::Rgb)
        .unwrap();
    let psnr = psnr_rgb(&rgb, &decoded);
    assert!(
        psnr >= min_psnr,
        "restart roundtrip PSNR {psnr:.2} dB below floor {min_psnr:.2} dB ({w}x{h} ri={interval} {sub:?})",
    );
}

#[test]
fn restart_roundtrip_64x64_420_ri4() {
    run_restart_roundtrip(64, 64, 4, ChromaSubsampling::Yuv420, 30.0);
}

#[test]
fn restart_roundtrip_320x240_422_ri16() {
    run_restart_roundtrip(320, 240, 16, ChromaSubsampling::Yuv422, 32.0);
}

#[test]
fn restart_roundtrip_1080p_420_ri120() {
    run_restart_roundtrip(1920, 1080, 120, ChromaSubsampling::Yuv420, 32.0);
}

/// Verify the encoder honors a caller-supplied quantization table:
/// build a max-quality (all-1) table, encode → decode, and assert the
/// reconstruction is exact (PSNR = infinity for a self-roundtrip when
/// quantization is lossless).
#[test]
fn custom_quant_tables_lossless_q1() {
    let w = 32;
    let h = 32;
    let rgb = gradient_rgb(w, h);
    let mut jpeg = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut jpeg, 80);
    enc.set_subsampling(ChromaSubsampling::Yuv444);
    enc.set_quant_tables([1u8; 64], [1u8; 64]);
    enc.encode_rgb(&rgb, w, h).expect("encode");
    let decoded = Decoder::new(&jpeg)
        .unwrap()
        .decode(PixelFormat::Rgb)
        .unwrap();
    let psnr = psnr_rgb(&rgb, &decoded);
    // With qi=1 quantization is effectively lossless on the DCT path;
    // residual error comes only from color conversion rounding (~1 LSB).
    assert!(
        psnr >= 45.0,
        "custom-quant (all-1) PSNR {psnr:.2} dB below expected ~lossless floor",
    );
}

/// Verify `clear_quant_tables` restores the quality-scaled default.
#[test]
fn custom_quant_clear_falls_back_to_quality() {
    let w = 32;
    let h = 32;
    let rgb = gradient_rgb(w, h);

    let mut jpeg_custom = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut jpeg_custom, 80);
    enc.set_quant_tables([1u8; 64], [1u8; 64]);
    enc.clear_quant_tables();
    enc.encode_rgb(&rgb, w, h).unwrap();

    let mut jpeg_default = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg_default, 80)
        .encode_rgb(&rgb, w, h)
        .unwrap();

    assert_eq!(
        jpeg_custom, jpeg_default,
        "clear should reproduce default output"
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
