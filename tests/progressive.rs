//! Progressive (SOF2) encode → decode round-trip tests.
//!
//! Exercises the four-scan spectral plan (DC interleaved + per-
//! component AC) in `progressive_encode`. We verify three things:
//!
//! 1. The output JPEG carries the SOF2 marker (= valid progressive
//!    framing).
//! 2. Multiple SOS segments appear (one per scan), in the expected
//!    order.
//! 3. The image decodes successfully via our own progressive decoder
//!    AND via the `image` crate, with pixel-level reconstruction
//!    close enough to the encoded source (JPEG is lossy, so we use a
//!    PSNR floor rather than exact-byte equality).

use jpeg_rusturbo::decode;
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

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sse: u64 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x as i32 - y as i32;
        sse += (d * d) as u64;
    }
    let mse = sse as f64 / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    20.0 * (255.0_f64 / mse.sqrt()).log10()
}

/// Walk top-level JPEG segments until SOS (inclusive). Returns each
/// `(marker_byte, payload_bytes)` pair so the test can inspect SOF /
/// SOS / DHT framing.
fn collect_marker_segments(jpeg: &[u8]) -> Vec<(u8, Vec<u8>)> {
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "missing SOI");
    let mut i = 2;
    let mut out = Vec::new();
    while i + 4 <= jpeg.len() {
        assert_eq!(jpeg[i], 0xFF, "expected marker prefix at offset {i}");
        let marker = jpeg[i + 1];
        i += 2;
        let len = u16::from_be_bytes([jpeg[i], jpeg[i + 1]]) as usize;
        let payload = jpeg[i + 2..i + len].to_vec();
        out.push((marker, payload));
        i += len;
        if marker == 0xDA {
            // SOS — stop here; entropy data follows.
            break;
        }
    }
    out
}

/// As above, but also count subsequent SOS occurrences in the
/// remainder of the stream. Each SOS is preceded by an entropy
/// segment that contains no marker bytes (0xFF stuffing handles the
/// only collision), so a simple `[0xFF, 0xDA]` scan over the tail
/// gives the scan count for a progressive JPEG.
fn count_sos(jpeg: &[u8]) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i + 1 < jpeg.len() {
        if jpeg[i] == 0xFF && jpeg[i + 1] == 0xDA {
            count += 1;
            i += 2;
        } else {
            i += 1;
        }
    }
    count
}

#[test]
fn progressive_emits_sof2_and_four_scans() {
    let pixels = gradient_rgb(64, 64);
    let mut out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.set_progressive(true);
        enc.encode_rgb(&pixels, 64, 64).unwrap();
    }
    let segs = collect_marker_segments(&out);
    let sof2_count = segs.iter().filter(|(m, _)| *m == 0xC2).count();
    let sof0_count = segs.iter().filter(|(m, _)| *m == 0xC0).count();
    assert_eq!(sof2_count, 1, "expected one SOF2");
    assert_eq!(sof0_count, 0, "SOF0 should be absent in progressive output");
    assert_eq!(count_sos(&out), 4, "expected 4 SOS segments (DC + AC × 3)");
}

#[test]
fn progressive_decodes_via_self_decoder() {
    let pixels = gradient_rgb(128, 96);
    let mut out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.set_progressive(true);
        enc.encode_rgb(&pixels, 128, 96).unwrap();
    }
    let decoded = decode::decode(&out, PixelFormat::Rgb).expect("self-decoder rejected progressive output");
    assert_eq!(decoded.len(), pixels.len());
    let p = psnr(&pixels, &decoded);
    assert!(p > 33.0, "progressive roundtrip PSNR too low: {p:.2}");
}

#[test]
fn progressive_decodes_via_image_crate() {
    let pixels = gradient_rgb(128, 96);
    let mut out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.set_progressive(true);
        enc.encode_rgb(&pixels, 128, 96).unwrap();
    }
    let img = image::ImageReader::with_format(std::io::Cursor::new(&out), image::ImageFormat::Jpeg)
        .decode()
        .expect("image crate rejected progressive output");
    assert_eq!(img.width(), 128);
    assert_eq!(img.height(), 96);
    let p = psnr(&pixels, img.to_rgb8().as_raw());
    assert!(p > 33.0, "image-crate roundtrip PSNR too low: {p:.2}");
}

#[test]
fn progressive_roundtrip_4_2_0_natural_size() {
    // Larger image at 4:2:0 — exercises the per-component AC scan
    // walk with many MCUs (= EOB-run accumulation across blocks).
    let pixels = gradient_rgb(256, 192);
    let mut out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut out, 75);
        enc.set_subsampling(ChromaSubsampling::Yuv420);
        enc.set_progressive(true);
        enc.encode_rgb(&pixels, 256, 192).unwrap();
    }
    let decoded = decode::decode(&out, PixelFormat::Rgb).unwrap();
    let p = psnr(&pixels, &decoded);
    assert!(p > 32.0, "4:2:0 progressive PSNR too low: {p:.2}");
}

#[test]
fn progressive_preserves_exif_icc_passthrough() {
    // Progressive + metadata should still emit APP1 / APP2 ahead of
    // SOF2 (same placement as the baseline path).
    let pixels = gradient_rgb(48, 32);
    let exif: Vec<u8> = b"II\x2A\x00\x08\x00\x00\x00\x00\x00\x00\x00".to_vec();
    let icc: Vec<u8> = (0..512u32).map(|i| (i & 0xFF) as u8).collect();
    let mut out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.set_progressive(true);
        enc.set_exif(Some(exif));
        enc.set_icc_profile(Some(icc));
        enc.encode_rgb(&pixels, 48, 32).unwrap();
    }
    let segs = collect_marker_segments(&out);
    assert!(segs.iter().any(|(m, _)| *m == 0xE1), "APP1 missing");
    assert!(segs.iter().any(|(m, _)| *m == 0xE2), "APP2 missing");
    assert!(segs.iter().any(|(m, _)| *m == 0xC2), "SOF2 missing");
}

#[test]
fn baseline_unchanged_when_progressive_off() {
    // Sanity: not setting `progressive` still emits the SOF0 baseline
    // stream. The byte-identical-output guarantee we maintain for
    // existing callers depends on this.
    let pixels = gradient_rgb(32, 32);
    let mut out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.encode_rgb(&pixels, 32, 32).unwrap();
    }
    let segs = collect_marker_segments(&out);
    assert!(segs.iter().any(|(m, _)| *m == 0xC0), "expected SOF0 in baseline output");
    assert!(!segs.iter().any(|(m, _)| *m == 0xC2), "SOF2 leaked into baseline output");
    assert_eq!(count_sos(&out), 1, "baseline should emit exactly one SOS");
}
