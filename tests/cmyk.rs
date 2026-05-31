//! CMYK (4-component) encode + decode integration tests.
//!
//! Covers the new `JpegEncoder::encode_cmyk` entry point and the
//! `PixelFormat::Cmyk` decode-output path. Mirrors the structure of
//! `tests/grayscale.rs` — gradient inputs, roundtrip tolerance,
//! optimize-Huffman composition, configuration-knob behaviour.

use image::{ImageFormat, ImageReader};
use jpeg_rusturbo::decode::{DecodeError, Decoder};
use jpeg_rusturbo::{JpegEncoder, PixelFormat};
use std::io::Cursor;

/// Smoothly-varying 4-channel CMYK image. Each channel walks its own
/// independent gradient so per-channel quantization round-trip is
/// exercised on a non-degenerate signal (a uniform single-colour ink
/// would compress to near-zero AC and understate the test).
fn gradient_cmyk(w: u32, h: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let c = (((x) * 255) / w.max(1)) as u8;
            let m = (((y) * 255) / h.max(1)) as u8;
            let yk = (((x + y) * 255) / (w + h).max(1)) as u8;
            // Linear ramp on K too — high-frequency K (e.g. xy mod
            // 256) is dominated by quant noise even at q=95.
            let k = ((((w - x) + (h - y)) * 255) / (w + h).max(1)) as u8;
            buf.extend_from_slice(&[c, m, yk, k]);
        }
    }
    buf
}

fn max_abs_diff(a: &[u8], b: &[u8]) -> i32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| ((*x as i32) - (*y as i32)).abs())
        .max()
        .unwrap_or(0)
}

fn encode_cmyk(pixels: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    let mut jpeg = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut jpeg, quality);
    enc.encode_cmyk(pixels, w, h).expect("encode_cmyk");
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "SOI");
    assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "EOI");
    jpeg
}

fn run_cmyk_roundtrip(w: u32, h: u32, quality: u8, max_diff: i32) {
    let cmyk = gradient_cmyk(w, h);
    let jpeg = encode_cmyk(&cmyk, w, h, quality);

    let dec = Decoder::new(&jpeg).expect("our decoder headers");
    let info = dec.info();
    assert_eq!(info.width, w);
    assert_eq!(info.height, h);
    assert_eq!(info.components, 4, "expected 4-component CMYK JPEG");

    let decoded = dec.decode(PixelFormat::Cmyk).expect("our decode → Cmyk");
    assert_eq!(decoded.len(), (w * h * 4) as usize);
    let d = max_abs_diff(&cmyk, &decoded);
    assert!(
        d <= max_diff,
        "cmyk roundtrip max diff {} > {} (q={}, {}x{})",
        d,
        max_diff,
        quality,
        w,
        h
    );
}

#[test]
fn cmyk_roundtrip_8x8() {
    run_cmyk_roundtrip(8, 8, 80, 10);
}

#[test]
fn cmyk_roundtrip_17x17_unaligned() {
    run_cmyk_roundtrip(17, 17, 80, 12);
}

#[test]
fn cmyk_roundtrip_1592x1124() {
    run_cmyk_roundtrip(1592, 1124, 80, 12);
}

#[test]
fn cmyk_roundtrip_high_quality() {
    run_cmyk_roundtrip(64, 64, 95, 6);
}

/// CMYK source decoded as RGB (or any non-CMYK PixelFormat) must
/// return `Unsupported` — the crate does not perform CMYK→RGB.
#[test]
fn cmyk_source_decoded_as_non_cmyk_unsupported() {
    let (w, h) = (16, 16);
    let cmyk = gradient_cmyk(w, h);
    let jpeg = encode_cmyk(&cmyk, w, h, 80);

    for fmt in [
        PixelFormat::Rgb,
        PixelFormat::Bgr,
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Gray,
    ] {
        let err = jpeg_rusturbo::decode::decode(&jpeg, fmt).unwrap_err();
        assert!(
            matches!(err, DecodeError::Unsupported(_)),
            "expected Unsupported for {:?}, got {:?}",
            fmt,
            err,
        );
    }
}

/// 3-component (YCbCr) source decoded into `PixelFormat::Cmyk` must
/// likewise return `Unsupported` — pass-through only.
#[test]
fn ycbcr_source_decoded_as_cmyk_unsupported() {
    let (w, h) = (16, 16);
    let rgb = vec![100u8; (w * h * 3) as usize];
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, 80)
        .encode_rgb(&rgb, w, h)
        .expect("encode_rgb");
    let err = jpeg_rusturbo::decode::decode(&jpeg, PixelFormat::Cmyk).unwrap_err();
    assert!(matches!(err, DecodeError::Unsupported(_)));
}

/// `set_optimize_huffman(true) + encode_cmyk(...)` must produce a
/// strictly smaller output AND decode to the same pixels as the
/// standard-tables path.
#[test]
fn optimize_huffman_composes() {
    let (w, h) = (256, 192);
    let cmyk = gradient_cmyk(w, h);

    let std_jpeg = encode_cmyk(&cmyk, w, h, 80);

    let mut opt_jpeg = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut opt_jpeg, 80);
        enc.set_optimize_huffman(true);
        enc.encode_cmyk(&cmyk, w, h).expect("encode_cmyk optimize");
    }
    assert!(
        opt_jpeg.len() < std_jpeg.len(),
        "optimize-Huffman cmyk ({} B) should be smaller than standard ({} B)",
        opt_jpeg.len(),
        std_jpeg.len()
    );

    let dec_std = jpeg_rusturbo::decode::decode(&std_jpeg, PixelFormat::Cmyk).unwrap();
    let dec_opt = jpeg_rusturbo::decode::decode(&opt_jpeg, PixelFormat::Cmyk).unwrap();
    assert_eq!(
        dec_std, dec_opt,
        "optimized vs standard Huffman should decode to identical pixels"
    );
}

/// Progressive CMYK is explicitly unsupported.
#[test]
fn progressive_cmyk_rejected() {
    let mut out = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
    enc.set_progressive(true);
    let err = enc
        .encode_cmyk(&vec![128u8; 16 * 16 * 4], 16, 16)
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
}

/// `set_threads(n)` on CMYK must be a no-op (serial-only path) — the
/// emitted bytes are identical regardless of thread count.
#[test]
fn threads_setter_is_noop_for_cmyk() {
    let (w, h) = (80, 48);
    let cmyk = gradient_cmyk(w, h);

    let mut serial = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut serial, 80);
        enc.set_threads(1);
        enc.encode_cmyk(&cmyk, w, h).unwrap();
    }
    let mut threaded = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut threaded, 80);
        enc.set_threads(4);
        enc.encode_cmyk(&cmyk, w, h).unwrap();
    }
    assert_eq!(serial, threaded);
}

/// `set_subsampling(...)` on CMYK must be a no-op too (no chroma to
/// subsample). Output bytes are identical across the three modes.
#[test]
fn subsampling_setter_is_noop_for_cmyk() {
    let (w, h) = (40, 24);
    let cmyk = gradient_cmyk(w, h);

    let mut a = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut a, 80);
        enc.set_subsampling(jpeg_rusturbo::ChromaSubsampling::Yuv420);
        enc.encode_cmyk(&cmyk, w, h).unwrap();
    }
    let mut b = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut b, 80);
        enc.set_subsampling(jpeg_rusturbo::ChromaSubsampling::Yuv444);
        enc.encode_cmyk(&cmyk, w, h).unwrap();
    }
    assert_eq!(a, b);
}

/// `encode` (generic) with `PixelFormat::Cmyk` must produce
/// byte-identical output to `encode_cmyk`.
#[test]
fn generic_encode_cmyk_matches_convenience() {
    let (w, h) = (40, 24);
    let cmyk = gradient_cmyk(w, h);

    let mut a = Vec::new();
    JpegEncoder::new_with_quality(&mut a, 80)
        .encode_cmyk(&cmyk, w, h)
        .unwrap();
    let mut b = Vec::new();
    JpegEncoder::new_with_quality(&mut b, 80)
        .encode(&cmyk, w, h, PixelFormat::Cmyk)
        .unwrap();
    assert_eq!(a, b);
}

/// File-size sanity: a 4-component CMYK JPEG with shared Huffman
/// tables and one DQT should be **smaller** than the same content
/// stored as four independent grayscale JPEGs (which each pay for
/// their own SOI / APP0 / DQT / SOF / DHT / SOS / EOI framing).
#[test]
fn cmyk_smaller_than_four_grayscale() {
    let (w, h) = (256, 256);
    let cmyk = gradient_cmyk(w, h);

    let one = encode_cmyk(&cmyk, w, h, 80);

    let mut four_total: usize = 0;
    for ch in 0..4 {
        let mut plane = Vec::with_capacity((w * h) as usize);
        for j in 0..h {
            for i in 0..w {
                let off = ((j * w + i) * 4 + ch as u32) as usize;
                plane.push(cmyk[off]);
            }
        }
        let mut g = Vec::new();
        JpegEncoder::new_with_quality(&mut g, 80)
            .encode_grayscale(&plane, w, h)
            .unwrap();
        four_total += g.len();
    }
    assert!(
        one.len() < four_total,
        "single CMYK JPEG ({} B) should be smaller than 4× grayscale ({} B)",
        one.len(),
        four_total
    );
}

/// `set_quant_tables(luma, chroma)` on CMYK: only the luma table is
/// consulted; the chroma table is silently ignored. Verify by
/// inspecting DQT segments — exactly one is emitted, regardless of
/// what the chroma argument carries.
#[test]
fn custom_quant_chroma_ignored() {
    let (w, h) = (40, 24);
    let cmyk = gradient_cmyk(w, h);
    let luma = [50u8; 64];

    let mut a = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut a, 80);
        enc.set_quant_tables(luma, [10u8; 64]);
        enc.encode_cmyk(&cmyk, w, h).unwrap();
    }
    let mut b = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut b, 80);
        enc.set_quant_tables(luma, [200u8; 64]);
        enc.encode_cmyk(&cmyk, w, h).unwrap();
    }
    assert_eq!(a, b, "chroma quant arg must not affect CMYK output");

    // Count DQT segments (marker 0xFF 0xDB) — must be exactly one.
    let mut dqts = 0;
    let mut i = 0;
    while i + 1 < a.len() {
        if a[i] == 0xFF && a[i + 1] == 0xDB {
            dqts += 1;
        }
        i += 1;
    }
    assert_eq!(dqts, 1, "expected exactly one DQT segment on CMYK output");
}

/// EXIF + ICC must compose on CMYK — both segments appear in the
/// output and the stream still round-trips.
#[test]
fn exif_and_icc_compose() {
    let (w, h) = (32, 32);
    let cmyk = gradient_cmyk(w, h);
    let exif: Vec<u8> = (0u8..40).collect();
    let icc: Vec<u8> = (0u8..200).collect();

    let mut out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.set_exif(Some(exif.clone()));
        enc.set_icc_profile(Some(icc.clone()));
        enc.encode_cmyk(&cmyk, w, h).unwrap();
    }

    // APP1 (0xFF 0xE1) and APP2 (0xFF 0xE2) present.
    let has_app1 = out.windows(2).any(|w| w == [0xFF, 0xE1]);
    let has_app2 = out.windows(2).any(|w| w == [0xFF, 0xE2]);
    assert!(has_app1, "APP1 / EXIF segment missing");
    assert!(has_app2, "APP2 / ICC segment missing");

    // Still decodes.
    let decoded = jpeg_rusturbo::decode::decode(&out, PixelFormat::Cmyk).unwrap();
    assert_eq!(decoded.len(), (w * h * 4) as usize);
}

/// APP14 (Adobe marker) on a CMYK stream: this crate intentionally
/// does NOT apply the YCCK transform regardless of the APP14 transform
/// byte. Synthesize a CMYK JPEG, inject an APP14 segment after the
/// JFIF APP0, and confirm the decoder produces the same plain-CMYK
/// output it would without APP14.
#[test]
fn app14_adobe_marker_is_consumed_and_ignored() {
    let (w, h) = (16, 16);
    let cmyk = gradient_cmyk(w, h);
    let plain = encode_cmyk(&cmyk, w, h, 80);

    // Inject APP14 ("Adobe\0", version 100, flags0/1, transform=2 =
    // "YCCK") right after the JFIF APP0 segment. The JFIF APP0 lives
    // at bytes 2..20 (SOI + APP0 marker + 16-byte length + payload).
    // Locate it by searching for the JFIF identifier.
    let jfif_pos = plain
        .windows(5)
        .position(|w| w == b"JFIF\0")
        .expect("JFIF identifier present");
    // The APP0 segment starts 4 bytes before the identifier (FF E0
    // + 2-byte length); its length field is at jfif_pos - 2. Read
    // it to find where APP0 ends.
    let app0_len = ((plain[jfif_pos - 2] as usize) << 8) | (plain[jfif_pos - 1] as usize);
    let app0_end = (jfif_pos - 2) + app0_len;

    // APP14 segment: marker (2) + length (2) + "Adobe" (5) + version
    // (2) + flags0 (2) + flags1 (2) + transform (1) = 16 bytes.
    let app14: [u8; 16] = [
        0xFF, 0xEE, // APP14 marker
        0x00, 0x0E, // length = 14 (excludes marker; includes itself)
        b'A', b'd', b'o', b'b', b'e', // identifier
        0x00, 0x64, // version 100
        0x00, 0x00, // flags0
        0x00, 0x00, // flags1
        0x02, // transform = 2 (YCCK) — intentionally ignored
    ];

    let mut spliced = Vec::with_capacity(plain.len() + app14.len());
    spliced.extend_from_slice(&plain[..app0_end]);
    spliced.extend_from_slice(&app14);
    spliced.extend_from_slice(&plain[app0_end..]);

    let decoded_plain = jpeg_rusturbo::decode::decode(&plain, PixelFormat::Cmyk).unwrap();
    let decoded_spliced = jpeg_rusturbo::decode::decode(&spliced, PixelFormat::Cmyk).unwrap();
    assert_eq!(
        decoded_plain, decoded_spliced,
        "APP14 must be consumed-and-ignored — output identical to plain CMYK",
    );
}

/// Cross-decoder smoke test: feed our CMYK JPEG to the `image` crate
/// and verify it can parse the header (or fail gracefully). We do not
/// assert per-pixel equality because the `image` crate's CMYK support
/// has historically been spotty — this test documents the current
/// state and locks in the behaviour we observe today.
#[test]
fn cross_decoder_image_crate_smoke() {
    let (w, h) = (64, 64);
    let cmyk = gradient_cmyk(w, h);
    let jpeg = encode_cmyk(&cmyk, w, h, 85);

    let result = ImageReader::with_format(Cursor::new(&jpeg), ImageFormat::Jpeg).decode();
    // Either:
    //   - `image` v0.25 decodes the CMYK stream into some
    //     representation (often RGB after applying its own
    //     CMYK→sRGB conversion) — we just confirm dimensions.
    //   - `image` returns an error because CMYK support is gated
    //     behind a feature flag. Both outcomes are acceptable;
    //     this test pins down today's behaviour.
    match result {
        Ok(img) => {
            assert_eq!(img.width(), w);
            assert_eq!(img.height(), h);
        }
        Err(e) => {
            eprintln!(
                "note: `image` crate did not decode plain CMYK JPEG: {e}; \
                 this is acceptable — CMYK→RGB is out of scope for this crate"
            );
        }
    }
}
