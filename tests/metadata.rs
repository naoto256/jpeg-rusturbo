//! APP1 (EXIF) / APP2 (ICC) pass-through tests.
//!
//! We don't parse the embedded payloads back — the encoder is just a
//! transport, and verifying APP1 / APP2 segment framing by byte
//! inspection is enough to assert correctness against the JPEG spec
//! (T.81 B.1.4 + Exif 2.32 + ICC.1).

use jpeg_rusturbo::JpegEncoder;

const W: u32 = 16;
const H: u32 = 16;

fn solid_rgb(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for _ in 0..(w * h) {
        out.extend_from_slice(&rgb);
    }
    out
}

/// Walk top-level JPEG segments and return `(marker_byte, segment_bytes)`
/// for every APPn / SOFn / DQT / DHT / SOS we find before the
/// entropy-coded segment starts. We stop at SOS — past that the
/// stream is bit-packed entropy data with possible byte stuffing and
/// no spec-mandated segment framing.
fn collect_marker_segments(jpeg: &[u8]) -> Vec<(u8, Vec<u8>)> {
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "missing SOI");
    let mut i = 2;
    let mut out = Vec::new();
    while i + 4 <= jpeg.len() {
        assert_eq!(jpeg[i], 0xFF, "expected marker prefix at offset {i}");
        let marker = jpeg[i + 1];
        i += 2;
        // Standalone markers (no length): SOI/EOI/RSTn — not expected
        // in the pre-SOS region of a baseline JPEG.
        let len = u16::from_be_bytes([jpeg[i], jpeg[i + 1]]) as usize;
        let payload = jpeg[i + 2..i + len].to_vec();
        out.push((marker, payload));
        i += len;
        if marker == 0xDA {
            // SOS — stop here; what follows is entropy-coded data.
            break;
        }
    }
    out
}

#[test]
fn no_metadata_emits_no_app1_app2() {
    let pixels = solid_rgb(W, H, [128, 128, 128]);
    let mut out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.encode_rgb(&pixels, W, H).unwrap();
    }
    let segs = collect_marker_segments(&out);
    let has_app1 = segs.iter().any(|(m, _)| *m == 0xE1);
    let has_app2 = segs.iter().any(|(m, _)| *m == 0xE2);
    assert!(!has_app1, "APP1 segment present without set_exif");
    assert!(!has_app2, "APP2 segment present without set_icc_profile");
}

#[test]
fn set_exif_emits_one_app1_with_identifier() {
    // Minimal Exif TIFF header: byte order + magic + IFD0 offset +
    // empty IFD0. The encoder doesn't introspect the payload — any
    // byte string passes through verbatim.
    let exif: Vec<u8> = b"II\x2A\x00\x08\x00\x00\x00\x00\x00\x00\x00".to_vec();
    let pixels = solid_rgb(W, H, [200, 100, 50]);
    let mut out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.set_exif(Some(exif.clone()));
        enc.encode_rgb(&pixels, W, H).unwrap();
    }
    let segs = collect_marker_segments(&out);
    let app1: Vec<&(u8, Vec<u8>)> = segs.iter().filter(|(m, _)| *m == 0xE1).collect();
    assert_eq!(app1.len(), 1, "expected exactly one APP1 segment");
    let payload = &app1[0].1;
    assert!(
        payload.starts_with(b"Exif\0\0"),
        "APP1 payload missing the Exif\\0\\0 identifier"
    );
    assert_eq!(&payload[6..], &exif[..], "APP1 payload doesn't match input");
}

#[test]
fn set_icc_small_emits_one_app2_with_identifier() {
    // Pretend ICC profile (just bytes; the encoder doesn't validate
    // the ICC structure).
    let icc: Vec<u8> = (0..1024u32).map(|i| (i & 0xFF) as u8).collect();
    let pixels = solid_rgb(W, H, [10, 200, 60]);
    let mut out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.set_icc_profile(Some(icc.clone()));
        enc.encode_rgb(&pixels, W, H).unwrap();
    }
    let segs = collect_marker_segments(&out);
    let app2: Vec<&(u8, Vec<u8>)> = segs.iter().filter(|(m, _)| *m == 0xE2).collect();
    assert_eq!(app2.len(), 1, "expected exactly one APP2 segment");
    let payload = &app2[0].1;
    assert!(
        payload.starts_with(b"ICC_PROFILE\0"),
        "APP2 payload missing the ICC_PROFILE\\0 identifier"
    );
    // After identifier comes (seq=1, total=1) then the profile bytes.
    assert_eq!(payload[12], 1, "seq number");
    assert_eq!(payload[13], 1, "total segments");
    assert_eq!(&payload[14..], &icc[..], "ICC payload doesn't match input");
}

#[test]
fn set_icc_large_splits_into_multi_segment() {
    // 200 KB profile forces multi-segment chunking. The per-segment
    // payload cap is 65519 bytes (= 65533 - 12-byte ID - 2-byte seq
    // header), so 200_000 bytes needs 4 segments
    // (4 * 65519 = 262076 > 200000).
    let icc: Vec<u8> = (0..200_000u32)
        .map(|i| (i.wrapping_mul(31) & 0xFF) as u8)
        .collect();
    let pixels = solid_rgb(W, H, [20, 20, 200]);
    let mut out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.set_icc_profile(Some(icc.clone()));
        enc.encode_rgb(&pixels, W, H).unwrap();
    }
    let segs = collect_marker_segments(&out);
    let app2: Vec<&(u8, Vec<u8>)> = segs.iter().filter(|(m, _)| *m == 0xE2).collect();
    assert!(app2.len() >= 2, "expected multi-segment APP2 chain");
    let total_first = app2[0].1[13];
    assert_eq!(
        app2.len() as u8,
        total_first,
        "segment count {} doesn't match total field {}",
        app2.len(),
        total_first,
    );
    // Each segment's seq increments and total stays constant.
    for (i, (_, payload)) in app2.iter().enumerate() {
        assert!(payload.starts_with(b"ICC_PROFILE\0"));
        assert_eq!(payload[12], (i + 1) as u8, "seq mismatch at segment {i}");
        assert_eq!(payload[13], total_first, "total mismatch at segment {i}");
    }
    // Concatenated payloads (post-header) should reproduce the input.
    let mut joined = Vec::new();
    for (_, payload) in &app2 {
        joined.extend_from_slice(&payload[14..]);
    }
    assert_eq!(joined, icc, "reassembled ICC payload doesn't match input");
}

#[test]
fn metadata_does_not_break_decode() {
    // Sanity: a JPEG with EXIF + ICC is still decodable by the
    // `image` crate (= the JPEG framing is correct, not just our
    // own parser thinks so).
    let exif: Vec<u8> = b"II\x2A\x00\x08\x00\x00\x00\x00\x00\x00\x00".to_vec();
    let icc: Vec<u8> = (0..256u32).map(|i| i as u8).collect();
    let pixels = solid_rgb(64, 64, [100, 150, 200]);
    let mut out = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut out, 80);
        enc.set_exif(Some(exif));
        enc.set_icc_profile(Some(icc));
        enc.encode_rgb(&pixels, 64, 64).unwrap();
    }
    let img = image::ImageReader::with_format(std::io::Cursor::new(&out), image::ImageFormat::Jpeg)
        .decode()
        .expect("image crate should decode JPEG-with-metadata");
    assert_eq!(img.width(), 64);
    assert_eq!(img.height(), 64);
}
