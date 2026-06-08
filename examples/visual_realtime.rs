//! Real-time encode/decode visualizer. Generates an animated frame,
//! encodes with `jpeg-rusturbo` at q=80 4:2:0, decodes with
//! `jpeg-rusturbo`, and displays the decoded buffer in a minifb
//! window. Title bar reports the encode + decode time per frame
//! plus the active encoder mode (baseline / progressive / metadata
//! pass-through).
//!
//! Keys:
//!   - `ESC`: quit
//!   - `P`: toggle progressive (SOF2) ↔ baseline (SOF0)
//!   - `M`: toggle EXIF + ICC pass-through (synthetic small blobs)
//!   - `S`: cycle subsampling 4:2:0 → 4:2:2 → 4:4:4
//!   - `[` / `]`: decrease / increase JPEG quality by 5
//!   - `<` / `>`: cycle FPS cap (uncapped / 30 / 60 / 120 / 250)
//!
//! All toggles affect the *encoder* only — the decode path is
//! identical, so a visual divergence between modes would indicate
//! an encoder bug.
//!
//! Run: `cargo run --release --example visual_realtime`

use jpeg_rusturbo::{ChromaSubsampling, JpegEncoder, PixelFormat, decode};
use minifb::{Key, Window, WindowOptions};
use std::time::Instant;

const W: usize = 1280;
const H: usize = 720;
const FPS_CAPS: [Option<usize>; 5] = [None, Some(30), Some(60), Some(120), Some(250)];

fn main() {
    let mut window = Window::new(
        "jpeg-rusturbo: encode → decode roundtrip (ESC to quit)",
        W,
        H,
        WindowOptions::default(),
    )
    .unwrap();
    let mut fps_cap_idx = 2usize;
    apply_fps_cap(&mut window, FPS_CAPS[fps_cap_idx]);

    let mut rgb = vec![0u8; W * H * 3];
    let mut buf = vec![0u32; W * H];
    let mut jpeg = Vec::with_capacity(W * H);
    let mut frame: u32 = 0;
    let mut last_report = Instant::now();
    let mut t_enc_sum = 0.0;
    let mut t_dec_sum = 0.0;
    let mut count = 0;

    // Mode toggles. The decoded buffer is bit-identical in shape
    // regardless of mode (= same width / height / pixel format), so
    // any visual artifact while toggling between modes points at an
    // encoder bug rather than a UI sync issue.
    let mut progressive = false;
    let mut with_metadata = false;
    let mut subsampling = ChromaSubsampling::Yuv420;
    let mut quality = 80u8;
    let mut prev_p = false;
    let mut prev_m = false;
    let mut prev_s = false;
    let mut prev_lbracket = false;
    let mut prev_rbracket = false;
    let mut prev_comma = false;
    let mut prev_period = false;
    // Small synthetic EXIF + ICC blobs; the encoder routes them
    // through as APP1 / APP2 segments without parsing.
    let exif: Vec<u8> = b"II\x2A\x00\x08\x00\x00\x00\x00\x00\x00\x00".to_vec();
    let icc: Vec<u8> = (0..512u32).map(|i| (i & 0xFF) as u8).collect();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Edge-triggered toggles so a held key doesn't flip every
        // frame.
        let p_now = window.is_key_down(Key::P);
        if p_now && !prev_p {
            progressive = !progressive;
        }
        prev_p = p_now;
        let m_now = window.is_key_down(Key::M);
        if m_now && !prev_m {
            with_metadata = !with_metadata;
        }
        prev_m = m_now;
        let s_now = window.is_key_down(Key::S);
        if s_now && !prev_s {
            subsampling = next_subsampling(subsampling);
        }
        prev_s = s_now;
        let lbracket_now = window.is_key_down(Key::LeftBracket);
        if lbracket_now && !prev_lbracket {
            quality = quality.saturating_sub(5).max(1);
        }
        prev_lbracket = lbracket_now;
        let rbracket_now = window.is_key_down(Key::RightBracket);
        if rbracket_now && !prev_rbracket {
            quality = quality.saturating_add(5).min(100);
        }
        prev_rbracket = rbracket_now;
        let comma_now = window.is_key_down(Key::Comma);
        if comma_now && !prev_comma {
            fps_cap_idx = if fps_cap_idx == 0 {
                FPS_CAPS.len() - 1
            } else {
                fps_cap_idx - 1
            };
            apply_fps_cap(&mut window, FPS_CAPS[fps_cap_idx]);
        }
        prev_comma = comma_now;
        let period_now = window.is_key_down(Key::Period);
        if period_now && !prev_period {
            fps_cap_idx = (fps_cap_idx + 1) % FPS_CAPS.len();
            apply_fps_cap(&mut window, FPS_CAPS[fps_cap_idx]);
        }
        prev_period = period_now;

        let phase = frame as f32 * 0.04;
        render(&mut rgb, W as u32, H as u32, phase);

        // ---- Encode (q=80 4:2:0, mode-dependent) ----
        let t0 = Instant::now();
        jpeg.clear();
        let mut enc = JpegEncoder::new_with_quality(&mut jpeg, quality);
        enc.set_subsampling(subsampling);
        enc.set_progressive(progressive);
        if with_metadata {
            enc.set_exif(Some(exif.clone()));
            enc.set_icc_profile(Some(icc.clone()));
        }
        enc.encode_rgb(&rgb, W as u32, H as u32).unwrap();
        let t_enc = t0.elapsed().as_secs_f64() * 1000.0;

        // ---- Decode ----
        let t1 = Instant::now();
        let decoded = decode::decode(&jpeg, PixelFormat::Rgb).unwrap();
        let t_dec = t1.elapsed().as_secs_f64() * 1000.0;

        for (i, chunk) in decoded.chunks_exact(3).enumerate() {
            buf[i] = ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | chunk[2] as u32;
        }

        t_enc_sum += t_enc;
        t_dec_sum += t_dec;
        count += 1;
        if last_report.elapsed().as_secs_f64() > 0.5 {
            let n = count as f64;
            let mode = match (progressive, with_metadata) {
                (false, false) => "baseline",
                (true, false) => "progressive",
                (false, true) => "baseline+meta",
                (true, true) => "progressive+meta",
            };
            window.set_title(&format!(
                "jpeg-rusturbo [{mode} {} q={quality}] (P/M/S/[ ]/<>): enc {:.2} ms · dec {:.2} ms · JPEG {} KB ({} fps / {})",
                subsampling_label(subsampling),
                t_enc_sum / n,
                t_dec_sum / n,
                jpeg.len() / 1024,
                (n / last_report.elapsed().as_secs_f64()) as u32,
                fps_cap_label(FPS_CAPS[fps_cap_idx]),
            ));
            t_enc_sum = 0.0;
            t_dec_sum = 0.0;
            count = 0;
            last_report = Instant::now();
        }
        window.update_with_buffer(&buf, W, H).unwrap();
        frame = frame.wrapping_add(1);
    }
}

fn apply_fps_cap(window: &mut Window, cap: Option<usize>) {
    window.set_target_fps(cap.unwrap_or(0));
}

fn fps_cap_label(cap: Option<usize>) -> &'static str {
    match cap {
        None => "uncapped",
        Some(30) => "cap 30 fps",
        Some(60) => "cap 60 fps",
        Some(120) => "cap 120 fps",
        Some(250) => "cap 250 fps",
        Some(_) => "cap custom fps",
    }
}

fn next_subsampling(subsampling: ChromaSubsampling) -> ChromaSubsampling {
    match subsampling {
        ChromaSubsampling::Yuv420 => ChromaSubsampling::Yuv422,
        ChromaSubsampling::Yuv422 => ChromaSubsampling::Yuv444,
        ChromaSubsampling::Yuv444 => ChromaSubsampling::Yuv420,
    }
}

fn subsampling_label(subsampling: ChromaSubsampling) -> &'static str {
    match subsampling {
        ChromaSubsampling::Yuv420 => "4:2:0",
        ChromaSubsampling::Yuv422 => "4:2:2",
        ChromaSubsampling::Yuv444 => "4:4:4",
    }
}

/// Animated procedural scene: gradient sky, color bars, an orbiting
/// ring, and a slowly-shifting hue checkerboard patch.
fn render(out: &mut [u8], w: u32, h: u32, phase: f32) {
    let cx = w as f32 / 2.0 + (phase.cos() * (w as f32) * 0.2);
    let cy = h as f32 / 2.0 + (phase.sin() * (h as f32) * 0.15);
    let ring_r = (w.min(h) as f32) * 0.18;

    for y in 0..h {
        for x in 0..w {
            let fy = y as f32 / h as f32;

            let mut r = 60.0 + (1.0 - fy) * 80.0;
            let mut g = 110.0 + (1.0 - fy) * 80.0;
            let mut b = 200.0 + (1.0 - fy) * 50.0;

            if fy > 0.66 {
                let bar = (x * 6 / w) % 6;
                let (br, bg, bb) = match bar {
                    0 => (255, 255, 255),
                    1 => (255, 255, 0),
                    2 => (0, 255, 255),
                    3 => (0, 255, 0),
                    4 => (255, 0, 255),
                    _ => (255, 0, 0),
                };
                r = br as f32;
                g = bg as f32;
                b = bb as f32;
            }

            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if (dist - ring_r).abs() < 4.0 {
                r = 240.0;
                g = 240.0;
                b = 240.0;
            }

            let patch_x0 = (w as i32) * 65 / 100;
            let patch_y0 = (h as i32) * 12 / 100;
            let patch_w = (w as i32) * 20 / 100;
            let patch_h = (h as i32) * 25 / 100;
            let xi = x as i32;
            let yi = y as i32;
            if xi >= patch_x0
                && xi < patch_x0 + patch_w
                && yi >= patch_y0
                && yi < patch_y0 + patch_h
            {
                let cell = ((xi - patch_x0) / 8 + (yi - patch_y0) / 8 + (phase * 4.0) as i32) & 1;
                if cell == 0 {
                    r = 20.0;
                    g = 20.0;
                    b = 20.0;
                } else {
                    r = 235.0;
                    g = 235.0;
                    b = 235.0;
                }
            }

            let i = (y as usize * w as usize + x as usize) * 3;
            out[i] = r.clamp(0.0, 255.0) as u8;
            out[i + 1] = g.clamp(0.0, 255.0) as u8;
            out[i + 2] = b.clamp(0.0, 255.0) as u8;
        }
    }
}
