//! Real-time encode/decode visualizer. Generates an animated frame,
//! encodes with `jpeg-rusturbo` at q=80 4:2:0, decodes with
//! `jpeg-rusturbo`, and displays the decoded buffer in a minifb
//! window. Title bar reports the encode + decode time per frame.
//!
//! Run: `cargo run --release --example visual_realtime`

use jpeg_rusturbo::{ChromaSubsampling, JpegEncoder, PixelFormat, decode};
use minifb::{Key, Window, WindowOptions};
use std::time::Instant;

const W: usize = 1280;
const H: usize = 720;

fn main() {
    let mut window = Window::new(
        "jpeg-rusturbo: encode → decode roundtrip (ESC to quit)",
        W,
        H,
        WindowOptions::default(),
    )
    .unwrap();
    window.set_target_fps(60);

    let mut rgb = vec![0u8; W * H * 3];
    let mut buf = vec![0u32; W * H];
    let mut jpeg = Vec::with_capacity(W * H);
    let mut frame: u32 = 0;
    let mut last_report = Instant::now();
    let mut t_enc_sum = 0.0;
    let mut t_dec_sum = 0.0;
    let mut count = 0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let phase = frame as f32 * 0.04;
        render(&mut rgb, W as u32, H as u32, phase);

        // ---- Encode (q=80 4:2:0) ----
        let t0 = Instant::now();
        jpeg.clear();
        let mut enc = JpegEncoder::new_with_quality(&mut jpeg, 80);
        enc.set_subsampling(ChromaSubsampling::Yuv420);
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
            window.set_title(&format!(
                "jpeg-rusturbo: enc {:.2} ms · dec {:.2} ms · roundtrip {:.2} ms · JPEG {} KB ({} fps cap 60)",
                t_enc_sum / n,
                t_dec_sum / n,
                (t_enc_sum + t_dec_sum) / n,
                jpeg.len() / 1024,
                (n / last_report.elapsed().as_secs_f64()) as u32,
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
