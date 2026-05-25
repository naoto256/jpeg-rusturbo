//! Progressive Huffman scan decoder (SOF2).
//!
//! Progressive JPEG splits each block's 64 DCT coefficients across
//! multiple scans, each updating either a spectral band (DC alone,
//! or an AC range Ss..=Se) or refining low-order bits of values
//! already deposited by a prior scan. The decoder therefore keeps a
//! per-component coefficient buffer of the **whole** image's blocks
//! in zig-zag order, mutates it scan-by-scan, then performs dequant
//! + IDCT once all scans have arrived.
//!
//! Scan types (per T.81 G.1.1, derived from SOS's Ss/Se/Ah/Al):
//!
//! | Ss      | Ah  | meaning                              |
//! |---------|-----|--------------------------------------|
//! | 0       | 0   | DC first scan (initial DC, << Al)    |
//! | 0       | > 0 | DC refinement (one bit at position Al)|
//! | 1..=63  | 0   | AC first scan (band Ss..=Se, EOBRUN) |
//! | 1..=63  | > 0 | AC refinement (corrects existing,    |
//! |         |     |  inserts new ±1 << Al, EOBRUN)       |
//!
//! References: ITU-T T.81 Annex G, libjpeg-turbo `jdphuff.c` /
//! `jdarith.c` (for the dequant/IDCT contract).

use crate::arch;
use crate::tables::ZIGZAG;

use super::baseline::{DecodedPlane, DecodedPlanes};
use super::error::{DecodeError, Result};
use super::huffman::{BitReader, HuffmanDecodeTable, extend};
use super::markers::{
    Component, DecoderHeaders, FrameHeader, HuffmanTableSpec, MarkerReader, QuantTable, ScanHeader,
};

/// Per-component coefficient grid that accumulates across scans.
struct CoeffComponent {
    component: Component,
    blocks_x: usize,
    blocks_y: usize,
    /// `blocks_x * blocks_y` entries, each a 64-coefficient zig-zag
    /// block. Updated in place by each scan.
    blocks: Vec<[i16; 64]>,
}

impl CoeffComponent {
    fn block_mut(&mut self, bx: usize, by: usize) -> &mut [i16; 64] {
        let idx = by * self.blocks_x + bx;
        &mut self.blocks[idx]
    }
}

/// Top-level progressive decode: drive the scan loop, then finalize.
pub fn decode_progressive(
    src: &[u8],
    entropy_start: usize,
    headers: &DecoderHeaders,
    first_scan: ScanHeader,
) -> Result<DecodedPlanes> {
    let frame = &headers.frame;

    // ---- Allocate per-component coefficient buffer ----
    //
    // Block count per row / column is `ceil(W*Hi/(Hmax*8))` —
    // matching what the encoder emits per T.81 (A.2.4). An earlier
    // attempt used `mcus_x * Hi` which over-counts by one block per
    // row when `W mod (Hmax*8) != 0` (e.g. a 533-wide 4:2:0 image:
    // `ceil(533*2/16) = 67` vs. `ceil(533/16)*2 = 68`), causing the
    // scan loop to over-read entropy and de-sync. Single-component
    // frames are normalized to H=V=1 in `parse_sof`, so this is also
    // correct for the grayscale 900x675 H=2 V=2 case.
    let mut comps: Vec<CoeffComponent> = Vec::with_capacity(frame.components.len());
    let h_max = frame.h_max() as usize;
    let v_max = frame.v_max() as usize;
    for comp in &frame.components {
        let blocks_x = (frame.width as usize * comp.h as usize)
            .div_ceil(h_max * 8)
            .max(1);
        let blocks_y = (frame.height as usize * comp.v as usize)
            .div_ceil(v_max * 8)
            .max(1);
        let total = blocks_x
            .checked_mul(blocks_y)
            .ok_or(DecodeError::Malformed("progressive block count overflow"))?;
        comps.push(CoeffComponent {
            component: *comp,
            blocks_x,
            blocks_y,
            blocks: vec![[0i16; 64]; total],
        });
    }

    // Each scan can update DHT / DQT / DRI; clone to a working copy.
    let mut huffman_specs: Vec<HuffmanTableSpec> = headers.huffman.clone();
    let mut quant_tables: Vec<QuantTable> = headers.quant.clone();
    let mut restart_interval: u16 = headers.restart_interval;

    let mut scan: Option<ScanHeader> = Some(first_scan);
    let mut pos: usize = entropy_start;

    while let Some(s) = scan {
        let (dc_tables, ac_tables) = super::baseline::build_huffman_tables(&huffman_specs)?;
        let (after_pos, pending_marker) = run_progressive_scan(
            src,
            pos,
            frame,
            &s,
            &dc_tables,
            &ac_tables,
            restart_interval,
            &mut comps,
        )?;
        pos = after_pos;

        // The BitReader consumes 0xFF + id when it surfaces a non-RST
        // marker, so the marker prefix is gone from the stream; pass
        // the marker id forward as `pending_marker` so MarkerReader
        // can dispatch on it before walking further bytes.
        let mut mr = MarkerReader::resume_at(src, pos);
        scan = mr.next_scan_or_end(
            frame,
            &mut huffman_specs,
            &mut quant_tables,
            &mut restart_interval,
            pending_marker,
        )?;
        pos = mr.pos();
    }

    finalize_to_planes(frame, &comps, &quant_tables)
}

/// Run one progressive scan. Returns `(pos, pending_marker)` where
/// `pos` is the byte position immediately past the marker the bit
/// reader consumed when entropy data ended, and `pending_marker` is
/// the marker code if one was already pulled off the stream.
///
/// Why both: the BitReader surfaces `0xFF <id>` non-RST markers by
/// consuming both bytes and exposing `id` via `BitReader::marker()`.
/// The marker prefix is therefore gone from `src[pos..]`, so the
/// subsequent `MarkerReader` must dispatch on `pending_marker` first
/// rather than expecting `0xFF` at `pos`.
#[allow(clippy::too_many_arguments)]
fn run_progressive_scan(
    src: &[u8],
    entropy_start: usize,
    frame: &FrameHeader,
    scan: &ScanHeader,
    dc_tables: &[Option<super::baseline::DcTablePair>; 4],
    ac_tables: &[Option<super::baseline::AcTablePair>; 4],
    restart_interval: u16,
    comps: &mut [CoeffComponent],
) -> Result<(usize, Option<u8>)> {
    validate_scan(scan, frame)?;

    let mut br = BitReader::new(src, entropy_start);

    // Per-scan state. EOBRUN is per-component for non-interleaved AC
    // scans (only one component runs, but stored in an array keyed by
    // component index for code symmetry).
    let mut prev_dc = [0i32; 4];
    let mut eobrun: u32 = 0;
    let mut mcus_since_restart: u32 = 0;
    let restart_interval = restart_interval as u32;

    if scan.ss == 0 {
        // ---- DC scan: may be interleaved (Ns ≥ 1) ----
        let mcu_layout = interleaved_mcu_layout(scan, frame, comps)?;
        for my in 0..mcu_layout.mcus_y {
            for mx in 0..mcu_layout.mcus_x {
                if restart_interval > 0 && mcus_since_restart == restart_interval {
                    handle_restart(&mut br, &mut prev_dc, &mut eobrun)?;
                    mcus_since_restart = 0;
                }
                for entry in &mcu_layout.entries {
                    for v in 0..entry.v_blocks {
                        for h in 0..entry.h_blocks {
                            let bx = mx * entry.h_blocks + h;
                            let by = my * entry.v_blocks + v;
                            if bx >= comps[entry.comp_idx].blocks_x
                                || by >= comps[entry.comp_idx].blocks_y
                            {
                                // Off-image block padding: decode but
                                // discard so the bit-reader stays in
                                // sync (libjpeg-turbo does this too).
                                let mut dummy = [0i16; 64];
                                if scan.ah == 0 {
                                    decode_dc_first_block(
                                        &mut br,
                                        &dc_tables[entry.dc_table as usize]
                                            .as_ref()
                                            .ok_or(DecodeError::Malformed(
                                                "scan refers to undefined DC table",
                                            ))?
                                            .slow,
                                        &mut prev_dc[entry.comp_idx],
                                        scan.al,
                                        &mut dummy,
                                    )?;
                                } else {
                                    decode_dc_refine_block(&mut br, scan.al, &mut dummy)?;
                                }
                                continue;
                            }
                            let coef = comps[entry.comp_idx].block_mut(bx, by);
                            if scan.ah == 0 {
                                decode_dc_first_block(
                                    &mut br,
                                    &dc_tables[entry.dc_table as usize]
                                        .as_ref()
                                        .ok_or(DecodeError::Malformed(
                                            "scan refers to undefined DC table",
                                        ))?
                                        .slow,
                                    &mut prev_dc[entry.comp_idx],
                                    scan.al,
                                    coef,
                                )?;
                            } else {
                                decode_dc_refine_block(&mut br, scan.al, coef)?;
                            }
                        }
                    }
                }
                mcus_since_restart += 1;
            }
        }
    } else {
        // ---- AC scan: always non-interleaved (Ns == 1, enforced) ----
        let sc = &scan.components[0];
        let comp_idx = find_component_idx(frame, sc.component_id)?;
        // Progressive scans still go through the canonical decode_symbol
        // + get_bits path; the combined-LUT fast path is wired only for
        // baseline AC (progressive AC-refine zero-skip semantics differ).
        let ac_tbl = &ac_tables[sc.ac_table as usize]
            .as_ref()
            .ok_or(DecodeError::Malformed("scan refers to undefined AC table"))?
            .slow;
        let blocks_x = comps[comp_idx].blocks_x;
        let blocks_y = comps[comp_idx].blocks_y;
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                if restart_interval > 0 && mcus_since_restart == restart_interval {
                    handle_restart(&mut br, &mut prev_dc, &mut eobrun)?;
                    mcus_since_restart = 0;
                }
                let coef = comps[comp_idx].block_mut(bx, by);
                if scan.ah == 0 {
                    decode_ac_first_block(
                        &mut br,
                        ac_tbl,
                        scan.ss,
                        scan.se,
                        scan.al,
                        &mut eobrun,
                        coef,
                    )?;
                } else {
                    decode_ac_refine_block(
                        &mut br,
                        ac_tbl,
                        scan.ss,
                        scan.se,
                        scan.ah,
                        scan.al,
                        &mut eobrun,
                        coef,
                    )?;
                }
                mcus_since_restart += 1;
            }
        }
    }

    Ok((br.pos(), br.marker()))
}

fn validate_scan(scan: &ScanHeader, _frame: &FrameHeader) -> Result<()> {
    if scan.ss > 63 || scan.se > 63 || scan.ss > scan.se {
        return Err(DecodeError::Malformed("invalid SOS spectral selection"));
    }
    if scan.ss == 0 && scan.se != 0 {
        return Err(DecodeError::Malformed("DC scan must have Se == 0"));
    }
    if scan.ss != 0 && scan.components.len() != 1 {
        return Err(DecodeError::Malformed(
            "progressive AC scan must be non-interleaved",
        ));
    }
    if scan.al > 13 || scan.ah > 13 {
        return Err(DecodeError::Malformed("Ah/Al out of range"));
    }
    if scan.ah != 0 && scan.al >= scan.ah {
        return Err(DecodeError::Malformed("Al must be < Ah for refinement"));
    }
    Ok(())
}

/// Geometry for an interleaved scan's MCU walk.
struct McuEntry {
    comp_idx: usize,
    dc_table: u8,
    #[allow(dead_code)]
    ac_table: u8,
    h_blocks: usize,
    v_blocks: usize,
}
struct McuLayout {
    mcus_x: usize,
    mcus_y: usize,
    entries: Vec<McuEntry>,
}

fn interleaved_mcu_layout(
    scan: &ScanHeader,
    frame: &FrameHeader,
    comps: &[CoeffComponent],
) -> Result<McuLayout> {
    let h_max = frame.h_max() as usize;
    let v_max = frame.v_max() as usize;
    let mut entries = Vec::with_capacity(scan.components.len());
    for sc in &scan.components {
        let comp_idx = find_component_idx(frame, sc.component_id)?;
        let comp = comps[comp_idx].component;
        let (h_blocks, v_blocks) = if scan.components.len() == 1 {
            (1, 1) // non-interleaved
        } else {
            (comp.h as usize, comp.v as usize)
        };
        entries.push(McuEntry {
            comp_idx,
            dc_table: sc.dc_table,
            ac_table: sc.ac_table,
            h_blocks,
            v_blocks,
        });
    }
    let (mcus_x, mcus_y) = if scan.components.len() == 1 {
        let c = &comps[entries[0].comp_idx];
        (c.blocks_x, c.blocks_y)
    } else {
        let mcu_w_pixels = h_max * 8;
        let mcu_h_pixels = v_max * 8;
        (
            (frame.width as usize).div_ceil(mcu_w_pixels),
            (frame.height as usize).div_ceil(mcu_h_pixels),
        )
    };
    Ok(McuLayout {
        mcus_x,
        mcus_y,
        entries,
    })
}

fn find_component_idx(frame: &FrameHeader, id: u8) -> Result<usize> {
    frame
        .components
        .iter()
        .position(|c| c.id == id)
        .ok_or(DecodeError::Malformed("scan refers to unknown component"))
}

// ---------------- DC first scan ----------------

fn decode_dc_first_block(
    br: &mut BitReader,
    dc_tbl: &HuffmanDecodeTable,
    prev_dc: &mut i32,
    al: u8,
    coef: &mut [i16; 64],
) -> Result<()> {
    let t = br.decode_symbol(dc_tbl)?;
    let size = t as u32;
    let diff = if size == 0 {
        0
    } else {
        let bits = br.get_bits(size)?;
        extend(bits, size)
    };
    *prev_dc = prev_dc.wrapping_add(diff);
    coef[0] = ((*prev_dc) << al) as i16;
    Ok(())
}

// ---------------- DC refinement scan ----------------

fn decode_dc_refine_block(br: &mut BitReader, al: u8, coef: &mut [i16; 64]) -> Result<()> {
    let bit = br.get_bit()?;
    if bit != 0 {
        coef[0] |= 1i16 << al;
    }
    Ok(())
}

// ---------------- AC first scan ----------------

fn decode_ac_first_block(
    br: &mut BitReader,
    ac_tbl: &HuffmanDecodeTable,
    ss: u8,
    se: u8,
    al: u8,
    eobrun: &mut u32,
    coef: &mut [i16; 64],
) -> Result<()> {
    if *eobrun > 0 {
        *eobrun -= 1;
        return Ok(());
    }
    let mut k = ss as usize;
    let se = se as usize;
    while k <= se {
        let rs = br.decode_symbol(ac_tbl)?;
        let run = (rs >> 4) as usize;
        let size = (rs & 0x0F) as u32;
        if size == 0 {
            if run != 15 {
                // EOBn: run-length encoded "end of band run" of
                // 2^run + low `run` bits additional blocks.
                let mut r = 1u32 << run;
                if run != 0 {
                    let extra = br.get_bits(run as u32)?;
                    r += extra;
                }
                *eobrun = r - 1;
                break;
            }
            // ZRL: 16 zeros, no value.
            k += 16;
            continue;
        }
        k += run;
        if k > se {
            return Err(DecodeError::Malformed("AC first scan run exceeded band"));
        }
        let bits = br.get_bits(size)?;
        let v = extend(bits, size);
        coef[k] = (v << al) as i16;
        k += 1;
    }
    Ok(())
}

// ---------------- AC refinement scan ----------------
//
// The most complex of the four. The scan walks the band Ss..=Se. For
// every coefficient already non-zero from a prior scan we MUST read a
// correction bit (1 → add (1 << Al) with the existing sign, 0 → no
// change). RLE codes count only NEW non-zeros being introduced; the
// "run" itself counts zero positions in the band, but existing
// non-zeros encountered along the way still need their correction bits.
//
// EOBn ends the new-non-zero insertion for the rest of this band in
// this block plus the next (run-encoded) blocks. WITHIN an EOBn run,
// existing non-zeros in the affected positions still receive
// correction bits — only the introduction of new non-zeros is
// suppressed.

#[allow(clippy::too_many_arguments)]
fn decode_ac_refine_block(
    br: &mut BitReader,
    ac_tbl: &HuffmanDecodeTable,
    ss: u8,
    se: u8,
    _ah: u8,
    al: u8,
    eobrun: &mut u32,
    coef: &mut [i16; 64],
) -> Result<()> {
    let ss = ss as usize;
    let se = se as usize;
    let p1 = 1i16 << al; // value added to new positive non-zeros
    let m1 = (-1i16) << al; // value for new negative non-zeros

    // If we're inside an EOB run from a prior block, no new non-zeros
    // are introduced — but existing non-zeros in this band still get
    // refinement bits.
    if *eobrun > 0 {
        refine_existing_band(br, coef, ss, se, p1, m1)?;
        *eobrun -= 1;
        return Ok(());
    }

    let mut k = ss;
    while k <= se {
        let rs = br.decode_symbol(ac_tbl)?;
        let run = (rs >> 4) as usize;
        let size = (rs & 0x0F) as u32;
        if size != 0 && size != 1 {
            return Err(DecodeError::Malformed(
                "AC refine: new-nonzero size must be 0 or 1",
            ));
        }
        if size == 0 {
            if run != 15 {
                // EOBn: refine existing in remainder of band, then set
                // eobrun for subsequent blocks.
                let mut r = 1u32 << run;
                if run != 0 {
                    let extra = br.get_bits(run as u32)?;
                    r += extra;
                }
                *eobrun = r - 1;
                refine_existing_band(br, coef, k, se, p1, m1)?;
                return Ok(());
            }
            // ZRL: skip 16 zero positions (counting only zero positions),
            // refining existing non-zeros encountered along the way.
            // No new non-zero appended at the end.
            let mut skipped = 0usize;
            while skipped < 16 {
                if k > se {
                    return Err(DecodeError::Malformed("AC refine ZRL exceeded band"));
                }
                if coef[k] != 0 {
                    refine_one_existing(br, &mut coef[k], p1, m1)?;
                } else {
                    skipped += 1;
                }
                k += 1;
            }
            continue;
        }
        // size == 1: a new ±1 non-zero will be appended after `run`
        // additional zero positions are crossed. Pre-read its sign now.
        let sign_bit = br.get_bit()?;
        let new_val = if sign_bit != 0 { p1 } else { m1 };
        let mut zeros_to_skip = run;
        while k <= se {
            if coef[k] != 0 {
                refine_one_existing(br, &mut coef[k], p1, m1)?;
                k += 1;
            } else if zeros_to_skip > 0 {
                zeros_to_skip -= 1;
                k += 1;
            } else {
                // Found the destination for the new non-zero.
                coef[k] = new_val;
                k += 1;
                break;
            }
        }
        if zeros_to_skip > 0 {
            return Err(DecodeError::Malformed("AC refine run exceeded band"));
        }
    }
    Ok(())
}

/// Refine every existing non-zero in `coef[k_lo..=k_hi]` by reading a
/// correction bit per position and applying it.
fn refine_existing_band(
    br: &mut BitReader,
    coef: &mut [i16; 64],
    k_lo: usize,
    k_hi: usize,
    p1: i16,
    m1: i16,
) -> Result<()> {
    for v in coef.iter_mut().take(k_hi + 1).skip(k_lo) {
        if *v != 0 {
            refine_one_existing(br, v, p1, m1)?;
        }
    }
    Ok(())
}

#[inline]
fn refine_one_existing(br: &mut BitReader, v: &mut i16, p1: i16, m1: i16) -> Result<()> {
    let bit = br.get_bit()?;
    if bit != 0 {
        if *v > 0 {
            *v = v.saturating_add(p1);
        } else {
            *v = v.saturating_add(m1);
        }
    }
    Ok(())
}

// ---------------- Restart handling ----------------

fn handle_restart(br: &mut BitReader, prev_dc: &mut [i32; 4], eobrun: &mut u32) -> Result<()> {
    br.reset_bit_buffer();
    while br.marker().is_none() {
        br.get_bits(8)?;
    }
    let m = br
        .marker()
        .expect("marker is Some — while loop only exits on Some");
    if !(0xD0..=0xD7).contains(&m) {
        return Err(DecodeError::Malformed("expected RSTn between intervals"));
    }
    br.clear_marker();
    *prev_dc = [0; 4];
    *eobrun = 0;
    Ok(())
}

// ---------------- Finalize: dequant + IDCT + place into planes ----------------

fn finalize_to_planes(
    frame: &FrameHeader,
    comps: &[CoeffComponent],
    quant: &[QuantTable],
) -> Result<DecodedPlanes> {
    let qt_by_id = super::baseline::index_quant_tables(quant);
    let h_max = frame.h_max() as usize;
    let v_max = frame.v_max() as usize;
    let mcu_w_pixels = h_max * 8;
    let mcu_h_pixels = v_max * 8;
    let mcus_x = (frame.width as usize).div_ceil(mcu_w_pixels);
    let mcus_y = (frame.height as usize).div_ceil(mcu_h_pixels);

    let mut planes: Vec<DecodedPlane> = Vec::with_capacity(comps.len());
    for cc in comps {
        let comp = cc.component;
        let pw = (frame.width as u32 * comp.h as u32).div_ceil(h_max as u32) as usize;
        let ph = (frame.height as u32 * comp.v as u32).div_ceil(v_max as u32) as usize;
        let stride = mcus_x * (comp.h as usize) * 8;
        let padded_height = mcus_y * (comp.v as usize) * 8;
        let qt = qt_by_id[comp.qt as usize].ok_or(DecodeError::Malformed(
            "frame component refers to undefined quant table",
        ))?;
        let mut samples = vec![0u8; stride * padded_height];

        let mut nat_coef = [0i16; 64];
        let mut block = [0u8; 64];
        for by in 0..cc.blocks_y {
            for bx in 0..cc.blocks_x {
                let zz = &cc.blocks[by * cc.blocks_x + bx];
                // De-zig-zag and dequantize.
                for k in 0..64 {
                    nat_coef[ZIGZAG[k]] = zz[k].wrapping_mul(qt.values[ZIGZAG[k]] as i16);
                }
                arch::backend::dct::idct_islow(&nat_coef, &mut block);
                let base_x = bx * 8;
                let base_y = by * 8;
                for j in 0..8 {
                    let dst_off = (base_y + j) * stride + base_x;
                    samples[dst_off..dst_off + 8].copy_from_slice(&block[j * 8..j * 8 + 8]);
                }
            }
        }
        planes.push(DecodedPlane {
            component: comp,
            stride,
            padded_height,
            plane_width: pw,
            plane_height: ph,
            samples,
        });
    }

    Ok(DecodedPlanes {
        width: frame.width as u32,
        height: frame.height as u32,
        components: planes,
    })
}
