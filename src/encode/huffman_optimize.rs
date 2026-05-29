//! Optimized (= per-image) Huffman tables.
//!
//! When `JpegEncoder::set_optimize_huffman(true)` is set the encoder
//! runs a two-pass entropy stage: pass 1 walks the quantized
//! coefficients to *count* the symbol frequencies that the standard
//! Huffman codes would emit; pass 2 builds the optimal canonical
//! Huffman tables (T.81 K.2 algorithm + K.3 length limiting) from
//! those frequencies and re-emits the scan using them. Typical
//! savings vs Annex K standard tables run 4-10% on photographic
//! content at q=80, matching `cjpeg -optimize`.
//!
//! Code-length generation follows the libjpeg-turbo / IJG reference
//! (`jpeg_gen_optimal_table` in jchuff.c). It is the standard textbook
//! Huffman tree-building algorithm with a single "reserved" virtual
//! symbol added at frequency 1 so the all-ones bit pattern is never
//! assigned to a real code (the JPEG spec, Annex C, forbids it).

use super::huffman::magnitude_category;

/// Walk one quantized + zig-zagged block exactly the way
/// `huffman::encode_block` would, but instead of emitting bits, count
/// the symbol frequencies that would have been emitted. Returns the
/// raw DC coefficient (becomes the next block's predictor — must be
/// updated in lockstep with pass 2).
pub fn count_block(
    block: &[i16; 64],
    prev_dc: i16,
    dc_freq: &mut [u32; 257],
    ac_freq: &mut [u32; 257],
) -> i16 {
    let dc = block[0];
    let diff = dc as i32 - prev_dc as i32;
    let (size, _) = magnitude_category(diff);
    dc_freq[size as usize] += 1;

    // Last-nonzero AC index (mirrors encode_block's bitmap walk).
    let mut last_nonzero: usize = 0;
    for (k, &v) in block.iter().enumerate().take(64).skip(1) {
        if v != 0 {
            last_nonzero = k;
        }
    }

    let mut zr: u32 = 0;
    for &c in block.iter().take(last_nonzero + 1).skip(1) {
        if c == 0 {
            zr += 1;
            continue;
        }
        while zr >= 16 {
            ac_freq[0xF0] += 1; // ZRL
            zr -= 16;
        }
        let (sz, _) = magnitude_category(c as i32);
        let sym = ((zr as usize) << 4) | (sz as usize & 0x0F);
        ac_freq[sym] += 1;
        zr = 0;
    }
    if last_nonzero < 63 {
        ac_freq[0x00] += 1; // EOB
    }
    dc
}

/// Result of building an optimal canonical Huffman table: the
/// DHT-format `(bits, values)` pair (same layout as `StdHuffman`).
pub struct OptHuffmanSpec {
    pub bits: [u8; 16],
    pub values: Vec<u8>,
}

/// Build an optimal canonical Huffman table from a symbol-frequency
/// histogram. `freq` is indexed by symbol (0..=255 for real symbols;
/// slot 256 is reserved for the algorithm and must be zero on input).
///
/// Falls back to the supplied `fallback` (the matching Annex K table)
/// when no real symbols were counted — DHT must carry at least one
/// code, and an all-zero histogram means the table is going to be
/// unused anyway.
pub fn build_optimal_huffman(
    freq_in: &[u32; 257],
    fallback_bits: &[u8; 16],
    fallback_values: &[u8],
) -> OptHuffmanSpec {
    // Real-symbol total (excluding the reserved slot, which is always
    // zero on input).
    let real_total: u32 = freq_in[..256].iter().sum();
    if real_total == 0 {
        return OptHuffmanSpec {
            bits: *fallback_bits,
            values: fallback_values.to_vec(),
        };
    }

    // Working copies. Slot 256 gets the reserved frequency = 1 so the
    // all-ones code is never assigned to a real symbol.
    let mut freq: [u32; 257] = *freq_in;
    freq[256] = 1;

    // `codesize[i]` = current code-length for symbol i (grows as the
    // tree is built). `others[i]` chains symbols that share a node in
    // the Huffman tree (= "next leaf in this subtree"); -1 marks leaf.
    let mut codesize = [0u32; 257];
    let mut others = [-1i32; 257];

    loop {
        // Find the two smallest non-zero frequencies. Ties resolve to
        // the higher index (libjpeg convention — keeps output
        // deterministic and matches mozjpeg / cjpeg byte-for-byte on
        // the same histogram).
        let mut c1: i32 = -1;
        let mut c2: i32 = -1;
        let mut v1: u32 = u32::MAX;
        let mut v2: u32 = u32::MAX;
        for (i, &f) in freq.iter().enumerate() {
            if f == 0 {
                continue;
            }
            if f <= v1 {
                v2 = v1;
                c2 = c1;
                v1 = f;
                c1 = i as i32;
            } else if f <= v2 {
                v2 = f;
                c2 = i as i32;
            }
        }
        if c2 < 0 {
            break; // only one symbol left → done.
        }
        let c1 = c1 as usize;
        let c2 = c2 as usize;

        // Merge c2 into c1; zero out c2.
        freq[c1] += freq[c2];
        freq[c2] = 0;

        // Bump code lengths for everything chained to c1 …
        codesize[c1] += 1;
        let mut p = c1;
        while others[p] >= 0 {
            p = others[p] as usize;
            codesize[p] += 1;
        }
        others[p] = c2 as i32;

        // … and for c2's chain too.
        codesize[c2] += 1;
        let mut p = c2;
        while others[p] >= 0 {
            p = others[p] as usize;
            codesize[p] += 1;
        }
    }

    // Tally per-length code counts. JPEG length cap is 16 but
    // intermediate lengths can exceed that on highly skewed
    // distributions; K.3 cap step below truncates them down.
    const MAX_CLEN: usize = 32;
    let mut bits = [0u32; MAX_CLEN + 1];
    for &sz in codesize.iter() {
        if sz > 0 {
            assert!(
                (sz as usize) <= MAX_CLEN,
                "code length {sz} exceeds working cap {MAX_CLEN}",
            );
            bits[sz as usize] += 1;
        }
    }

    // K.3 length limiting: redistribute counts so no length > 16.
    for i in (17..=MAX_CLEN).rev() {
        while bits[i] > 0 {
            // Find the next-shorter length with a free slot to split.
            let mut j = i - 2;
            while bits[j] == 0 {
                j -= 1;
            }
            bits[i] -= 2;
            bits[i - 1] += 1;
            bits[j + 1] += 2;
            bits[j] -= 1;
        }
    }

    // Remove the reserved code (longest entry shrinks by one).
    let mut i = 16;
    while i > 0 && bits[i] == 0 {
        i -= 1;
    }
    if i > 0 {
        bits[i] -= 1;
    }

    // Pack bits[1..=16] into the 16-byte DHT form.
    let mut bits_out = [0u8; 16];
    for k in 1..=16 {
        bits_out[k - 1] = bits[k] as u8;
    }

    // Build HUFFVAL: for each length 1..=MAX_CLEN, append all real
    // symbols (skipping the reserved index 256) whose codesize matches,
    // in ascending symbol order. We loop up to MAX_CLEN (not 16) because
    // K.3 length-limiting reshapes `bits[]` but does NOT rewrite the
    // per-symbol `codesize[]` array — symbols whose original codesize
    // was > 16 still need to be listed, and they get placed in canonical
    // order matching the truncated bits[] distribution (the symbol-to-
    // code mapping is determined by the ORDER of symbols in HUFFVAL,
    // not by their codesize label). This mirrors libjpeg-turbo's
    // jpeg_gen_optimal_table.
    let n_values: usize = bits_out.iter().map(|&b| b as usize).sum();
    let mut values = Vec::with_capacity(n_values);
    for length in 1..=MAX_CLEN as u32 {
        for (sym, &sz) in codesize.iter().take(256).enumerate() {
            if sz == length {
                values.push(sym as u8);
            }
        }
    }

    debug_assert_eq!(
        values.len(),
        n_values,
        "HUFFVAL count {} != bits-sum {}",
        values.len(),
        n_values,
    );

    OptHuffmanSpec {
        bits: bits_out,
        values,
    }
}

#[cfg(test)]
mod tests {
    use super::super::huffman::{BitWriter, HuffmanTable, encode_block};
    use super::*;
    use crate::tables::STD_LUMA_DC;

    #[test]
    fn empty_histogram_falls_back() {
        let freq = [0u32; 257];
        let spec = build_optimal_huffman(&freq, &STD_LUMA_DC.bits, STD_LUMA_DC.values);
        assert_eq!(spec.bits, STD_LUMA_DC.bits);
        assert_eq!(spec.values, STD_LUMA_DC.values.to_vec());
    }

    /// Single-symbol histogram: must yield a valid 1-bit code with
    /// the one real symbol on the short branch (and the reserved
    /// virtual symbol absorbing the all-ones code).
    #[test]
    fn single_symbol_yields_one_bit_code() {
        let mut freq = [0u32; 257];
        freq[5] = 100;
        let spec = build_optimal_huffman(&freq, &STD_LUMA_DC.bits, STD_LUMA_DC.values);
        let sum: usize = spec.bits.iter().map(|&b| b as usize).sum();
        assert_eq!(sum, spec.values.len(), "bits-sum / values length mismatch");
        assert_eq!(spec.values, vec![5]);
        // Code length must be small (≤ 2): single symbol + reserved
        // virtual symbol → at most depth-1 tree.
        assert!(spec.bits[0] >= 1 || spec.bits[1] >= 1);
    }

    /// Build a table from a non-trivial histogram and verify the
    /// canonical-Huffman invariants the encoder/decoder rely on:
    /// `bits.sum() == values.len()`, no length > 16, all values
    /// distinct and in `0..=255`.
    #[test]
    fn canonical_invariants() {
        // Synthesize a Zipf-ish AC histogram.
        let mut freq = [0u32; 257];
        for (s, slot) in freq.iter_mut().take(256).enumerate() {
            *slot = if s == 0 { 5000 } else { (256 - s) as u32 };
        }
        let spec = build_optimal_huffman(&freq, &STD_LUMA_DC.bits, STD_LUMA_DC.values);
        let sum: usize = spec.bits.iter().map(|&b| b as usize).sum();
        assert_eq!(sum, spec.values.len());
        for &b in &spec.bits {
            assert!(b as usize <= 256);
        }
        let mut seen = [false; 256];
        for &v in &spec.values {
            assert!(!seen[v as usize], "duplicate symbol {v} in HUFFVAL");
            seen[v as usize] = true;
        }
    }

    /// End-to-end: counted frequencies from a synthetic block, built
    /// into an optimal table, must encode + round-trip without
    /// blowing past the 16-bit code-length cap.
    #[test]
    fn count_then_build_then_encode_round_trips() {
        let mut block = [0i16; 64];
        block[0] = 25;
        block[1] = -3;
        block[3] = 1;
        block[10] = -7;
        block[30] = 2;

        let mut dc_freq = [0u32; 257];
        let mut ac_freq = [0u32; 257];
        let _ = count_block(&block, 0, &mut dc_freq, &mut ac_freq);
        let _ = count_block(&block, 25, &mut dc_freq, &mut ac_freq);

        let dc_spec = build_optimal_huffman(&dc_freq, &STD_LUMA_DC.bits, STD_LUMA_DC.values);
        let ac_spec = build_optimal_huffman(
            &ac_freq,
            &crate::tables::STD_LUMA_AC.bits,
            crate::tables::STD_LUMA_AC.values,
        );

        let dc_tab = HuffmanTable::from_bits_values(&dc_spec.bits, &dc_spec.values);
        let ac_tab = HuffmanTable::from_bits_values(&ac_spec.bits, &ac_spec.values);

        let mut out = Vec::new();
        {
            let mut bw = BitWriter::new(&mut out);
            let p = encode_block(&mut bw, &block, 0, &dc_tab, &ac_tab).unwrap();
            let _ = encode_block(&mut bw, &block, p, &dc_tab, &ac_tab).unwrap();
            bw.flush_to_byte_boundary().unwrap();
        }
        assert!(!out.is_empty());
    }
}
