//! Baseline Huffman entropy coding (DC + AC).
//!
//! Optimizations applied:
//!
//!   1. **64-bit bit accumulator** — `BitWriter` keeps a `u64` of pending
//!      bits left-aligned at the MSB. Each `write_bits` call appends with
//!      a single shift+OR, then drains whole bytes only when the
//!      accumulator has at least 32 bits queued (so most calls drain
//!      4 bytes at a time, not 1).
//!
//!   2. **Branchless inner path** — `write_bits` is straight-line code on
//!      the common path (no flush). The drain branch fires once every
//!      ~3-5 symbols on typical content.
//!
//!   3. **Packed Huffman table** — `(length << 16) | code` packed into a
//!      single `u32` per symbol. One load per symbol, no struct field
//!      offsetting.
//!
//!   4. **Bitmap-driven AC scan** — `arch::backend::huffman::nonzero_bitmap`
//!      packs the per-coefficient zero/non-zero predicate into a single
//!      `u64`. The AC walk then uses `trailing_zeros` to jump directly
//!      to each next nonzero, replacing the previous group-of-8 skip +
//!      per-coefficient branch. On aarch64 the bitmap is built with
//!      `vceqz + vmovn + vaddv`; on x86_64 / scalar the obvious loop
//!      autovectorizes.
//!
//!   5. **Internal byte buffer** — the bit accumulator drains into a
//!      `Vec<u8>` we own, and we only call the user's `Write` when
//!      `flush_to_byte_boundary` is invoked (once per scan). This turns
//!      ~one `write_all` per emitted byte into one `write_all` per scan.
//!
//! Output is bit-exact identical to the previous (scalar, u32-accumulator)
//! implementation; we have a parallel reference encoder under `#[cfg(test)]`
//! that asserts this on a panel of synthetic blocks.

use std::io;

use crate::tables::StdHuffman;

/// Packed Huffman code table: one `u32` per symbol, layout
/// `(length << 16) | code`. `length` is in `0..=16` (0 means "this symbol
/// has no code" — only ever observed for unused entries), `code` is the
/// right-aligned bit pattern.
///
/// Single load per symbol, no struct field offset arithmetic.
pub struct HuffmanTable {
    pub packed: [u32; 256],
}

impl HuffmanTable {
    /// Build a packed encoder table directly from a DHT-format
    /// (bits[16], values) pair. Used by the optimized-Huffman path,
    /// which constructs its tables at runtime and never materializes a
    /// `StdHuffman`.
    pub fn from_bits_values(bits: &[u8; 16], values: &[u8]) -> Self {
        let total_codes: usize = bits.iter().map(|&b| b as usize).sum();
        debug_assert_eq!(
            total_codes,
            values.len(),
            "bits sum must equal values.len()",
        );
        let mut packed: [u32; 256] = [0; 256];
        let mut next_code: u32 = 0;
        let mut value_idx: usize = 0;
        for length in 1..=16u32 {
            let count = bits[(length - 1) as usize] as usize;
            for _ in 0..count {
                let sym = values[value_idx] as usize;
                packed[sym] = (length << 16) | next_code;
                next_code += 1;
                value_idx += 1;
            }
            next_code <<= 1;
        }
        HuffmanTable { packed }
    }

    pub fn from_std(table: &StdHuffman) -> Self {
        // Canonical-Huffman expansion (Annex C.2): see `from_bits_values`.
        Self::from_bits_values(&table.bits, table.values)
    }

    /// Helper for tests: split `packed[sym]` back into `(code, length)`.
    #[cfg(test)]
    fn split(&self, sym: usize) -> (u16, u8) {
        let p = self.packed[sym];
        ((p & 0xFFFF) as u16, (p >> 16) as u8)
    }
}

/// MSB-first bit accumulator.
///
/// Bits live in the high end of a `u64`: the next bit to emit is bit 63,
/// the bit after that is 62, and so on. `nbits` records how many of those
/// high bits are valid. New bits are OR'd in just below the current
/// queue. When 32 or more bits are queued we drain four bytes in one
/// shot (with byte-stuffing per JPEG B.1.1.5).
///
/// We accumulate output into an owned `Vec<u8>` and forward to the
/// user's `Write` only on `flush_to_byte_boundary`. This avoids one
/// `write_all` syscall per byte; the typical scan does ~one syscall
/// total.
pub struct BitWriter<W: io::Write> {
    inner: W,
    /// Pending output bytes. Sized large up-front to avoid reallocs in
    /// the hot path; we set capacity from a hint when the scan begins.
    buf: Vec<u8>,
    /// Bit queue, MSB-aligned. The next bit to emit is `(buffer >> 63) & 1`.
    buffer: u64,
    /// Number of valid bits currently in `buffer` (0..=63 after each
    /// `write_bits` returns; the drain step ensures we never need more
    /// than that).
    nbits: u32,
}

impl<W: io::Write> BitWriter<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            buf: Vec::new(),
            buffer: 0,
            nbits: 0,
        }
    }

    /// Reserve `cap` bytes of internal buffer up front. Cheap to call;
    /// no-op if the buffer is already at least that large.
    pub fn reserve(&mut self, cap: usize) {
        if self.buf.capacity() < cap {
            self.buf.reserve(cap - self.buf.capacity());
        }
    }

    /// Push the low `n_bits` of `value` into the stream MSB-first.
    /// `n_bits` must be in `0..=27` — comfortably above the largest
    /// single-token emission (a 16-bit Huffman code or an 11-bit
    /// magnitude tail).
    ///
    /// Branchless on the common path: shift, mask, OR, increment. The
    /// drain branch fires only when we've accumulated ≥32 queued bits.
    #[inline(always)]
    pub fn write_bits(&mut self, value: u32, n_bits: u32) -> io::Result<()> {
        self.push_bits(value, n_bits);
        Ok(())
    }

    /// Infallible hot-path form of `write_bits`.
    ///
    /// The public method keeps the existing API for progressive paths and
    /// tests, but baseline block entropy coding can avoid threading
    /// `io::Result` through every emitted Huffman token.
    #[inline(always)]
    fn push_bits(&mut self, value: u32, n_bits: u32) {
        debug_assert!(n_bits <= 27, "write_bits over-budget: {n_bits}");
        // `n_bits == 0` is allowed (used by callers when emitting a
        // zero-magnitude token); the resulting shift is well-defined
        // because we only shift by `n_bits` after masking.
        if n_bits == 0 {
            return;
        }
        // OR in just past the existing queue. The new bits are placed at
        // bits [63-nbits-n_bits .. 63-nbits] of `buffer`.
        let shift = 64 - self.nbits - n_bits;
        self.buffer |= (u64::from(value) & ((1u64 << n_bits) - 1)) << shift;
        self.nbits += n_bits;
        if self.nbits >= 32 {
            self.drain_high32();
        }
    }

    #[inline(always)]
    fn local_writer(&mut self) -> LocalBitWriter<'_> {
        LocalBitWriter {
            buf: &mut self.buf,
            buffer: self.buffer,
            nbits: self.nbits,
        }
    }

    /// Drain the top 32 bits of the accumulator as four bytes (with
    /// 0xFF stuffing). Called whenever the queue depth reaches 32+; on
    /// exit `nbits` is in `0..32`.
    ///
    /// Per-byte write goes through an unsafe pointer-bump path after a
    /// single `reserve(8)` (= 4 data bytes + worst-case 4 stuffing
    /// bytes) so the hot loop skips `Vec::push`'s per-call bounds /
    /// capacity check. Bit-identical output to the previous
    /// `push_stuffed × 4` implementation.
    #[inline]
    fn drain_high32(&mut self) {
        let high = (self.buffer >> 32) as u32;
        // 4 data bytes + up to 4 stuffing bytes = 8.
        self.buf.reserve(8);
        let len = self.buf.len();
        let mut written: usize = 0;
        // Safety: we just reserved 8 bytes; `written ∈ 0..=8` after the
        // loop (each byte writes 1 byte, optionally +1 for stuffing).
        // `u8` is plain-old-data so writing through `*mut u8` and then
        // `set_len` is sound.
        unsafe {
            let dst = self.buf.as_mut_ptr().add(len);
            for &b in &[
                (high >> 24) as u8,
                (high >> 16) as u8,
                (high >> 8) as u8,
                high as u8,
            ] {
                *dst.add(written) = b;
                written += 1;
                if b == 0xFF {
                    *dst.add(written) = 0;
                    written += 1;
                }
            }
            self.buf.set_len(len + written);
        }
        self.buffer <<= 32;
        self.nbits -= 32;
    }

    /// Flush the entropy stream to a byte boundary and emit a restart
    /// marker (`RSTn`, n in 0..=7) immediately after. Caller must
    /// reset the DC predictors to zero on the next block (F.1.5.4).
    /// Used by the encoder when a non-zero restart interval is set.
    pub fn write_restart(&mut self, n: u8) -> io::Result<()> {
        debug_assert!(n < 8);
        self.flush_to_byte_boundary()?;
        self.inner.write_all(&[0xFF, 0xD0 | (n & 0x07)])
    }

    /// Pad the final partial byte with 1-bits, drain everything, and
    /// flush the internal buffer to the inner writer. Required at the
    /// end of each entropy-coded segment (Annex F.1.5.5).
    pub fn flush_to_byte_boundary(&mut self) -> io::Result<()> {
        if self.nbits > 0 {
            // Pad to next byte boundary with 1-bits.
            let pad_bits = (8 - (self.nbits & 7)) & 7;
            if pad_bits > 0 {
                self.buffer |= ((1u64 << pad_bits) - 1) << (64 - self.nbits - pad_bits);
                self.nbits += pad_bits;
            }
            // Drain whole bytes.
            while self.nbits >= 8 {
                let byte = (self.buffer >> 56) as u8;
                push_stuffed(&mut self.buf, byte);
                self.buffer <<= 8;
                self.nbits -= 8;
            }
            debug_assert_eq!(self.nbits, 0, "flush should drain to zero");
        }
        // One write to the inner sink.
        self.inner.write_all(&self.buf)?;
        self.buf.clear();
        Ok(())
    }
}

struct LocalBitWriter<'a> {
    buf: &'a mut Vec<u8>,
    buffer: u64,
    nbits: u32,
}

impl LocalBitWriter<'_> {
    #[inline(always)]
    fn push_bits(&mut self, value: u32, n_bits: u32) {
        debug_assert!(n_bits <= 27, "write_bits over-budget: {n_bits}");
        if n_bits == 0 {
            return;
        }
        let shift = 64 - self.nbits - n_bits;
        self.buffer |= (u64::from(value) & ((1u64 << n_bits) - 1)) << shift;
        self.nbits += n_bits;
        if self.nbits >= 32 {
            self.drain_high32();
        }
    }

    #[inline(always)]
    fn push_packed(&mut self, entry: u32) {
        self.push_bits(entry & 0xFFFF, entry >> 16);
    }

    #[inline(always)]
    fn push_packed_with_bits(&mut self, entry: u32, bits: u32, n_bits: u32) {
        let code = entry & 0xFFFF;
        let len = entry >> 16;
        self.push_bits((code << n_bits) | bits, len + n_bits);
    }

    #[inline]
    fn drain_high32(&mut self) {
        let high = (self.buffer >> 32) as u32;
        // 4 data bytes + up to 4 stuffing bytes = 8.
        self.buf.reserve(8);
        let len = self.buf.len();
        let mut written: usize = 0;
        // Safety: we just reserved 8 bytes; `written ∈ 0..=8` after the
        // loop (each byte writes 1 byte, optionally +1 for stuffing).
        // `u8` is plain-old-data so writing through `*mut u8` and then
        // `set_len` is sound.
        unsafe {
            let dst = self.buf.as_mut_ptr().add(len);
            for &b in &[
                (high >> 24) as u8,
                (high >> 16) as u8,
                (high >> 8) as u8,
                high as u8,
            ] {
                *dst.add(written) = b;
                written += 1;
                if b == 0xFF {
                    *dst.add(written) = 0;
                    written += 1;
                }
            }
            self.buf.set_len(len + written);
        }
        self.buffer <<= 32;
        self.nbits -= 32;
    }

    #[inline(always)]
    fn finish(self) -> (u64, u32) {
        (self.buffer, self.nbits)
    }
}

#[inline(always)]
fn push_stuffed(buf: &mut Vec<u8>, byte: u8) {
    buf.push(byte);
    if byte == 0xFF {
        buf.push(0x00);
    }
}

/// Encode one 8x8 block's quantized + zig-zagged coefficients using
/// the supplied DC and AC tables. Returns the new running DC predictor
/// (the raw DC coefficient of this block, used as the predictor for the
/// next block of the same component).
///
/// `block` is in zig-zag order (so `block[0]` is DC, `block[1..]` is
/// AC scan).
pub fn encode_block<W: io::Write>(
    bw: &mut BitWriter<W>,
    block: &[i16; 64],
    prev_dc: i16,
    dc_tab: &HuffmanTable,
    ac_tab: &HuffmanTable,
) -> io::Result<i16> {
    let mut local = bw.local_writer();

    // ----- DC term: difference-coded (F.1.2.1) -----
    //
    // Emit `(huff_code, magnitude_bits)` as a single fused write. JPEG
    // bounds DC magnitude category at 11 and Huffman code length at 16,
    // so the combined width is `≤ 27` — fits `write_bits`' budget. When
    // `size == 0` (`diff == 0`) `bits` is 0 and the fused value
    // collapses to the bare Huffman code.
    let dc = block[0];
    let diff = dc as i32 - prev_dc as i32;
    let (size, bits) = magnitude_category(diff);
    local.push_packed_with_bits(dc_tab.packed[size as usize], bits, size as u32);

    // ----- AC terms: run-length of zeros + magnitude (F.1.2.2) -----
    // Build a 64-bit bitmap of nonzero positions, then drive the scan
    // via trailing/leading-zeros so every walked position is a hit.
    let ac_bitmap = crate::arch::backend::huffman::nonzero_bitmap(block) & !1u64;
    // On aarch64 only, precompute (size, bits) for every coefficient
    // in one SIMD pass — the NEON path covers all 64 lanes in ~5
    // vector ops, replacing the ~10 per-coefficient
    // `magnitude_category` calls in the inner loop. x86_64 stays on
    // the per-coefficient inline path: epi16 has no vector `clz`
    // outside AVX-512, so a scalar pre-pass over all 64 lanes is
    // measurably worse than the bitmap-skipping inline form.
    #[cfg(target_arch = "aarch64")]
    let (sizes, bits_lut) = {
        let mut sizes = [0u8; 64];
        let mut bits_lut = [0u16; 64];
        crate::arch::backend::huffman::ac_magnitudes(block, &mut sizes, &mut bits_lut);
        (sizes, bits_lut)
    };

    if ac_bitmap == 0 {
        // All AC coefficients zero → emit EOB and we're done.
        local.push_packed(ac_tab.packed[0x00]);
        let (buffer, nbits) = local.finish();
        bw.buffer = buffer;
        bw.nbits = nbits;
        return Ok(dc);
    }

    let ac_packed = &ac_tab.packed;
    let eob = ac_packed[0x00];
    let zrl = ac_packed[0xF0];
    let last_nonzero = 63 - ac_bitmap.leading_zeros() as usize;
    let mut remaining = ac_bitmap;
    let mut k: usize = 1;
    while remaining != 0 {
        let next_k = remaining.trailing_zeros() as usize;
        let mut zero_run = (next_k - k) as u32;
        // ZRL (F0): emit 16-zero placeholder until run < 16.
        while zero_run >= 16 {
            local.push_packed(zrl);
            zero_run -= 16;
        }
        // (size, bits) — from the precomputed lut on aarch64, from a
        // per-coefficient `magnitude_category` call on x86_64. JPEG
        // bounds AC magnitude category at 10; we mask the symbol field
        // to 4 bits as a belt-and-braces guard.
        #[cfg(target_arch = "aarch64")]
        let (size, bits): (u32, u32) = (sizes[next_k] as u32, bits_lut[next_k] as u32);
        #[cfg(not(target_arch = "aarch64"))]
        let (size, bits): (u32, u32) = {
            let (s, b) = magnitude_category(block[next_k] as i32);
            (s as u32, b)
        };
        let symbol = ((zero_run as usize) << 4) | (size as usize & 0x0F);
        // Fuse the Huffman code and magnitude-bits emits into one
        // `write_bits` call. AC magnitude category is bounded at 10
        // and Huffman code length at 16, so the combined width fits
        // `write_bits`' 27-bit budget. Halves the per-coefficient
        // shift / mask / OR / drain-check chain on the AC hot path.
        local.push_packed_with_bits(ac_packed[symbol], bits, size);
        remaining &= !(1u64 << next_k);
        k = next_k + 1;
    }

    // Trailing zeros → EOB (symbol 0x00).
    if last_nonzero < 63 {
        local.push_packed(eob);
    }

    let (buffer, nbits) = local.finish();
    bw.buffer = buffer;
    bw.nbits = nbits;
    Ok(dc)
}

/// JPEG "magnitude category" (Annex F.1.2):
///   - `size`  = ⌈log2(|x|+1)⌉, in `0..=11`
///   - `bits`  = the size-bit representation of `x`:
///     positive `x`: low `size` bits of `x`;
///     negative `x`: low `size` bits of `x - 1` (1's complement).
#[inline]
pub(crate) fn magnitude_category(value: i32) -> (u8, u32) {
    if value == 0 {
        return (0, 0);
    }
    let abs = value.unsigned_abs();
    // BSR: position of MSB. `leading_zeros` on a 32-bit value gives us
    // 32 - bit_length; thus bit_length = 32 - leading_zeros.
    let size = (32 - abs.leading_zeros()) as u8;
    let bits = if value > 0 {
        abs
    } else {
        // 1's complement of |x|, masked to `size` bits.
        (value - 1) as u32 & ((1u32 << size) - 1)
    };
    (size, bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::{STD_CHROMA_AC, STD_CHROMA_DC, STD_LUMA_AC, STD_LUMA_DC};

    #[test]
    fn magnitude_zero() {
        assert_eq!(magnitude_category(0), (0, 0));
    }

    #[test]
    fn magnitude_positive() {
        // 5 → size=3, bits=0b101
        assert_eq!(magnitude_category(5), (3, 0b101));
    }

    #[test]
    fn magnitude_negative() {
        // -5 → size=3, bits=0b010 (1's complement of 5 within 3 bits)
        assert_eq!(magnitude_category(-5), (3, 0b010));
    }

    #[test]
    fn bitwriter_stuffs_ff() {
        let mut out = Vec::new();
        {
            let mut bw = BitWriter::new(&mut out);
            // Push 8 ones.
            bw.write_bits(0xFF, 8).unwrap();
            bw.flush_to_byte_boundary().unwrap();
        }
        assert_eq!(out, vec![0xFF, 0x00]);
    }

    #[test]
    fn bitwriter_packs_multibyte() {
        // Three 12-bit values → 36 bits. After the 32-bit drain the
        // remaining 4 bits get padded with 1s on flush, yielding 5
        // bytes total: hex AAA BBB CCC = bits
        //   1010 1010 1010 1011 1011 1011 1100 1100 1100
        // Pad final 4 bits with 1111 → 0xAA 0xAB 0xBB 0xCC 0xCF
        let mut out = Vec::new();
        {
            let mut bw = BitWriter::new(&mut out);
            bw.write_bits(0xAAA, 12).unwrap();
            bw.write_bits(0xBBB, 12).unwrap();
            bw.write_bits(0xCCC, 12).unwrap();
            bw.flush_to_byte_boundary().unwrap();
        }
        assert_eq!(out, vec![0xAA, 0xAB, 0xBB, 0xCC, 0xCF]);
    }

    #[test]
    fn bitwriter_stuffs_ff_during_high32_drain() {
        let mut out = Vec::new();
        {
            let mut bw = BitWriter::new(&mut out);
            bw.write_bits(0x12FF, 16).unwrap();
            bw.write_bits(0x34FF, 16).unwrap();
            bw.flush_to_byte_boundary().unwrap();
        }
        assert_eq!(out, vec![0x12, 0xFF, 0x00, 0x34, 0xFF, 0x00]);
    }

    #[test]
    fn packed_table_matches_canonical() {
        // Canonical luma DC: symbol 0 has length 2 and code 0. (Annex K.3.)
        let t = HuffmanTable::from_std(&STD_LUMA_DC);
        let (code, len) = t.split(0);
        assert_eq!(len, 2);
        assert_eq!(code, 0);
    }

    /// Reference encoder using the pre-Phase-2.5 formulation: 32-bit
    /// accumulator, parallel `code`/`size` arrays, no NEON. Bit-exact
    /// equivalence between this and `encode_block` is the contract we
    /// preserve.
    fn reference_encode(
        block: &[i16; 64],
        prev_dc: i16,
        dc_tab: &HuffmanTable,
        ac_tab: &HuffmanTable,
    ) -> Vec<u8> {
        // Local mini-bitwriter (u32, single-byte drain). Mirror of the
        // pre-2.5 implementation.
        struct RefBW {
            out: Vec<u8>,
            buf: u32,
            n: u32,
        }
        impl RefBW {
            fn new() -> Self {
                Self {
                    out: Vec::new(),
                    buf: 0,
                    n: 0,
                }
            }
            fn write(&mut self, value: u32, nb: u32) {
                if nb == 0 {
                    return;
                }
                self.buf |= (value & ((1u32 << nb) - 1)) << (32 - self.n - nb);
                self.n += nb;
                while self.n >= 8 {
                    let byte = (self.buf >> 24) as u8;
                    self.buf <<= 8;
                    self.n -= 8;
                    self.out.push(byte);
                    if byte == 0xFF {
                        self.out.push(0x00);
                    }
                }
            }
            fn flush(&mut self) {
                if self.n > 0 {
                    let pad = 8 - self.n;
                    self.write((1u32 << pad) - 1, pad);
                }
            }
        }

        let mut bw = RefBW::new();
        let dc = block[0];
        let diff = dc as i32 - prev_dc as i32;
        let (size, bits) = magnitude_category(diff);
        let (code, len) = dc_tab.split(size as usize);
        bw.write(code as u32, len as u32);
        if size > 0 {
            bw.write(bits, size as u32);
        }
        let mut last_nonzero = 0usize;
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
                let (cd, ln) = ac_tab.split(0xF0);
                bw.write(cd as u32, ln as u32);
                zr -= 16;
            }
            let (sz, bs) = magnitude_category(c as i32);
            let sym = ((zr as u8) << 4) | (sz & 0x0F);
            let (cd, ln) = ac_tab.split(sym as usize);
            bw.write(cd as u32, ln as u32);
            bw.write(bs, sz as u32);
            zr = 0;
        }
        if last_nonzero < 63 {
            let (cd, ln) = ac_tab.split(0x00);
            bw.write(cd as u32, ln as u32);
        }
        bw.flush();
        bw.out
    }

    fn opt_encode(
        block: &[i16; 64],
        prev_dc: i16,
        dc_tab: &HuffmanTable,
        ac_tab: &HuffmanTable,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        {
            let mut bw = BitWriter::new(&mut out);
            encode_block(&mut bw, block, prev_dc, dc_tab, ac_tab).unwrap();
            bw.flush_to_byte_boundary().unwrap();
        }
        out
    }

    fn assert_eq_blocks(label: &str, block: &[i16; 64], prev_dc: i16) {
        let dc = HuffmanTable::from_std(&STD_LUMA_DC);
        let ac = HuffmanTable::from_std(&STD_LUMA_AC);
        let r = reference_encode(block, prev_dc, &dc, &ac);
        let o = opt_encode(block, prev_dc, &dc, &ac);
        assert_eq!(o, r, "{label}: optimized output diverges from reference");

        // Also exercise chroma tables — different code lengths, catches
        // length-overflow bugs that luma might mask.
        let dcc = HuffmanTable::from_std(&STD_CHROMA_DC);
        let acc = HuffmanTable::from_std(&STD_CHROMA_AC);
        let r2 = reference_encode(block, prev_dc, &dcc, &acc);
        let o2 = opt_encode(block, prev_dc, &dcc, &acc);
        assert_eq!(
            o2, r2,
            "{label} (chroma): optimized output diverges from reference"
        );
    }

    #[test]
    fn equiv_all_zero() {
        let block = [0i16; 64];
        assert_eq_blocks("all_zero", &block, 0);
        assert_eq_blocks("all_zero_dc_diff", &block, 42);
    }

    #[test]
    fn equiv_dc_only() {
        let mut block = [0i16; 64];
        block[0] = 100;
        assert_eq_blocks("dc_only_pos", &block, 0);
        block[0] = -50;
        assert_eq_blocks("dc_only_neg", &block, 100);
    }

    #[test]
    fn equiv_full_random() {
        // Deterministic LCG. Range chosen to stay within JPEG's 11-bit
        // magnitude category window.
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        let mut block = [0i16; 64];
        for v in block.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((state >> 53) as i16).wrapping_sub(512); // -512..=511 ≈ 11 bits incl. sign
        }
        assert_eq_blocks("full_random", &block, 0);
        assert_eq_blocks("full_random_with_pred", &block, 7);
    }

    #[test]
    fn equiv_sparse_ac() {
        // Few non-zero ACs, long zero runs — exercises the NEON
        // zero-skip path heavily.
        let mut block = [0i16; 64];
        block[0] = 33;
        block[1] = -2;
        block[5] = 1;
        block[31] = -7;
        block[32] = 4;
        block[63] = 1;
        assert_eq_blocks("sparse_ac", &block, 0);
    }

    #[test]
    fn equiv_dense_ac() {
        // Every AC nonzero — NEON skip never fires.
        let mut block = [0i16; 64];
        for (i, v) in block.iter_mut().enumerate() {
            *v = ((i as i16) % 7) - 3;
            if *v == 0 {
                *v = 1;
            }
        }
        assert_eq_blocks("dense_ac", &block, 0);
    }

    #[test]
    fn equiv_zrl_path() {
        // Force a zero run ≥16 to hit the ZRL emission.
        let mut block = [0i16; 64];
        block[0] = 5;
        block[1] = 3;
        block[20] = -2; // 18-zero run between 1 and 20
        block[40] = 1;
        assert_eq_blocks("zrl_path", &block, 0);
    }

    #[test]
    fn equiv_eob_at_various_positions() {
        for last in [1usize, 7, 8, 9, 16, 32, 47, 62, 63] {
            let mut block = [0i16; 64];
            block[0] = 10;
            block[last] = -1;
            assert_eq_blocks(&format!("eob_last={last}"), &block, 0);
        }
    }

    #[test]
    fn equiv_max_magnitude() {
        // Largest value the magnitude-category encoder will emit.
        let mut block = [0i16; 64];
        block[0] = 1023;
        block[1] = -1023;
        block[2] = 1023;
        assert_eq_blocks("max_magnitude", &block, -1023);
    }
}
