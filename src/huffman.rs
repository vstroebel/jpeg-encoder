/*
 * The default huffman tables are taken from
 * section K.3 Typical Huffman tables for 8-bit precision luminance and chrominance
 */

use alloc::vec::Vec;

#[derive(Copy, Clone, Debug)]
pub enum CodingClass {
    Dc = 0,
    Ac = 1,
}

static DEFAULT_LUMA_DC_CODE_LENGTHS: [u8; 16] = [
    0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

static DEFAULT_LUMA_DC_VALUES: [u8; 12] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
];

static DEFAULT_CHROMA_DC_CODE_LENGTHS: [u8; 16] = [
    0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
];

static DEFAULT_CHROMA_DC_VALUES: [u8; 12] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
];

static DEFAULT_LUMA_AC_CODE_LENGTHS: [u8; 16] = [
    0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
];

static DEFAULT_LUMA_AC_VALUES: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
    0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
    0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];

static DEFAULT_CHROMA_AC_CODE_LENGTHS: [u8; 16] = [
    0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00, 0x01, 0x02, 0x77,
];

static DEFAULT_CHROMA_AC_VALUES: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0,
    0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
    0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
    0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
    0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
    0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
    0xF9, 0xFA,
];

pub struct HuffmanTable {
    lookup_table: [(u8, u16); 256],
    length: [u8; 16],
    values: Vec<u8>,
}

impl HuffmanTable {
    pub fn new(length: &[u8; 16], values: &[u8]) -> HuffmanTable {
        HuffmanTable {
            lookup_table: create_lookup_table(length, values),
            length: *length,
            values: values.to_vec(),
        }
    }

    pub fn default_luma_dc() -> HuffmanTable {
        Self::new(&DEFAULT_LUMA_DC_CODE_LENGTHS, &DEFAULT_LUMA_DC_VALUES)
    }

    pub fn default_luma_ac() -> HuffmanTable {
        Self::new(&DEFAULT_LUMA_AC_CODE_LENGTHS, &DEFAULT_LUMA_AC_VALUES)
    }

    pub fn default_chroma_dc() -> HuffmanTable {
        Self::new(&DEFAULT_CHROMA_DC_CODE_LENGTHS, &DEFAULT_CHROMA_DC_VALUES)
    }

    pub fn default_chroma_ac() -> HuffmanTable {
        Self::new(&DEFAULT_CHROMA_AC_CODE_LENGTHS, &DEFAULT_CHROMA_AC_VALUES)
    }

    /// Generates an optimized huffman table as described in Section K.2
    #[allow(clippy::needless_range_loop)]
    pub fn new_optimized(freq: [u32; 257]) -> HuffmanTable {
        // Merged counts are sums of counts; u64 so they cannot wrap.
        let mut freq = freq.map(u64::from);
        let mut others = [-1i32; 257];
        let mut codesize = [0usize; 257];

        // Find Huffman code sizes
        // Figure K.1
        loop {
            let mut v1 = None;
            let mut v1_min = u64::MAX;

            // Find the largest value with the least non zero frequency
            for (i, &f) in freq.iter().enumerate() {
                if f > 0 && f <= v1_min {
                    v1_min = f;
                    v1 = Some(i);
                }
            }

            let mut v1 = match v1 {
                Some(v) => v,
                None => break,
            };

            let mut v2 = None;
            let mut v2_min = u64::MAX;

            // Find the next largest value with the least non zero frequency
            for (i, &f) in freq.iter().enumerate() {
                if f > 0 && f <= v2_min && i != v1 {
                    v2_min = f;
                    v2 = Some(i);
                }
            }

            let mut v2 = match v2 {
                Some(v) => v,
                None => break,
            };

            freq[v1] += freq[v2];
            freq[v2] = 0;

            codesize[v1] += 1;
            while others[v1] >= 0 {
                v1 = others[v1] as usize;
                codesize[v1] += 1;
            }

            others[v1] = v2 as i32;

            codesize[v2] += 1;
            while others[v2] >= 0 {
                v2 = others[v2] as usize;
                codesize[v2] += 1;
            }
        }

        // Find the number of codes of each size
        // Figure K.2

        // K.2 sizes this 33 on the assumption that no code exceeds 32 bits,
        // which skewed counts break: 33 symbols with near-Fibonacci counts
        // (~18.5M total) already go deeper. 257 symbols cannot go past 256.
        let mut bits = [0u16; 257];

        for &size in codesize.iter() {
            if size > 0 {
                bits[size] += 1;
            }
        }

        // Limiting code lengths to 16 bits
        // Figure K.3

        let mut i = bits.len() - 1;

        while i > 16 {
            while bits[i] > 0 {
                let mut j = i - 2;
                while bits[j] == 0 {
                    j -= 1;
                }

                bits[i] -= 2;
                bits[i - 1] += 1;
                bits[j + 1] += 2;
                bits[j] -= 1;
            }

            i -= 1;
        }

        while bits[i] == 0 {
            debug_assert!(i > 0, "Error creating codesizes for frequency {:?}", freq);
            i -= 1;
        }
        bits[i] -= 1;

        // Sorting of input values according to code size
        // Figure K.4
        let mut huffval = [0u8; 256];

        let mut k = 0;

        // Every size K.1 produced, not just up to 32: see `bits` above.
        for i in 1..=256 {
            for j in 0..=255 {
                if codesize[j] == i {
                    huffval[k] = j as u8;
                    k += 1;
                }
            }
        }

        let mut length = [0u8; 16];
        for (i, v) in length.iter_mut().enumerate() {
            // At most 256 real symbols, and never all at one length.
            *v = bits[i + 1] as u8;
        }

        let values = huffval[0..k].to_vec();

        HuffmanTable {
            lookup_table: create_lookup_table(&length, &values),
            length,
            values,
        }
    }

    #[inline]
    pub fn get_for_value(&self, value: u8) -> &(u8, u16) {
        let res = &self.lookup_table[value as usize];
        debug_assert!(res.0 > 0, "Got zero size code for value: {}", value);
        res
    }

    pub fn length(&self) -> &[u8; 16] {
        &self.length
    }

    pub fn values(&self) -> &[u8] {
        &self.values
    }
}

// Create huffman table code sizes as defined in Figure C.1
fn create_sizes(code_lengths: &[u8; 16]) -> [u8; 256] {
    let mut sizes = [0u8; 256];

    let mut k = 0;

    for (i, &length) in code_lengths.iter().enumerate() {
        for _ in 0..length {
            sizes[k] = (i + 1) as u8;
            k += 1;
        }
    }

    sizes
}

// Create huffman table codes as defined in Figure C.2
fn create_codes(sizes: &[u8; 256]) -> [u16; 256] {
    let mut codes = [0u16; 256];

    let mut current_code = 0;
    let mut current_size = sizes[0];

    for (&size, code) in sizes.iter().take_while(|s| **s != 0).zip(codes.iter_mut()) {
        if current_size != size {
            let size_diff = size - current_size;
            current_code <<= size_diff as usize;
            current_size = size;
        }

        *code = current_code;
        current_code += 1;
    }

    codes
}

// Create huffman table codes as defined in Figure C.3
fn create_lookup_table(code_length: &[u8; 16], values: &[u8]) -> [(u8, u16); 256] {
    let sizes = create_sizes(code_length);
    let codes = create_codes(&sizes);

    let mut lookup_table = [(0u8, 0u16); 256];

    for (i, &value) in values.iter().enumerate() {
        lookup_table[value as usize] = (sizes[i], codes[i]);
    }

    lookup_table
}

/// Properties of [`HuffmanTable::new_optimized`] under its caller contract:
/// `freq[256] == 1` (the reserved all-ones code) and symbols drawn from the
/// sets the encoder can actually emit.
#[cfg(test)]
mod optimized_props {
    use super::HuffmanTable;
    use proptest::prelude::*;

    /// Every AC symbol `writer::count_block` can count: EOB, ZRL and
    /// `(run << 4) | size` for run 0..=15, size 1..=10.
    fn ac_symbols() -> alloc::vec::Vec<usize> {
        let mut s = alloc::vec![0x00, 0xF0];
        for run in 0..16 {
            for size in 1..=10 {
                s.push((run << 4) | size);
            }
        }
        s
    }

    /// Asserts the table is a complete-minus-one prefix code covering every
    /// nonzero symbol, with lengths in 1..=16.
    fn check(freq: &[u32; 257]) {
        let t = HuffmanTable::new_optimized(*freq);
        let n: usize = t.length().iter().map(|&l| l as usize).sum();
        assert_eq!(n, t.values().len(), "BITS sum != HUFFVAL count");
        // Kraft sum strictly below 1: the all-ones code stays reserved.
        let kraft: u64 = t
            .length()
            .iter()
            .enumerate()
            .map(|(i, &l)| (l as u64) << (16 - (i + 1)))
            .sum();
        assert!(kraft < 1 << 16, "Kraft sum {kraft} >= 2^16");
        for (sym, &f) in freq[..256].iter().enumerate() {
            if f > 0 {
                let (len, code) = *t.get_for_value(sym as u8);
                assert!((1..=16).contains(&len), "symbol {sym} len {len}");
                assert_ne!(u32::from(code), (1u32 << len) - 1, "all-ones code used");
            }
        }
        // Prefix-free over the assigned codes.
        let codes: alloc::vec::Vec<(u8, u16)> =
            t.values().iter().map(|&v| *t.get_for_value(v)).collect();
        for (i, &(la, ca)) in codes.iter().enumerate() {
            for &(lb, cb) in &codes[i + 1..] {
                let (s, ls, l, ll) = if la <= lb {
                    (ca, la, cb, lb)
                } else {
                    (cb, lb, ca, la)
                };
                assert_ne!(l >> (ll - ls), s, "prefix collision");
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4096))]

        /// Arbitrary counts over the reachable AC alphabet.
        #[test]
        fn ac_uniform(counts in proptest::collection::vec(0u32..1_000_000, 162)) {
            let mut freq = [0u32; 257];
            for (&s, &c) in ac_symbols().iter().zip(&counts) {
                freq[s] = c;
            }
            freq[256] = 1;
            check(&freq);
        }

        /// Heavily skewed counts (powers of two up to 2^24) build deep trees,
        /// which uniform counts never do; this is what exercises K.3.
        #[test]
        fn ac_skewed(exps in proptest::collection::vec(0u32..25, 162)) {
            let mut freq = [0u32; 257];
            for (&s, &e) in ac_symbols().iter().zip(&exps) {
                freq[s] = 1 << e;
            }
            freq[256] = 1;
            check(&freq);
        }

        /// DC: the 12 categories, any counts.
        #[test]
        fn dc_any(counts in proptest::collection::vec(0u32..u32::MAX / 16, 12)) {
            let mut freq = [0u32; 257];
            freq[..12].copy_from_slice(&counts);
            freq[256] = 1;
            check(&freq);
        }
    }

    /// Tie-free Fibonacci counts (`f(n) = f(n-1) + f(n-2) + 1`) are the worst
    /// case for code depth: 40 symbols give a natural Huffman depth of ~39,
    /// past the 33-entry `bits` array before K.3 gets to limit it. Exact
    /// Fibonacci ties do not do it; K.1's tie-breaking rebalances them.
    #[test]
    fn fibonacci_depth() {
        let syms = ac_symbols();
        let mut freq = [0u32; 257];
        let (mut a, mut b) = (1u32, 1u32);
        for &s in syms.iter().take(40) {
            freq[s] = a;
            (a, b) = (b, a + b + 1);
        }
        freq[256] = 1;
        check(&freq);
    }

    /// Total count above `u32::MAX`: `freq[v1] += freq[v2]` must not wrap.
    #[test]
    fn total_over_u32() {
        let mut freq = [0u32; 257];
        freq[0] = u32::MAX / 2 + 1;
        freq[1] = u32::MAX / 2 + 1;
        freq[2] = 7;
        freq[256] = 1;
        check(&freq);
    }
}
