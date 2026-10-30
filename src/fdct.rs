//! Forward discrete cosine transform of one 8×8 block.
//!
//! An implementation of the transform as the JPEG standard defines it
//! (ITU-T T.81, A.3.3):
//!
//! ```text
//! F(u,v) = 1/4 C(u) C(v) Σx Σy f(x,y) cos((2x+1)uπ/16) cos((2y+1)vπ/16)
//! ```
//!
//! with `C(0) = 1/√2` and `C(k) = 1` otherwise. The input holds the
//! level-shifted samples, `f(x,y)` at `data[y * 8 + x]`. The quantization
//! tables carry a factor of 8 (see `quantization.rs`), so the output is
//! `8·F(u,v)`, at `data[v * 8 + u]`.
//!
//! The transform is separable: a vertical 8-point pass over the columns, then
//! a horizontal one over the rows. With the factor of 8 shared between them,
//! each pass computes
//!
//! ```text
//! X(k) = Σn x(n) · √2 C(k) cos((2n+1)kπ/16)
//! ```
//!
//! as an even half over the sums `s(n) = x(n) + x(7-n)` (outputs 0, 2, 4 and 6)
//! and an odd half over the differences `d(n) = x(n) - x(7-n)` (outputs 1, 3,
//! 5 and 7). Outputs 0 and 4 need no multiplication; the others multiply by
//! `√2 cos(mπ/16)`, held as integers scaled by `2^CONST_BITS`. The vertical
//! pass keeps `PASS1_BITS` extra fraction bits, which the horizontal pass
//! drops with rounding. Every intermediate fits in an `i32`.
//!
//! The scalar, `wide` and AVX2 versions round at the same points, so they
//! produce identical coefficients.

use crate::encoder::AlignedBlock;
use core::ops::{Add, Sub};

pub(crate) const CONST_BITS: u32 = 13;
pub(crate) const PASS1_BITS: u32 = 3;

// `round(2^CONST_BITS · √2 · cos(mπ/16))` for m = 1, 2, 3, 5, 6 and 7
// (`the_constants_follow_from_the_definition` recomputes them).
pub(crate) const C1: i32 = 11363;
pub(crate) const C2: i32 = 10703;
pub(crate) const C3: i32 = 9633;
pub(crate) const C5: i32 = 6436;
pub(crate) const C6: i32 = 4433;
pub(crate) const C7: i32 = 2260;

/// Integer lanes the passes run on: one value, or one row of eight.
pub(crate) trait Lane: Copy + Add<Output = Self> + Sub<Output = Self> {
    fn times(self, constant: i32) -> Self;
    fn shift_left(self, bits: u32) -> Self;
    /// Divide by `2^bits`, rounding halves up.
    fn descale(self, bits: u32) -> Self;
}

impl Lane for i32 {
    #[inline(always)]
    fn times(self, constant: i32) -> Self {
        self * constant
    }

    #[inline(always)]
    fn shift_left(self, bits: u32) -> Self {
        self << bits
    }

    #[inline(always)]
    fn descale(self, bits: u32) -> Self {
        (self + (1 << (bits - 1))) >> bits
    }
}

/// One 8-point pass before scaling: outputs 0 and 4 as plain sums, the
/// other six as products scaled by `2^CONST_BITS`.
#[inline(always)]
pub(crate) fn butterfly<L: Lane>(x: &[L; 8]) -> [L; 8] {
    let s0 = x[0] + x[7];
    let s1 = x[1] + x[6];
    let s2 = x[2] + x[5];
    let s3 = x[3] + x[4];
    let d0 = x[0] - x[7];
    let d1 = x[1] - x[6];
    let d2 = x[2] - x[5];
    let d3 = x[3] - x[4];

    let e0 = s0 + s3;
    let e1 = s1 + s2;
    let f0 = s0 - s3;
    let f1 = s1 - s2;

    [
        e0 + e1,
        d0.times(C1) + d1.times(C3) + d2.times(C5) + d3.times(C7),
        f0.times(C2) + f1.times(C6),
        d0.times(C3) - d1.times(C7) - d2.times(C1) - d3.times(C5),
        e0 - e1,
        d0.times(C5) - d1.times(C1) + d2.times(C7) + d3.times(C3),
        f0.times(C6) - f1.times(C2),
        d0.times(C7) - d1.times(C5) + d2.times(C3) - d3.times(C1),
    ]
}

/// The vertical pass: keep `PASS1_BITS` fraction bits.
#[inline(always)]
pub(crate) fn vertical<L: Lane>(x: &[L; 8]) -> [L; 8] {
    let raw = butterfly(x);
    core::array::from_fn(|k| {
        if k % 4 == 0 {
            raw[k].shift_left(PASS1_BITS)
        } else {
            raw[k].descale(CONST_BITS - PASS1_BITS)
        }
    })
}

/// The horizontal pass: drop every fraction bit.
#[inline(always)]
pub(crate) fn horizontal<L: Lane>(x: &[L; 8]) -> [L; 8] {
    let raw = butterfly(x);
    core::array::from_fn(|k| {
        if k % 4 == 0 {
            raw[k].descale(PASS1_BITS)
        } else {
            raw[k].descale(CONST_BITS + PASS1_BITS)
        }
    })
}

#[cfg(not(feature = "use_wide"))]
pub use fdct_scalar as fdct;

#[cfg(any(test, not(feature = "use_wide")))]
pub fn fdct_scalar(data: &mut AlignedBlock) {
    let mut columns = [[0i32; 8]; 8];
    for (x, column) in columns.iter_mut().enumerate() {
        let samples: [i32; 8] = core::array::from_fn(|y| i32::from(data.data[y * 8 + x]));
        *column = vertical(&samples);
    }
    for (v, out) in data.data.chunks_exact_mut(8).enumerate() {
        let row: [i32; 8] = core::array::from_fn(|x| columns[x][v]);
        for (coefficient, value) in out.iter_mut().zip(horizontal(&row)) {
            *coefficient = value as i16;
        }
    }
}

#[cfg(feature = "use_wide")]
impl Lane for wide::i32x8 {
    #[inline(always)]
    fn times(self, constant: i32) -> Self {
        self * wide::i32x8::splat(constant)
    }

    #[inline(always)]
    fn shift_left(self, bits: u32) -> Self {
        self << bits
    }

    #[inline(always)]
    fn descale(self, bits: u32) -> Self {
        (self + wide::i32x8::splat(1 << (bits - 1))) >> bits
    }
}

#[cfg(feature = "use_wide")]
fn transpose(rows: [wide::i32x8; 8]) -> [wide::i32x8; 8] {
    let rows = rows.map(wide::i32x8::to_array);
    core::array::from_fn(|i| wide::i32x8::new(core::array::from_fn(|j| rows[j][i])))
}

#[cfg(feature = "use_wide")]
pub fn fdct(data: &mut AlignedBlock) {
    use wide::i32x8;

    // Each vector is a row: the vertical pass runs on all eight columns at once.
    let rows: [i32x8; 8] = core::array::from_fn(|y| {
        i32x8::new(core::array::from_fn(|x| i32::from(data.data[y * 8 + x])))
    });
    let vertical = vertical(&rows);
    let horizontal = horizontal(&transpose(vertical));
    for (v, row) in transpose(horizontal).into_iter().enumerate() {
        for (u, value) in row.to_array().into_iter().enumerate() {
            data.data[v * 8 + u] = value as i16;
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    extern crate std;

    use super::*;
    use core::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2};
    use std::vec::Vec;

    /// `8·F(u,v)` straight from the definition, in `f64`.
    pub(crate) fn reference(samples: &[i16; 64]) -> [f64; 64] {
        let basis = |k: usize, n: usize| ((2 * n + 1) as f64 * k as f64 * PI / 16.0).cos();
        let weight = |k: usize| if k == 0 { FRAC_1_SQRT_2 } else { 1.0 };
        core::array::from_fn(|i| {
            let (v, u) = (i / 8, i % 8);
            let mut sum = 0.0;
            for y in 0..8 {
                for x in 0..8 {
                    sum += f64::from(samples[y * 8 + x]) * basis(u, x) * basis(v, y);
                }
            }
            8.0 * 0.25 * weight(u) * weight(v) * sum
        })
    }

    /// Blocks at the extremes of the sample range, then pseudo-random ones.
    pub(crate) fn blocks() -> Vec<[i16; 64]> {
        let mut blocks = Vec::new();
        for (low, high) in [(-128, -128), (127, 127), (-128, 127), (127, -128)] {
            blocks.push(core::array::from_fn(|i| {
                if (i / 8 + i % 8) % 2 == 0 { low } else { high }
            }));
            blocks.push(core::array::from_fn(
                |i| if i % 2 == 0 { low } else { high },
            ));
            blocks.push(core::array::from_fn(|i| {
                if (i / 8) % 2 == 0 { low } else { high }
            }));
        }
        blocks.push(core::array::from_fn(|i| (i as i16) * 4 - 128));
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        for round in 0..20_000 {
            blocks.push(core::array::from_fn(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let bits = (state >> 32) as i16;
                if round % 4 == 0 {
                    if bits & 1 == 0 { -128 } else { 127 }
                } else {
                    (bits & 0xff) - 128
                }
            }));
        }
        blocks
    }

    #[test]
    fn the_constants_follow_from_the_definition() {
        for (m, constant) in [(1, C1), (2, C2), (3, C3), (5, C5), (6, C6), (7, C7)] {
            let exact = f64::from(1u32 << CONST_BITS) * SQRT_2 * (f64::from(m) * PI / 16.0).cos();
            assert_eq!(constant, exact.round() as i32, "m = {m}");
        }
    }

    #[test]
    fn the_transform_is_the_definition_within_one() {
        for samples in blocks() {
            let expected = reference(&samples);
            let mut block = AlignedBlock::new(samples);
            fdct_scalar(&mut block);
            for (i, (&got, want)) in block.data.iter().zip(expected).enumerate() {
                assert!(
                    (f64::from(got) - want).abs() <= 1.0,
                    "coefficient {i} of {samples:?}: {got} against {want}"
                );
            }
        }
    }

    #[cfg(feature = "use_wide")]
    #[test]
    fn the_wide_transform_matches_the_scalar_one() {
        for samples in blocks() {
            let mut scalar = AlignedBlock::new(samples);
            let mut wide = AlignedBlock::new(samples);
            fdct_scalar(&mut scalar);
            fdct(&mut wide);
            assert_eq!(scalar.data, wide.data, "{samples:?}");
        }
    }
}
