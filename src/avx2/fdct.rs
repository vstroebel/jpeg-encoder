//! The forward DCT of [`crate::fdct`], eight lanes at a time with AVX2.
//!
//! Each vector holds one row of the block, so the vertical pass transforms
//! all eight columns at once. A transpose lets the horizontal pass do the
//! same for the rows, and a second one restores the layout. The integer
//! operations and rounding points are the scalar version's, so the
//! coefficients are identical.

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128i, __m256i, _mm_load_si128, _mm256_add_epi32, _mm256_cvtepi16_epi32, _mm256_mullo_epi32,
    _mm256_packs_epi32, _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set1_epi32,
    _mm256_setzero_si256, _mm256_slli_epi32, _mm256_srai_epi32, _mm256_store_si256,
    _mm256_sub_epi32, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi32,
    _mm256_unpacklo_epi64,
};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, _mm_load_si128, _mm256_add_epi32, _mm256_cvtepi16_epi32, _mm256_mullo_epi32,
    _mm256_packs_epi32, _mm256_permute2x128_si256, _mm256_permute4x64_epi64, _mm256_set1_epi32,
    _mm256_setzero_si256, _mm256_slli_epi32, _mm256_srai_epi32, _mm256_store_si256,
    _mm256_sub_epi32, _mm256_unpackhi_epi32, _mm256_unpackhi_epi64, _mm256_unpacklo_epi32,
    _mm256_unpacklo_epi64,
};

use crate::encoder::AlignedBlock;
use crate::fdct::{C1, C2, C3, C5, C6, C7, CONST_BITS, PASS1_BITS};

const CB: i32 = CONST_BITS as i32;
const P1: i32 = PASS1_BITS as i32;

pub fn fdct_avx2(data: &mut AlignedBlock) {
    // SAFETY: the encoder takes the AVX2 path only after detecting AVX2.
    unsafe { fdct_avx2_internal(data) }
}

#[target_feature(enable = "avx2")]
fn fdct_avx2_internal(data: &mut AlignedBlock) {
    let mut rows = [_mm256_setzero_si256(); 8];
    for (y, row) in rows.iter_mut().enumerate() {
        // SAFETY: the block is 32-byte aligned, so each 16-byte row is 16-byte aligned.
        let samples = unsafe { _mm_load_si128(data.data.as_ptr().add(y * 8) as *const __m128i) };
        *row = _mm256_cvtepi16_epi32(samples);
    }

    let vertical = pass::<false>(&rows);
    let horizontal = pass::<true>(&transpose(&vertical));
    let out = transpose(&horizontal);

    for v in (0..8).step_by(2) {
        // `packs` interleaves the two rows by 128-bit halves; put them back in order.
        let packed =
            _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packs_epi32(out[v], out[v + 1]));
        // SAFETY: two rows of i16 are 32 bytes, and the block is 32-byte aligned.
        unsafe { _mm256_store_si256(data.data.as_mut_ptr().add(v * 8) as *mut __m256i, packed) };
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn times(value: __m256i, constant: i32) -> __m256i {
    _mm256_mullo_epi32(value, _mm256_set1_epi32(constant))
}

/// Divide by `2^BITS`, rounding halves up.
#[inline]
#[target_feature(enable = "avx2")]
fn descale<const BITS: i32>(value: __m256i) -> __m256i {
    _mm256_srai_epi32::<BITS>(_mm256_add_epi32(value, _mm256_set1_epi32(1 << (BITS - 1))))
}

#[inline]
#[target_feature(enable = "avx2")]
fn sum(terms: [__m256i; 4]) -> __m256i {
    _mm256_add_epi32(
        _mm256_add_epi32(terms[0], terms[1]),
        _mm256_add_epi32(terms[2], terms[3]),
    )
}

/// One 8-point pass on eight lanes: the vertical one (`LAST` false) keeps
/// `PASS1_BITS` fraction bits, the horizontal one drops them.
#[inline]
#[target_feature(enable = "avx2")]
fn pass<const LAST: bool>(x: &[__m256i; 8]) -> [__m256i; 8] {
    let s0 = _mm256_add_epi32(x[0], x[7]);
    let s1 = _mm256_add_epi32(x[1], x[6]);
    let s2 = _mm256_add_epi32(x[2], x[5]);
    let s3 = _mm256_add_epi32(x[3], x[4]);
    let d0 = _mm256_sub_epi32(x[0], x[7]);
    let d1 = _mm256_sub_epi32(x[1], x[6]);
    let d2 = _mm256_sub_epi32(x[2], x[5]);
    let d3 = _mm256_sub_epi32(x[3], x[4]);

    let e0 = _mm256_add_epi32(s0, s3);
    let e1 = _mm256_add_epi32(s1, s2);
    let f0 = _mm256_sub_epi32(s0, s3);
    let f1 = _mm256_sub_epi32(s1, s2);

    let plain = [_mm256_add_epi32(e0, e1), _mm256_sub_epi32(e0, e1)];
    let products = [
        sum([times(d0, C1), times(d1, C3), times(d2, C5), times(d3, C7)]),
        _mm256_add_epi32(times(f0, C2), times(f1, C6)),
        sum([
            times(d0, C3),
            times(d1, -C7),
            times(d2, -C1),
            times(d3, -C5),
        ]),
        sum([times(d0, C5), times(d1, -C1), times(d2, C7), times(d3, C3)]),
        _mm256_sub_epi32(times(f0, C6), times(f1, C2)),
        sum([times(d0, C7), times(d1, -C5), times(d2, C3), times(d3, -C1)]),
    ];

    if LAST {
        [
            descale::<P1>(plain[0]),
            descale::<{ CB + P1 }>(products[0]),
            descale::<{ CB + P1 }>(products[1]),
            descale::<{ CB + P1 }>(products[2]),
            descale::<P1>(plain[1]),
            descale::<{ CB + P1 }>(products[3]),
            descale::<{ CB + P1 }>(products[4]),
            descale::<{ CB + P1 }>(products[5]),
        ]
    } else {
        [
            _mm256_slli_epi32::<P1>(plain[0]),
            descale::<{ CB - P1 }>(products[0]),
            descale::<{ CB - P1 }>(products[1]),
            descale::<{ CB - P1 }>(products[2]),
            _mm256_slli_epi32::<P1>(plain[1]),
            descale::<{ CB - P1 }>(products[3]),
            descale::<{ CB - P1 }>(products[4]),
            descale::<{ CB - P1 }>(products[5]),
        ]
    }
}

/// Swap the rows and columns of an 8×8 matrix of `i32`.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose(m: &[__m256i; 8]) -> [__m256i; 8] {
    // Pairs, then quads, within each 128-bit half; then swap the halves.
    let a0 = _mm256_unpacklo_epi32(m[0], m[1]);
    let a1 = _mm256_unpackhi_epi32(m[0], m[1]);
    let a2 = _mm256_unpacklo_epi32(m[2], m[3]);
    let a3 = _mm256_unpackhi_epi32(m[2], m[3]);
    let a4 = _mm256_unpacklo_epi32(m[4], m[5]);
    let a5 = _mm256_unpackhi_epi32(m[4], m[5]);
    let a6 = _mm256_unpacklo_epi32(m[6], m[7]);
    let a7 = _mm256_unpackhi_epi32(m[6], m[7]);

    let b0 = _mm256_unpacklo_epi64(a0, a2);
    let b1 = _mm256_unpackhi_epi64(a0, a2);
    let b2 = _mm256_unpacklo_epi64(a1, a3);
    let b3 = _mm256_unpackhi_epi64(a1, a3);
    let b4 = _mm256_unpacklo_epi64(a4, a6);
    let b5 = _mm256_unpackhi_epi64(a4, a6);
    let b6 = _mm256_unpacklo_epi64(a5, a7);
    let b7 = _mm256_unpackhi_epi64(a5, a7);

    [
        _mm256_permute2x128_si256::<0x20>(b0, b4),
        _mm256_permute2x128_si256::<0x20>(b1, b5),
        _mm256_permute2x128_si256::<0x20>(b2, b6),
        _mm256_permute2x128_si256::<0x20>(b3, b7),
        _mm256_permute2x128_si256::<0x31>(b0, b4),
        _mm256_permute2x128_si256::<0x31>(b1, b5),
        _mm256_permute2x128_si256::<0x31>(b2, b6),
        _mm256_permute2x128_si256::<0x31>(b3, b7),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdct::{fdct_scalar, tests::blocks};

    #[test]
    fn the_avx2_transform_matches_the_scalar_one() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for samples in blocks() {
            let mut scalar = AlignedBlock::new(samples);
            let mut avx2 = AlignedBlock::new(samples);
            fdct_scalar(&mut scalar);
            fdct_avx2(&mut avx2);
            assert_eq!(scalar.data, avx2.data, "{samples:?}");
        }
    }
}
