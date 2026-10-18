/*
 * Portable translation of ../avx2/fdct.rs, originally ported from mozjpeg.
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2009, 2016, 2018, 2020, D. R. Commander.
 * Based on the x86 SIMD extension for IJG JPEG library
 * Copyright (C) 1999-2006, MIYASAKA Masaru.
 * See ../fdct.rs for the IJG license terms.
 */

use crate::encoder::AlignedBlock;
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
use fearless_simd::i32x4;
use fearless_simd::{Level, dispatch, i16x16, i32x8, prelude::*};

pub fn fdct(data: &mut AlignedBlock) {
    let level = Level::try_detect().unwrap_or_else(Level::baseline);
    dispatch!(level, simd => transform(simd, data));
}

// All helpers must inline into dispatch!'s target-feature context.
// Intrinsics absent from the portable API live in kernel! wrappers.
#[inline(always)]
fn madd<S: Simd>(a: i16x16<S>, b: i16x16<S>) -> i32x8<S> {
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    let simd = a.simd;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if let Some(avx2) = simd.level().as_avx2() {
            return madd_avx2(avx2, a.into(), b.into()).simd_into(simd);
        }
        if let Some(sse2) = simd.level().as_sse2() {
            let (al, ah) = a.split();
            let (bl, bh) = b.split();
            let low: i32x4<S> = madd_sse2(sse2, al.into(), bl.into()).simd_into(simd);
            let high: i32x4<S> = madd_sse2(sse2, ah.into(), bh.into()).simd_into(simd);
            return low.combine(high);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = simd.level().as_neon() {
        let (al, ah) = a.split();
        let (bl, bh) = b.split();
        let low: i32x4<S> = madd_neon(neon, al.into(), bl.into()).simd_into(simd);
        let high: i32x4<S> = madd_neon(neon, ah.into(), bh.into()).simd_into(simd);
        return low.combine(high);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(wasm) = simd.level().as_wasm_simd128() {
        let (al, ah) = a.split();
        let (bl, bh) = b.split();
        let low: i32x4<S> = madd_wasm(wasm, al.into(), bl.into()).simd_into(simd);
        let high: i32x4<S> = madd_wasm(wasm, ah.into(), bh.into()).simd_into(simd);
        return low.combine(high);
    }
    madd_portable(a, b)
}

#[inline(always)]
fn madd_portable<S: Simd>(a: i16x16<S>, b: i16x16<S>) -> i32x8<S> {
    let (al, ah) = a.widen();
    let (bl, bh) = b.widen();
    let (even, odd) = (al * bl).deinterleave(ah * bh);
    // SIMD addition wraps, including the (-32768 * -32768) * 2 corner case.
    even + odd
}

#[cfg(target_arch = "x86")]
use core::arch::x86::{__m128i, __m256i, _mm_madd_epi16, _mm256_madd_epi16, _mm256_packs_epi32};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m128i, __m256i, _mm_madd_epi16, _mm256_madd_epi16, _mm256_packs_epi32};

fearless_simd::kernel!(
    #[inline(always)]
    fn madd_avx2(simd: Avx2, a: __m256i, b: __m256i) -> __m256i {
        _mm256_madd_epi16(a, b)
    }
);
fearless_simd::kernel!(
    #[inline(always)]
    fn madd_sse2(simd: Sse2, a: __m128i, b: __m128i) -> __m128i {
        _mm_madd_epi16(a, b)
    }
);

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    int16x8_t, int32x4_t, vget_high_s16, vget_low_s16, vmull_s16, vpaddq_s32,
};
fearless_simd::kernel!(
    #[inline(always)]
    fn madd_neon(simd: Neon, a: int16x8_t, b: int16x8_t) -> int32x4_t {
        let low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
        let high = vmull_s16(vget_high_s16(a), vget_high_s16(b));
        vpaddq_s32(low, high)
    }
);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use core::arch::wasm32::{i32x4_dot_i16x8, v128};
fearless_simd::kernel!(
    #[inline(always)]
    fn madd_wasm(simd: WasmSimd128, a: v128, b: v128) -> v128 {
        i32x4_dot_i16x8(a, b)
    }
);

// AVX2 unpack operates separately in each 128-bit block. Whole-vector
// interleave would instead put all of a's lower half into the first output.
#[inline(always)]
fn unpack16<S: Simd>(a: i16x16<S>, b: i16x16<S>) -> (i16x16<S>, i16x16<S>) {
    let (al, ah) = a.split();
    let (bl, bh) = b.split();
    let (ll, lh) = al.interleave(bl);
    let (hl, hh) = ah.interleave(bh);
    (ll.combine(hl), lh.combine(hh))
}

#[inline(always)]
fn unpack32<S: Simd>(a: i16x16<S>, b: i16x16<S>) -> (i16x16<S>, i16x16<S>) {
    let a = i32x8::from_bytes(a.to_bytes());
    let b = i32x8::from_bytes(b.to_bytes());
    let (al, ah) = a.split();
    let (bl, bh) = b.split();
    let (ll, lh) = al.interleave(bl);
    let (hl, hh) = ah.interleave(bh);
    (
        i16x16::from_bytes(ll.combine(hl).to_bytes()),
        i16x16::from_bytes(lh.combine(hh).to_bytes()),
    )
}

#[inline(always)]
fn pack<S: Simd>(a: i32x8<S>, b: i32x8<S>) -> i16x16<S> {
    // Narrow each 128-bit half: [a0..a3, b0..b3, a4..a7, b4..b7].
    // Descaled FDCT values fit i16.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(avx2) = a.simd.level().as_avx2() {
        return pack_avx2(avx2, a.into(), b.into()).simd_into(a.simd);
    }
    let (al, ah) = a.split();
    let (bl, bh) = b.split();
    al.relaxed_narrow(bl).combine(ah.relaxed_narrow(bh))
}

// The intrinsic has exactly the half-wise layout required by pack, including
// when called from the AVX-512 backend.
fearless_simd::kernel!(
    #[inline(always)]
    fn pack_avx2(simd: Avx2, a: __m256i, b: __m256i) -> __m256i {
        _mm256_packs_epi32(a, b)
    }
);

#[inline(always)]
fn swap_halves<S: Simd>(a: i16x16<S>) -> i16x16<S> {
    let (low, high) = a.split();
    high.combine(low)
}

#[inline(always)]
fn column_pair<S: Simd, const REVERSED: bool>(a: i16x16<S>) -> i16x16<S> {
    // Input: [first_top, second_top, first_bottom, second_bottom].
    // Each group holds four i16 values. Join the top and bottom of each column.
    let halves = if REVERSED { [1, 3, 0, 2] } else { [0, 2, 1, 3] };
    a.swizzle_dyn(core::array::from_fn::<_, 32, _>(|byte| {
        (halves[byte / 8] * 8 + byte % 8) as u8
    }))
}

#[inline(always)]
fn transpose<S: Simd>(v: [i16x16<S>; 4]) -> [i16x16<S>; 4] {
    // Input pairs rows (0,4), (1,5), (2,6), (3,7).
    let (a, b) = unpack16(v[0], v[1]);
    let (c, d) = unpack16(v[2], v[3]);
    let (a, c) = unpack32(a, c);
    let (b, d) = unpack32(b, d);
    // Output pairs columns (1,0), (3,2), (4,5), (6,7).
    [
        column_pair::<_, true>(a),
        column_pair::<_, true>(c),
        column_pair::<_, false>(b),
        column_pair::<_, false>(d),
    ]
}

#[inline(always)]
fn coefficients<S: Simd>(simd: S, low: [i16; 2], high: [i16; 2]) -> i16x16<S> {
    i16x16::from_fn(simd, |i| if i < 8 { low[i % 2] } else { high[i % 2] })
}

#[inline(always)]
fn descale<S: Simd, const FIRST: bool>(low: i32x8<S>, high: i32x8<S>) -> i16x16<S> {
    let shift = if FIRST { 11 } else { 15 };
    let rounding = i32x8::splat(low.simd, 1 << (shift - 1));
    pack((low + rounding) >> shift, (high + rounding) >> shift)
}

#[inline(always)]
fn pass<S: Simd, const FIRST: bool>(v: [i16x16<S>; 4]) -> [i16x16<S>; 4] {
    let simd = v[0].simd;
    let diff07 = v[0] - v[3]; // tmp6_7
    let sum01 = swap_halves(v[0] + v[3]); // tmp0_1
    let sum23 = v[1] + v[2]; // tmp3_2
    let diff23 = v[1] - v[2]; // tmp4_5
    let even_sum = sum01 + sum23; // tmp10_11
    let even_diff = sum01 - sum23; // tmp13_12
    let signs = i16x16::from_fn(simd, |i| if i < 8 { 1 } else { -1 });
    let even = swap_halves(even_sum) + even_sum * signs;
    let data04 = if FIRST {
        even << 2
    } else {
        // Unlike the original AVX2 constant, round EVERY i16 lane.
        (even + i16x16::splat(simd, 2)) >> 2
    };

    // Distribute the shared 4433 product, so each output is one dot product:
    // data2 = tmp13*(4433+6270) + tmp12*4433
    // data6 = tmp12*(4433-15137) + tmp13*4433
    let c26 = coefficients(simd, [4433 + 6270, 4433], [4433 - 15137, 4433]);
    let (a, b) = unpack16(even_diff, swap_halves(even_diff));
    let data26 = descale::<_, FIRST>(madd(a, c26), madd(b, c26));

    // Combine z5 into z3 and z4 before multiplication.
    let z34 = diff23 + diff07;
    let (a, b) = unpack16(z34, swap_halves(z34));
    let c34 = coefficients(simd, [9633 - 16069, 9633], [9633 - 3196, 9633]);
    let zlow = madd(a, c34);
    let zhigh = madd(b, c34);

    // Similarly combine z1/z2 into the four remaining odd products.
    let c75 = coefficients(simd, [2446 - 7373, -7373], [16819 - 20995, -20995]);
    let (a, b) = unpack16(diff23, swap_halves(diff07));
    let data75 = descale::<_, FIRST>(madd(a, c75) + zlow, madd(b, c75) + zhigh);
    let c31 = coefficients(simd, [25172 - 20995, -20995], [12299 - 7373, -7373]);
    let (a, b) = unpack16(diff07, swap_halves(diff23));
    let data31 = descale::<_, FIRST>(madd(a, c31) + zlow, madd(b, c31) + zhigh);
    [data04, data31, data26, data75]
}

#[inline(always)]
fn transform<S: Simd>(simd: S, block: &mut AlignedBlock) {
    let data = &mut block.data;
    let (r0, r1) = i16x16::from_slice(simd, &data[..16]).split();
    let (r2, r3) = i16x16::from_slice(simd, &data[16..32]).split();
    let (r4, r5) = i16x16::from_slice(simd, &data[32..48]).split();
    let (r6, r7) = i16x16::from_slice(simd, &data[48..]).split();
    let rows = [
        r0.combine(r4),
        r1.combine(r5),
        r2.combine(r6),
        r3.combine(r7),
    ];
    let [a, b, c, d] = pass::<_, true>(transpose(rows));
    let (b0, b1) = b.split();
    let (d0, d1) = d.split();
    let columns = [a, b1.combine(d1), c, b0.combine(d0)];
    let [a, b, c, d] = pass::<_, false>(transpose(columns));
    let (a0, a1) = a.split();
    let (b0, b1) = b.split();
    let (c0, c1) = c.split();
    let (d0, d1) = d.split();
    a0.combine(b1).store_slice(&mut data[..16]);
    c0.combine(b0).store_slice(&mut data[16..32]);
    a1.combine(d1).store_slice(&mut data[32..48]);
    c1.combine(d0).store_slice(&mut data[48..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdct::fdct_scalar;
    use alloc::{vec, vec::Vec};

    fn levels() -> Vec<Level> {
        let detected = Level::try_detect().unwrap_or_else(Level::baseline);
        let mut levels = vec![Level::fallback(), detected];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(s) = detected.as_sse2() {
                levels.push(s.level());
            }
            if let Some(s) = detected.as_sse4_2() {
                levels.push(s.level());
            }
            if let Some(s) = detected.as_avx2() {
                levels.push(s.level());
            }
        }
        levels.dedup_by_key(|level| core::mem::discriminant(level));
        levels
    }

    #[inline(always)]
    fn check_layout<S: Simd>(simd: S) {
        let a = i16x16::from_fn(simd, |i| i as i16);
        let b = a + i16x16::splat(simd, 100);
        let (low, high) = unpack16(a, b);
        assert_eq!(
            <[i16; 16]>::from(low),
            [
                0, 100, 1, 101, 2, 102, 3, 103, 8, 108, 9, 109, 10, 110, 11, 111
            ]
        );
        assert_eq!(
            <[i16; 16]>::from(high),
            [
                4, 104, 5, 105, 6, 106, 7, 107, 12, 112, 13, 113, 14, 114, 15, 115
            ]
        );
        let (low, high) = unpack32(a, b);
        assert_eq!(
            <[i16; 16]>::from(low),
            [
                0, 1, 100, 101, 2, 3, 102, 103, 8, 9, 108, 109, 10, 11, 110, 111
            ]
        );
        assert_eq!(
            <[i16; 16]>::from(high),
            [
                4, 5, 104, 105, 6, 7, 106, 107, 12, 13, 112, 113, 14, 15, 114, 115
            ]
        );
        let a = i32x8::simd_from(simd, [-8192, -4096, -32768, -1, 0, 32767, 4096, 8192]);
        let b = i32x8::from_fn(simd, |i| 100 + i as i32);
        assert_eq!(
            <[i16; 16]>::from(pack(a, b)),
            [
                -8192, -4096, -32768, -1, 100, 101, 102, 103, 0, 32767, 4096, 8192, 104, 105, 106,
                107
            ]
        );

        let rows = core::array::from_fn(|r| {
            i16x16::from_fn(simd, |i| ((r + (i / 8) * 4) * 8 + i % 8) as i16)
        });
        let result = transpose(rows);
        for (v, columns) in result.into_iter().zip([[1, 0], [3, 2], [4, 5], [6, 7]]) {
            let expected = core::array::from_fn(|i| ((i % 8) * 8 + columns[i / 8]) as i16);
            assert_eq!(<[i16; 16]>::from(v), expected);
        }
    }

    #[test]
    fn lane_layout_and_narrowing() {
        for level in levels() {
            dispatch!(level, simd => check_layout(simd));
        }
    }

    #[inline(always)]
    fn check_madd<S: Simd>(simd: S) {
        let inputs = [
            ([i16::MIN; 16], [i16::MIN; 16]),
            (
                core::array::from_fn(|i| (i as i16 - 8) * 4095),
                core::array::from_fn(|i| if i % 2 == 0 { i16::MIN } else { i16::MAX }),
            ),
            (
                core::array::from_fn(|i| i as i16 - 8),
                core::array::from_fn(|i| 15 - i as i16),
            ),
        ];
        for (a, b) in inputs {
            let expected: [i32; 8] = core::array::from_fn(|i| {
                (i32::from(a[2 * i]) * i32::from(b[2 * i]))
                    .wrapping_add(i32::from(a[2 * i + 1]) * i32::from(b[2 * i + 1]))
            });
            let a = a.simd_into(simd);
            let b = b.simd_into(simd);
            assert_eq!(<[i32; 8]>::from(madd(a, b)), expected);
            assert_eq!(<[i32; 8]>::from(madd_portable(a, b)), expected);
        }
    }

    #[test]
    fn adjacent_pair_multiply_add() {
        for level in levels() {
            dispatch!(level, simd => check_madd(simd));
        }
    }

    fn random_block(state: &mut u64) -> [i16; 64] {
        core::array::from_fn(|_| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (*state & 255) as i16 - 128
        })
    }

    #[inline(always)]
    fn compare<S: Simd>(simd: S, input: [i16; 64]) {
        let mut actual = AlignedBlock::new(input);
        let mut expected = actual;
        fdct_scalar(&mut expected);
        transform(simd, &mut actual);
        assert_eq!(
            actual.data,
            expected.data,
            "level: {:?}, input: {:?}",
            simd.level(),
            input
        );
    }

    #[inline(always)]
    fn check_blocks<S: Simd>(simd: S) {
        for value in [-128, -1, 0, 1, 127] {
            compare(simd, [value; 64]);
        }
        for i in 0..64 {
            for value in [-128, 1, 127] {
                let mut input = [0; 64];
                input[i] = value;
                compare(simd, input);
            }
        }
        for axis in [1, 8, 9] {
            compare(
                simd,
                core::array::from_fn(|i| if (i / axis) % 2 == 0 { -128 } else { 127 }),
            );
        }
        compare(simd, crate::fdct::tests::INPUT1);
        compare(simd, crate::fdct::tests::INPUT2);
        let mut state = 12345;
        for _ in 0..10_000 {
            compare(simd, random_block(&mut state));
        }
    }

    #[test]
    fn matches_scalar_on_every_backend() {
        for level in levels() {
            dispatch!(level, simd => check_blocks(simd));
        }
    }

    #[test]
    fn rounds_odd_columns_in_rows_zero_and_four() {
        let input = random_block(&mut 12345);
        for level in levels() {
            let mut block = AlignedBlock::new(input);
            dispatch!(level, simd => transform(simd, &mut block));
            // The original AVX2 path gives -99, -1334 and -1228 here.
            assert_eq!(
                (block.data[1], block.data[33], block.data[37]),
                (-98, -1333, -1227)
            );
        }
    }
}
