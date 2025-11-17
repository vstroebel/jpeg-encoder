#[cfg(target_arch = "arm")] // 32-bit ARM  with NEON
use std::arch::arm::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
fn rgb_to_ycbcr_simd(
    r: [i32; 8],
    g: [i32; 8],
    b: [i32; 8],
) -> ([i32; 8], [i32; 8], [i32; 8]) {
    // To avoid floating point math this scales everything by 2^16 which gives
    // a precision of approx 4 digits.
    //
    // Non scaled conversion:
    // Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
    // Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128
    // Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B  + 128

    // Load input arrays into NEON registers (2 registers per channel)
    let r_lo = load_i32x4(r[..4]);
    let r_hi = load_i32x4(r[4..]);
    let g_lo = load_i32x4(g[..4]);
    let g_hi = load_i32x4(g[4..]);
    let b_lo = load_i32x4(b[..4]);
    let b_hi = load_i32x4(b[4..]);

    let y1_mul = vdupq_n_s32(19595);
    let y2_mul = vdupq_n_s32(38470);
    let y3_mul = vdupq_n_s32(7471);

    let cb1_mul = vdupq_n_s32(-11059);
    let cb2_mul = vdupq_n_s32(21709);
    let cb3_mul = vdupq_n_s32(32768);
    let cb4_mul = vdupq_n_s32(128 << 16);

    let cr1_mul = vdupq_n_s32(32768);
    let cr2_mul = vdupq_n_s32(27439);
    let cr3_mul = vdupq_n_s32(5329);
    let cr4_mul = vdupq_n_s32(128 << 16);

    // Process low 4 elements
    let y_lo = vmlaq_s32(
        vmlaq_s32(vmulq_s32(y1_mul, r_lo), y2_mul, g_lo),
        y3_mul,
        b_lo,
    );
    let cb_lo = vaddq_s32(
        vmlaq_s32(
            vmlsq_s32(vmulq_s32(cb1_mul, r_lo), cb2_mul, g_lo),
            cb3_mul,
            b_lo,
        ),
        cb4_mul,
    );
    let cr_lo = vaddq_s32(
        vmlsq_s32(
            vmlsq_s32(vmulq_s32(cr1_mul, r_lo), cr2_mul, g_lo),
            cr3_mul,
            b_lo,
        ),
        cr4_mul,
    );

    // Process high 4 elements
    let y_hi = vmlaq_s32(
        vmlaq_s32(vmulq_s32(y1_mul, r_hi), y2_mul, g_hi),
        y3_mul,
        b_hi,
    );
    let cb_hi = vaddq_s32(
        vmlaq_s32(
            vmlsq_s32(vmulq_s32(cb1_mul, r_hi), cb2_mul, g_hi),
            cb3_mul,
            b_hi,
        ),
        cb4_mul,
    );
    let cr_hi = vaddq_s32(
        vmlsq_s32(
            vmlsq_s32(vmulq_s32(cr1_mul, r_hi), cr2_mul, g_hi),
            cr3_mul,
            b_hi,
        ),
        cr4_mul,
    );

    #[target_feature(enable = "neon")]
    #[inline]
    fn round_shift(v: int32x4_t) -> int32x4_t {
        let round = vdupq_n_s32(0x7FFF);
        vshrq_n_s32(vaddq_s32(v, round), 16)
    }

    // Round and shift
    let y_lo = round_shift(y_lo);
    let y_hi = round_shift(y_hi);
    let cb_lo = round_shift(cb_lo);
    let cb_hi = round_shift(cb_hi);
    let cr_lo = round_shift(cr_lo);
    let cr_hi = round_shift(cr_hi);

    // Store results back to arrays
    let mut y_out = [0i32; 8];
    let mut cb_out = [0i32; 8];
    let mut cr_out = [0i32; 8];

    vst1q_s32(y_out.as_mut_ptr(), y_lo);
    vst1q_s32(y_out.as_mut_ptr().add(4), y_hi);
    vst1q_s32(cb_out.as_mut_ptr(), cb_lo);
    vst1q_s32(cb_out.as_mut_ptr().add(4), cb_hi);
    vst1q_s32(cr_out.as_mut_ptr(), cr_lo);
    vst1q_s32(cr_out.as_mut_ptr().add(4), cr_hi);

    (y_out, cb_out, cr_out)
}

#[target_feature(enable = "neon")]
fn load_i32x4(arr: &[i32; 4]) -> int32x4_t {
    // Safety preconditions. Optimized away in release mode, no runtime cost.
    assert!(core::mem::size_of::<int32x4_t>() == core::mem::size_of::<[i32; 4]>());
    // SAFETY: size checked above.
    // NEON load intrinsics do not care if data is aligned.
    // Both types are plain old data: no pointers, lifetimes, etc.
    vld1q_s32(arr.as_ptr())
}

#[target_feature(enable = "neon")]
fn store_i32x4(arr: &mut [i32], vec: int32x4_t) {
    // Safety preconditions. Optimized away in release mode, no runtime cost.
    assert!(arr.len() >= core::mem::size_of::<int32x4_t>());
    // SAFETY: size checked above.
    // NEON load intrinsics do not care if data is aligned.
    // Both types are plain old data: no pointers, lifetimes, etc.
    vst1q_s32(arr.as_mut_ptr(), vec);
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8_to_i32<const stride: usize>(values: &[u8]) -> [i32; 8] {
    // avoid bounds checks further down
    let values = &values[..7*stride + 1];

    [
        values[0 * stride] as i32,
        values[1 * stride] as i32,
        values[2 * stride] as i32,
        values[3 * stride] as i32,
        values[4 * stride] as i32,
        values[5 * stride] as i32,
        values[6 * stride] as i32,
        values[7 * stride] as i32,
    ]
}