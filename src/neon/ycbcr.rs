#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[unsafe(no_mangle)]
#[inline(never)]
unsafe fn rgb_to_ycbcr_simd(r: int32x4_t, g: int32x4_t, b: int32x4_t) -> (int32x4_t, int32x4_t, int32x4_t) {
    // To avoid floating point math this scales everything by 2^16 which gives
    // a precision of approx 4 digits.
    //
    // Non scaled conversion:
    // Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
    // Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128
    // Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B  + 128

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

    // Y = y1_mul * r + y2_mul * g + y3_mul * b
    let y = vmlaq_s32(vmlaq_s32(vmulq_s32(y1_mul, r), y2_mul, g), y3_mul, b);
    
    // Cb = cb1_mul * r - cb2_mul * g + cb3_mul * b + cb4_mul
    let cb = vaddq_s32(
        vmlaq_s32(vmlsq_s32(vmulq_s32(cb1_mul, r), cb2_mul, g), cb3_mul, b),
        cb4_mul
    );
    
    // Cr = cr1_mul * r - cr2_mul * g - cr3_mul * b + cr4_mul
    let cr = vaddq_s32(
        vmlsq_s32(vmlsq_s32(vmulq_s32(cr1_mul, r), cr2_mul, g), cr3_mul, b),
        cr4_mul
    );

    #[inline(always)]
    unsafe fn round_shift(v: int32x4_t) -> int32x4_t {
        let round = vdupq_n_s32(0x7FFF);
        vshrq_n_s32(vaddq_s32(v, round), 16)
    }

    (
        round_shift(y),
        round_shift(cb),
        round_shift(cr)
    )
}