#[cfg(target_arch = "arm")] // 32-bit ARM  with NEON
use std::arch::arm::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use alloc::vec::Vec;

use crate::{rgb_to_ycbcr, ImageBuffer, JpegColorType};

macro_rules! ycbcr_image_neon {
    ($name:ident, $num_colors:expr, $o1:expr, $o2:expr, $o3:expr) => {
        pub(crate) struct $name<'a>(pub &'a [u8], pub u16, pub u16);

        impl<'a> $name<'a> {
            #[target_feature(enable = "neon")]
            fn fill_buffers_neon(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
                #[inline]
                #[target_feature(enable = "neon")]
                fn load3(data: &[u8], offset: usize) -> [i32; 8] {
                    load_channel::<3>(data, offset)
                }

                let [y_buffer, cb_buffer, cr_buffer, _] = buffers;
                y_buffer.reserve(self.width() as usize);
                cb_buffer.reserve(self.width() as usize);
                cr_buffer.reserve(self.width() as usize);

                let mut data = &self.0[(y as usize * self.1 as usize * $num_colors)..];

                for _ in 0..self.width() / 8 {
                    let r = load3(&data[$o1..]);
                    let g = load3(&data[$o2..]);
                    let b = load3(&data[$o3..]);

                    data = &data[($num_colors * 8)..];

                    let (y, cb, cr) = rgb_to_ycbcr_simd(r, g, b);

                    let y: [u8; 8] = y.map(|x| x as u8);
                    y_buffer.extend_from_slice(&y);

                    let cb: [u8; 8] = cb.map(|x| x as u8);
                    cb_buffer.extend_from_slice(&cb);

                    let cr: [u8; 8] = cr.map(|x| x as u8);
                    cr_buffer.extend_from_slice(&cr);
                }

                for _ in 0..self.width() % 8 {
                    let (y, cb, cr) = rgb_to_ycbcr(data[$o1], data[$o2], data[$o3]);
                    data = &data[$num_colors..];

                    y_buffer.push(y);
                    cb_buffer.push(cb);
                    cr_buffer.push(cr);
                }
            }
        }

        impl<'a> ImageBuffer for $name<'a> {
            fn get_jpeg_color_type(&self) -> JpegColorType {
                JpegColorType::Ycbcr
            }

            fn width(&self) -> u16 {
                self.1
            }

            fn height(&self) -> u16 {
                self.2
            }

            #[inline(always)]
            fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
                unsafe {
                    self.fill_buffers_neon(y, buffers);
                }
            }
        }
    };
}

ycbcr_image_neon!(RgbImageNeon, 3, 0, 1, 2);
ycbcr_image_neon!(RgbaImageNeon, 4, 0, 1, 2);
ycbcr_image_neon!(BgrImageNeon, 3, 2, 1, 0);
ycbcr_image_neon!(BgraImageNeon, 4, 2, 1, 0);

#[target_feature(enable = "neon")]
fn rgb_to_ycbcr_simd(r: [i32; 8], g: [i32; 8], b: [i32; 8]) -> ([i32; 8], [i32; 8], [i32; 8]) {
    // To avoid floating point math this scales everything by 2^16 which gives
    // a precision of approx 4 digits.
    //
    // Non scaled conversion:
    // Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
    // Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128
    // Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B  + 128

    // Load input arrays into NEON registers (2 registers per channel)
    let r_lo = load_i32x4(r[..4].try_into().unwrap());
    let r_hi = load_i32x4(r[4..].try_into().unwrap());
    let g_lo = load_i32x4(g[..4].try_into().unwrap());
    let g_hi = load_i32x4(g[4..].try_into().unwrap());
    let b_lo = load_i32x4(b[..4].try_into().unwrap());
    let b_hi = load_i32x4(b[4..].try_into().unwrap());

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

    // TODO: refactor into safe stores
    unsafe {
        vst1q_s32(y_out.as_mut_ptr(), y_lo);
        vst1q_s32(y_out.as_mut_ptr().add(4), y_hi);
        vst1q_s32(cb_out.as_mut_ptr(), cb_lo);
        vst1q_s32(cb_out.as_mut_ptr().add(4), cb_hi);
        vst1q_s32(cr_out.as_mut_ptr(), cr_lo);
        vst1q_s32(cr_out.as_mut_ptr().add(4), cr_hi);
    }

    (y_out, cb_out, cr_out)
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i32x4(arr: &[i32; 4]) -> int32x4_t {
    // Safety preconditions. Optimized away in release mode, no runtime cost.
    assert!(core::mem::size_of::<int32x4_t>() == core::mem::size_of::<[i32; 4]>());
    // SAFETY: size checked above.
    // NEON load intrinsics do not care if data is aligned.
    // Both types are plain old data: no pointers, lifetimes, etc.
    unsafe { vld1q_s32(arr.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x4(arr: &mut [i32], vec: int32x4_t) {
    // Safety preconditions. Optimized away in release mode, no runtime cost.
    assert!(arr.len() >= core::mem::size_of::<int32x4_t>());
    // SAFETY: size checked above.
    // NEON load intrinsics do not care if data is aligned.
    // Both types are plain old data: no pointers, lifetimes, etc.
    unsafe {
        vst1q_s32(arr.as_mut_ptr(), vec);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_channel<const STRIDE: usize>(values: &[u8]) -> [i32; 8] {
    // avoid bounds checks further down
    let values = &values[..7 * STRIDE + 1];

    [
        values[0 * STRIDE] as i32,
        values[1 * STRIDE] as i32,
        values[2 * STRIDE] as i32,
        values[3 * STRIDE] as i32,
        values[4 * STRIDE] as i32,
        values[5 * STRIDE] as i32,
        values[6 * STRIDE] as i32,
        values[7 * STRIDE] as i32,
    ]
}
