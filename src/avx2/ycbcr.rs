#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256i, _mm256_add_epi32, _mm256_mullo_epi32, _mm256_set1_epi32, _mm256_set_epi32,
    _mm256_srli_epi32, _mm256_sub_epi32,
};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_mullo_epi32, _mm256_set1_epi32, _mm256_set_epi32,
    _mm256_srli_epi32, _mm256_sub_epi32,
};

use alloc::vec::Vec;

use crate::{rgb_to_ycbcr, ImageBuffer, JpegColorType};

macro_rules! ycbcr_image_avx2 {
    ($name:ident, $num_colors:expr, $o1:expr, $o2:expr, $o3:expr) => {
        pub(crate) struct $name<'a>(pub &'a [u8], pub u16, pub u16);

        impl<'a> $name<'a> {
            #[target_feature(enable = "avx2")]
            fn fill_buffers_avx2(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
                #[inline]
                #[target_feature(enable = "avx2")]
                fn load3(data: &[u8]) -> __m256i {
                    _ = data[7 * $num_colors]; // dummy indexing operation up front to avoid bounds checks later
                    _mm256_set_epi32(
                        data[0] as i32,
                        data[1 * $num_colors] as i32,
                        data[2 * $num_colors] as i32,
                        data[3 * $num_colors] as i32,
                        data[4 * $num_colors] as i32,
                        data[5 * $num_colors] as i32,
                        data[6 * $num_colors] as i32,
                        data[7 * $num_colors] as i32,
                    )
                }

                #[inline]
                #[target_feature(enable = "avx2")]
                fn avx_as_i32_array(data: __m256i) -> [i32; 8] {
                    // Safety preconditions. Optimized away in release mode, no runtime cost.
                    assert!(core::mem::size_of::<__m256i>() == core::mem::size_of::<[i32; 8]>());
                    assert!(core::mem::align_of::<__m256i>() >= core::mem::align_of::<[i32; 8]>());
                    // SAFETY: size and alignment preconditions checked above.
                    // Both types are plain old data: no pointers, lifetimes, etc.
                    unsafe { core::mem::transmute(data) }
                }

                let [y_buffer, cb_buffer, cr_buffer, _] = buffers;
                y_buffer.reserve(self.width() as usize);
                cb_buffer.reserve(self.width() as usize);
                cr_buffer.reserve(self.width() as usize);

                let ymulr = _mm256_set1_epi32(19595);
                let ymulg = _mm256_set1_epi32(38470);
                let ymulb = _mm256_set1_epi32(7471);

                let cbmulr = _mm256_set1_epi32(-11059);
                let cbmulg = _mm256_set1_epi32(21709);
                let cbmulb = _mm256_set1_epi32(32768);

                let crmulr = _mm256_set1_epi32(32768);
                let crmulg = _mm256_set1_epi32(27439);
                let crmulb = _mm256_set1_epi32(5329);

                let mut data = &self.0[(y as usize * self.1 as usize * $num_colors)..];

                for _ in 0..self.width() / 8 {
                    let r = load3(&data[$o1..]);
                    let g = load3(&data[$o2..]);
                    let b = load3(&data[$o3..]);

                    data = &data[($num_colors * 8)..];

                    let yr = _mm256_mullo_epi32(ymulr, r);
                    let yg = _mm256_mullo_epi32(ymulg, g);
                    let yb = _mm256_mullo_epi32(ymulb, b);

                    let y = _mm256_add_epi32(_mm256_add_epi32(yr, yg), yb);
                    let y = _mm256_add_epi32(y, _mm256_set1_epi32(0x7FFF));
                    let y = _mm256_srli_epi32(y, 16);
                    let y: [i32; 8] = avx_as_i32_array(y);
                    let mut y: [u8; 8] = y.map(|x| x as u8);
                    y.reverse();
                    y_buffer.extend_from_slice(&y);

                    let cbr = _mm256_mullo_epi32(cbmulr, r);
                    let cbg = _mm256_mullo_epi32(cbmulg, g);
                    let cbb = _mm256_mullo_epi32(cbmulb, b);

                    let cb = _mm256_add_epi32(_mm256_sub_epi32(cbr, cbg), cbb);
                    let cb = _mm256_add_epi32(cb, _mm256_set1_epi32(128 << 16));
                    let cb = _mm256_add_epi32(cb, _mm256_set1_epi32(0x7FFF));
                    let cb = _mm256_srli_epi32(cb, 16);
                    let cb: [i32; 8] = avx_as_i32_array(cb);
                    let mut cb: [u8; 8] = cb.map(|x| x as u8);
                    cb.reverse();
                    cb_buffer.extend_from_slice(&cb);

                    let crr = _mm256_mullo_epi32(crmulr, r);
                    let crg = _mm256_mullo_epi32(crmulg, g);
                    let crb = _mm256_mullo_epi32(crmulb, b);

                    let cr = _mm256_sub_epi32(_mm256_sub_epi32(crr, crg), crb);
                    let cr = _mm256_add_epi32(cr, _mm256_set1_epi32(128 << 16));
                    let cr = _mm256_add_epi32(cr, _mm256_set1_epi32(0x7FFF));
                    let cr = _mm256_srli_epi32(cr, 16);
                    let cr: [i32; 8] = avx_as_i32_array(cr);
                    let mut cr: [u8; 8] = cr.map(|x| x as u8);
                    cr.reverse();
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
                    self.fill_buffers_avx2(y, buffers);
                }
            }
        }
    };
}

ycbcr_image_avx2!(RgbImageAVX2, 3, 0, 1, 2);
ycbcr_image_avx2!(RgbaImageAVX2, 4, 0, 1, 2);
ycbcr_image_avx2!(BgrImageAVX2, 3, 2, 1, 0);
ycbcr_image_avx2!(BgraImageAVX2, 4, 2, 1, 0);
