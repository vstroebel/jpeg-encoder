use alloc::vec::Vec;

use core::simd::i32x16;
use core::simd::num::SimdInt;

use crate::{ImageBuffer, JpegColorType, rgb_to_ycbcr};



#[inline]
fn rgb_to_ycbcr_simd(r: i32x16, g: i32x16, b: i32x16) -> (i32x16, i32x16, i32x16) {
    // To avoid floating point math this scales everything by 2^16 which gives
    // a precision of approx 4 digits.
    //
    // Non scaled conversion:
    // Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
    // Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128
    // Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B  + 128

    let y1_mul: i32x16 = i32x16::splat(19595);
    let y2_mul: i32x16 = i32x16::splat(38470);
    let y3_mul: i32x16 = i32x16::splat(7471);

    let cb1_mul: i32x16 = i32x16::splat(-11059);
    let cb2_mul: i32x16 = i32x16::splat(21709);
    let cb3_mul: i32x16 = i32x16::splat(32768);
    let cb4_mul: i32x16 = i32x16::splat(128 << 16);

    let cr1_mul: i32x16 = i32x16::splat(32768);
    let cr2_mul: i32x16 = i32x16::splat(27439);
    let cr3_mul: i32x16 = i32x16::splat(5329);
    let cr4_mul: i32x16 = i32x16::splat(128 << 16);

    let y = y1_mul * r + y2_mul * g + y3_mul * b;
    let cb = cb1_mul * r - cb2_mul * g + cb3_mul * b + cb4_mul;
    let cr = cr1_mul * r - cr2_mul * g - cr3_mul * b + cr4_mul;

    fn round_shift(v: i32x16) -> i32x16 {
        (v + i32x16::splat(0x7FFF)) >> i32x16::splat(16)
    }

    (
        round_shift(y),
        round_shift(cb),
        round_shift(cr)
    )
}

#[inline(always)]
fn load(values: &[u8], offset: usize, stride: usize) -> i32x16 {
    assert!(values.len() >= (offset + 15 * stride + 1));

    i32x16::from([
        values[offset + 0 * stride] as i32,
        values[offset + 1 * stride] as i32,
        values[offset + 2 * stride] as i32,
        values[offset + 3 * stride] as i32,
        values[offset + 4 * stride] as i32,
        values[offset + 5 * stride] as i32,
        values[offset + 6 * stride] as i32,
        values[offset + 7 * stride] as i32,
        values[offset + 8 * stride] as i32,
        values[offset + 9 * stride] as i32,
        values[offset + 10 * stride] as i32,
        values[offset + 11 * stride] as i32,
        values[offset + 12 * stride] as i32,
        values[offset + 13 * stride] as i32,
        values[offset + 14 * stride] as i32,
        values[offset + 15 * stride] as i32,
    ])
}

#[inline(always)]
fn push(buffer: &mut Vec<u8>, values: i32x16) {
    buffer.extend_from_slice(&values.cast().to_array());
}


macro_rules! ycbcr_image_std_simd {
    ($name:ident, $num_colors:expr, $o1:expr, $o2:expr, $o3:expr) => {
        pub(crate) struct $name<'a>(pub &'a [u8], pub u16, pub u16);

        impl<'a> $name<'a> {
            fn fill_buffers_std_simd(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
                let num_colors = $num_colors;
                let o1 = $o1;
                let o2 = $o2;
                let o3 = $o3;

                let width = self.width() as usize;
                let y = y as usize;

                let line_start = width * y * num_colors;
                let line_end = line_start + (width * num_colors);

                if self.width() >= 16 {
                    let line = &self.0[line_start..line_end];

                    for chunk in line.chunks_exact(16 * num_colors) {
                        let r = load(chunk, o1, num_colors);
                        let g = load(chunk, o2, num_colors);
                        let b = load(chunk, o3, num_colors);

                        let (y, cb, cr) = rgb_to_ycbcr_simd(r, g, b);

                        push(&mut buffers[0], y);
                        push(&mut buffers[1], cb);
                        push(&mut buffers[2], cr);
                    }
                }

                let start = width / 16 * 16;

                for x in start..width {
                    let offset = line_start + x * num_colors;

                    let (y, cb, cr) = rgb_to_ycbcr(
                        self.0[offset + o1],
                        self.0[offset + o2],
                        self.0[offset + o3],
                    );

                    buffers[0].push(y);
                    buffers[1].push(cb);
                    buffers[2].push(cr);
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
                self.fill_buffers_std_simd(y, buffers);
            }
        }
    };
}

ycbcr_image_std_simd!(RgbImageStdSimd, 3, 0, 1, 2);
ycbcr_image_std_simd!(RgbaImageStdSimd, 4, 0, 1, 2);
ycbcr_image_std_simd!(BgrImageStdSimd, 3, 2, 1, 0);
ycbcr_image_std_simd!(BgraImageStdSimd, 4, 2, 1, 0);

/*
pub(crate) struct RgbImageStdSimd<'a> (pub &'a [u8], pub u16, pub u16);


impl<'a> RgbImageStdSimd<'a> {
    fn fill_buffers_std_simd(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
        let num_colors = 3;
        let o1 = 0;
        let o2 = 1;
        let o3 = 2;

        let width = self.width() as usize;
        let y = y as usize;

        let line_start = width * y * num_colors;
        let line_end = line_start + (width * num_colors);

        if self.width() >= 16 {
            let line = &self.0[line_start..line_end];

            for chunk in line.chunks_exact(16 * num_colors) {
                let r = load(chunk, o1, num_colors);
                let g = load(chunk, o2, num_colors);
                let b = load(chunk, o3, num_colors);

                let (y, cb, cr) = rgb_to_ycbcr_simd(r, g, b);

                push(&mut buffers[0], y);
                push(&mut buffers[1], cb);
                push(&mut buffers[2], cr);
            }
        }

        let start = width / 16 * 16;

        for x in start..width {
            let offset = line_start + x * num_colors;

            let (y, cb, cr) = rgb_to_ycbcr(
                self.0[offset + o1],
                self.0[offset + o2],
                self.0[offset + o3],
            );

            buffers[0].push(y);
            buffers[1].push(cb);
            buffers[2].push(cr);
        }
    }
}

impl<'a> ImageBuffer for RgbImageStdSimd<'a> {
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
        self.fill_buffers_std_simd(y, buffers);
    }
}


 */

#[cfg(test)]
mod tests {
    use core::simd::i32x16;
    use super::rgb_to_ycbcr_simd;

    fn assert_rgb_to_ycbcr(rgb: [u8; 3], ycbcr: [u8; 3]) {
        let r = i32x16::splat(rgb[0] as i32);
        let g = i32x16::splat(rgb[1] as i32);
        let b = i32x16::splat(rgb[2] as i32);

        let y_i = i32x16::splat(ycbcr[0] as i32);
        let cb_i = i32x16::splat(ycbcr[1] as i32);
        let cr_i = i32x16::splat(ycbcr[2] as i32);

        let (y_o, cb_o, cr_o) = rgb_to_ycbcr_simd(r, g, b);
        assert_eq!(y_i, y_o);
        assert_eq!(cb_i, cb_o);
        assert_eq!(cr_i, cr_o);
    }

    #[test]
    fn test_rgb_to_ycbcr() {
        assert_rgb_to_ycbcr([0, 0, 0], [0, 128, 128]);
        assert_rgb_to_ycbcr([255, 255, 255], [255, 128, 128]);
        assert_rgb_to_ycbcr([255, 0, 0], [76, 85, 255]);
        assert_rgb_to_ycbcr([0, 255, 0], [150, 44, 21]);
        assert_rgb_to_ycbcr([0, 0, 255], [29, 255, 107]);

        // Values taken from libjpeg for a common image

        assert_rgb_to_ycbcr([59, 109, 6], [82, 85, 111]);
        assert_rgb_to_ycbcr([29, 60, 11], [45, 109, 116]);
        assert_rgb_to_ycbcr([57, 114, 26], [87, 94, 107]);
        assert_rgb_to_ycbcr([30, 60, 6], [45, 106, 117]);
        assert_rgb_to_ycbcr([41, 75, 11], [58, 102, 116]);
        assert_rgb_to_ycbcr([145, 184, 108], [164, 97, 115]);
        assert_rgb_to_ycbcr([33, 85, 7], [61, 98, 108]);
        assert_rgb_to_ycbcr([61, 90, 40], [76, 108, 118]);
        assert_rgb_to_ycbcr([75, 127, 45], [102, 96, 109]);
        assert_rgb_to_ycbcr([30, 56, 14], [43, 111, 118]);
        assert_rgb_to_ycbcr([106, 142, 81], [124, 104, 115]);
        assert_rgb_to_ycbcr([35, 59, 11], [46, 108, 120]);
        assert_rgb_to_ycbcr([170, 203, 123], [184, 94, 118]);
        assert_rgb_to_ycbcr([45, 87, 16], [66, 100, 113]);
        assert_rgb_to_ycbcr([59, 109, 21], [84, 92, 110]);
        assert_rgb_to_ycbcr([100, 167, 36], [132, 74, 105]);
        assert_rgb_to_ycbcr([17, 53, 5], [37, 110, 114]);
        assert_rgb_to_ycbcr([226, 244, 220], [236, 119, 121]);
        assert_rgb_to_ycbcr([192, 214, 120], [197, 85, 125]);
        assert_rgb_to_ycbcr([63, 107, 22], [84, 93, 113]);
        assert_rgb_to_ycbcr([44, 78, 19], [61, 104, 116]);
        assert_rgb_to_ycbcr([72, 106, 54], [90, 108, 115]);
        assert_rgb_to_ycbcr([99, 123, 73], [110, 107, 120]);
        assert_rgb_to_ycbcr([188, 216, 148], [200, 99, 120]);
        assert_rgb_to_ycbcr([19, 46, 7], [33, 113, 118]);
        assert_rgb_to_ycbcr([56, 95, 40], [77, 107, 113]);
        assert_rgb_to_ycbcr([81, 120, 56], [101, 103, 114]);
        assert_rgb_to_ycbcr([9, 30, 0], [20, 117, 120]);
        assert_rgb_to_ycbcr([90, 118, 46], [101, 97, 120]);
        assert_rgb_to_ycbcr([24, 52, 0], [38, 107, 118]);
        assert_rgb_to_ycbcr([32, 69, 9], [51, 104, 114]);
        assert_rgb_to_ycbcr([74, 134, 33], [105, 88, 106]);
        assert_rgb_to_ycbcr([37, 74, 7], [55, 101, 115]);
        assert_rgb_to_ycbcr([69, 119, 31], [94, 92, 110]);
        assert_rgb_to_ycbcr([63, 112, 21], [87, 91, 111]);
        assert_rgb_to_ycbcr([90, 148, 17], [116, 72, 110]);
        assert_rgb_to_ycbcr([50, 97, 30], [75, 102, 110]);
        assert_rgb_to_ycbcr([99, 129, 72], [114, 105, 118]);
        assert_rgb_to_ycbcr([161, 196, 57], [170, 64, 122]);
        assert_rgb_to_ycbcr([10, 26, 1], [18, 118, 122]);
        assert_rgb_to_ycbcr([87, 128, 68], [109, 105, 112]);
        assert_rgb_to_ycbcr([111, 155, 73], [132, 94, 113]);
        assert_rgb_to_ycbcr([33, 75, 11], [55, 103, 112]);
        assert_rgb_to_ycbcr([70, 122, 51], [98, 101, 108]);
        assert_rgb_to_ycbcr([22, 74, 3], [50, 101, 108]);
        assert_rgb_to_ycbcr([88, 142, 45], [115, 89, 109]);
        assert_rgb_to_ycbcr([66, 107, 40], [87, 101, 113]);
        assert_rgb_to_ycbcr([18, 45, 0], [32, 110, 118]);
        assert_rgb_to_ycbcr([163, 186, 88], [168, 83, 124]);
        assert_rgb_to_ycbcr([47, 104, 4], [76, 88, 108]);
        assert_rgb_to_ycbcr([147, 211, 114], [181, 90, 104]);
        assert_rgb_to_ycbcr([42, 77, 18], [60, 104, 115]);
        assert_rgb_to_ycbcr([37, 72, 6], [54, 101, 116]);
        assert_rgb_to_ycbcr([84, 140, 55], [114, 95, 107]);
        assert_rgb_to_ycbcr([46, 98, 25], [74, 100, 108]);
        assert_rgb_to_ycbcr([48, 97, 20], [74, 98, 110]);
        assert_rgb_to_ycbcr([189, 224, 156], [206, 100, 116]);
        assert_rgb_to_ycbcr([36, 83, 0], [59, 94, 111]);
        assert_rgb_to_ycbcr([159, 186, 114], [170, 97, 120]);
        assert_rgb_to_ycbcr([75, 118, 46], [97, 99, 112]);
        assert_rgb_to_ycbcr([193, 233, 158], [212, 97, 114]);
        assert_rgb_to_ycbcr([76, 116, 48], [96, 101, 114]);
        assert_rgb_to_ycbcr([108, 157, 79], [133, 97, 110]);
        assert_rgb_to_ycbcr([180, 208, 155], [194, 106, 118]);
        assert_rgb_to_ycbcr([74, 126, 53], [102, 100, 108]);
        assert_rgb_to_ycbcr([72, 123, 46], [99, 98, 109]);
        assert_rgb_to_ycbcr([71, 123, 34], [97, 92, 109]);
        assert_rgb_to_ycbcr([130, 184, 72], [155, 81, 110]);
        assert_rgb_to_ycbcr([30, 61, 17], [47, 111, 116]);
        assert_rgb_to_ycbcr([27, 71, 0], [50, 100, 112]);
        assert_rgb_to_ycbcr([45, 73, 24], [59, 108, 118]);
        assert_rgb_to_ycbcr([139, 175, 93], [155, 93, 117]);
        assert_rgb_to_ycbcr([11, 38, 0], [26, 114, 118]);
        assert_rgb_to_ycbcr([34, 87, 15], [63, 101, 107]);
        assert_rgb_to_ycbcr([43, 76, 35], [61, 113, 115]);
        assert_rgb_to_ycbcr([18, 35, 7], [27, 117, 122]);
        assert_rgb_to_ycbcr([69, 97, 48], [83, 108, 118]);
        assert_rgb_to_ycbcr([139, 176, 50], [151, 71, 120]);
        assert_rgb_to_ycbcr([21, 51, 7], [37, 111, 117]);
        assert_rgb_to_ycbcr([209, 249, 189], [230, 105, 113]);
        assert_rgb_to_ycbcr([32, 66, 14], [50, 108, 115]);
        assert_rgb_to_ycbcr([100, 143, 67], [121, 97, 113]);
        assert_rgb_to_ycbcr([40, 96, 14], [70, 96, 107]);
        assert_rgb_to_ycbcr([88, 130, 64], [110, 102, 112]);
        assert_rgb_to_ycbcr([52, 112, 14], [83, 89, 106]);
        assert_rgb_to_ycbcr([49, 72, 25], [60, 108, 120]);
        assert_rgb_to_ycbcr([144, 193, 75], [165, 77, 113]);
        assert_rgb_to_ycbcr([49, 94, 1], [70, 89, 113]);
    }
}