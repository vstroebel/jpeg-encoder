/// Conversion from RGB to YCbCr
///
/// To avoid floating point math this scales everything by 2^16 which gives
/// a precision of approx 4 digits.
///
/// Non scaled conversion:
/// Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
/// Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128
/// Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B  + 128
///
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;

    let y = 19595 * r + 38470 * g + 7471 * b;
    let cb = -11059 * r - 21709 * g + 32768 * b + (128 << 16);
    let cr = 32768 * r - 27439 * g - 5329 * b + (128 << 16);

    let y = y >> 16;
    let cb = cb >> 16;
    let cr = cr >> 16;

    (y as u8, cb as u8, cr as u8)
}

pub trait ImageBuffer {
    fn width(&self) -> u32;

    fn height(&self) -> u32;

    fn fill_buffers(&self, x: u32, y: u32, buffers: &mut [Vec<u8>; 4]);
}

pub(crate) struct GrayImage<'a>(pub &'a [u8], pub u32, pub u32);

impl<'a> ImageBuffer for GrayImage<'a> {
    fn width(&self) -> u32 {
        self.1
    }

    fn height(&self) -> u32 {
        self.2
    }

    fn fill_buffers(&self, x: u32, y: u32, buffers: &mut [Vec<u8>; 4]) {
        let x = x.min(self.1 as u32 - 1);
        let y = y.min(self.2 as u32 - 1);

        let offset = (y * self.1 + x) as usize;

        buffers[0].push(self.0[offset + 0]);
    }
}

macro_rules! ycbcr_image {
    ($name:ident, $num_colors:expr, $o1:expr, $o2:expr, $o3:expr) => {
        pub(crate) struct $name<'a>(pub &'a [u8], pub u32, pub u32);

        impl<'a> ImageBuffer for $name<'a> {
            fn width(&self) -> u32 {
                self.1
            }

            fn height(&self) -> u32 {
                self.2
            }

            #[inline(always)]
            fn fill_buffers(&self, x: u32, y: u32, buffers: &mut [Vec<u8>; 4]) {
                let x = x.min(self.1 as u32 - 1);
                let y = y.min(self.2 as u32 - 1);

                let offset = (y * self.1 + x) as usize * $num_colors;
                let (y, cb, cr) = rgb_to_ycbcr(self.0[offset + $o1], self.0[offset + $o2], self.0[offset + $o3]);

                buffers[0].push(y);
                buffers[1].push(cb);
                buffers[2].push(cr);
            }
        }
    }
}

ycbcr_image!(RgbImage, 3, 0, 1, 2);
ycbcr_image!(RgbaImage, 4, 0, 1, 2);
ycbcr_image!(BgrImage, 3, 2, 1, 0);
ycbcr_image!(BgraImage, 4, 2, 1, 0);

pub(crate) struct YCbCrImage<'a>(pub &'a [u8], pub u32, pub u32);

impl<'a> ImageBuffer for YCbCrImage<'a> {
    fn width(&self) -> u32 {
        self.1
    }

    fn height(&self) -> u32 {
        self.2
    }

    fn fill_buffers(&self, x: u32, y: u32, buffers: &mut [Vec<u8>; 4]) {
        let x = x.min(self.1 as u32 - 1);
        let y = y.min(self.2 as u32 - 1);

        let offset = (y * self.1 + x) as usize * 3;

        buffers[0].push(self.0[offset + 0]);
        buffers[1].push(self.0[offset + 1]);
        buffers[2].push(self.0[offset + 2]);
    }
}

pub(crate) struct CmykImage<'a>(pub &'a [u8], pub u32, pub u32);

impl<'a> ImageBuffer for CmykImage<'a> {
    fn width(&self) -> u32 {
        self.1
    }

    fn height(&self) -> u32 {
        self.2
    }

    fn fill_buffers(&self, x: u32, y: u32, buffers: &mut [Vec<u8>; 4]) {
        let x = x.min(self.1 as u32 - 1);
        let y = y.min(self.2 as u32 - 1);

        let offset = (y * self.1 + x) as usize * 4;

        buffers[0].push(self.0[offset + 0]);
        buffers[1].push(self.0[offset + 1]);
        buffers[2].push(self.0[offset + 2]);
        buffers[3].push(self.0[offset + 3]);
    }
}
