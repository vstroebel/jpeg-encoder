#![allow(clippy::identity_op)]

use crate::encoder::JpegColorType;

/// Conversion from RGB to YCbCr
#[inline]
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {

    // To avoid floating point math this scales everything by 2^16 which gives
    // a precision of approx 4 digits.
    //
    // Non scaled conversion:
    // Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
    // Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128
    // Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B  + 128

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

/// Conversion from CMYK to YCCK (YCbCrK)
#[inline]
pub fn cmyk_to_ycck(c: u8, m: u8, y: u8, k: u8) -> (u8, u8, u8, u8) {
    let (y, cb, cr) = rgb_to_ycbcr(c, m, y);
    (y, cb, cr, 255 - k)
}

/// # Buffer used as input value for image encoding
///
/// Image encoding with [Encoder::encode_image](crate::Encoder::encode_image) needs an ImageBuffer
/// as input for the image data. For convenience the [Encoder::encode](crate::Encoder::encode)
/// function contains implementaions for common byte based pixel formats.
/// Users that needs other pixel formats or don't have the data available as byte slices
/// can create their own buffer implementations.
///
/// ## Example: ImageBuffer implementation for RgbImage from the `image` crate
/// ```no_run
/// use image::RgbImage;
/// use jpeg_encoder::{ImageBuffer, JpegColorType, rgb_to_ycbcr};
///
/// pub struct RgbImageBuffer {
///     image: RgbImage,
/// }
///
/// impl ImageBuffer for RgbImageBuffer {
///     fn get_jpeg_color_type(&self) -> JpegColorType {
///         // Rgb images are encoded as YCbCr in JFIF files
///         JpegColorType::Ycbcr
///     }
///
///     fn width(&self) -> u16 {
///         self.image.width() as u16
///     }
///
///     fn height(&self) -> u16 {
///         self.image.height() as u16
///     }
///
///     fn fill_buffers(&self, x: u16, y: u16, buffers: &mut [Vec<u8>; 4]){
///         let pixel = self.image.get_pixel(x as u32 ,y as u32);
///
///         let (y,cb,cr) = rgb_to_ycbcr(pixel[0], pixel[1], pixel[2]);
///
///         // For YCbCr the 4th buffer is not used
///         buffers[0].push(y);
///         buffers[1].push(cb);
///         buffers[2].push(cr);
///     }
/// }
///
/// ```
pub trait ImageBuffer {
    /// The color type used in the image encoding
    fn get_jpeg_color_type(&self) -> JpegColorType;

    /// Width of the image
    fn width(&self) -> u16;

    /// Height of the image
    fn height(&self) -> u16;

    /// Add color values for the position to color component buffers
    fn fill_buffers(&self, x: u16, y: u16, buffers: &mut [Vec<u8>; 4]);
}

pub(crate) struct GrayImage<'a>(pub &'a [u8], pub u16, pub u16);

impl<'a> ImageBuffer for GrayImage<'a> {
    fn get_jpeg_color_type(&self) -> JpegColorType {
        JpegColorType::Luma
    }

    fn width(&self) -> u16 {
        self.1
    }

    fn height(&self) -> u16 {
        self.2
    }

    fn fill_buffers(&self, x: u16, y: u16, buffers: &mut [Vec<u8>; 4]) {
        let offset = usize::from(y) * usize::from(self.1) + usize::from(x);

        buffers[0].push(self.0[offset + 0]);
    }
}

macro_rules! ycbcr_image {
    ($name:ident, $num_colors:expr, $o1:expr, $o2:expr, $o3:expr) => {
        pub(crate) struct $name<'a>(pub &'a [u8], pub u16, pub u16);

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
            fn fill_buffers(&self, x: u16, y: u16, buffers: &mut [Vec<u8>; 4]) {
                let offset = (usize::from(y) * usize::from(self.1) + usize::from(x)) * $num_colors;
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

pub(crate) struct YCbCrImage<'a>(pub &'a [u8], pub u16, pub u16);

impl<'a> ImageBuffer for YCbCrImage<'a> {
    fn get_jpeg_color_type(&self) -> JpegColorType {
        JpegColorType::Ycbcr
    }

    fn width(&self) -> u16 {
        self.1
    }

    fn height(&self) -> u16 {
        self.2
    }

    fn fill_buffers(&self, x: u16, y: u16, buffers: &mut [Vec<u8>; 4]) {
        let offset = (usize::from(y) * usize::from(self.1) + usize::from(x)) * 3;

        buffers[0].push(self.0[offset + 0]);
        buffers[1].push(self.0[offset + 1]);
        buffers[2].push(self.0[offset + 2]);
    }
}

pub(crate) struct CmykImage<'a>(pub &'a [u8], pub u16, pub u16);

impl<'a> ImageBuffer for CmykImage<'a> {
    fn get_jpeg_color_type(&self) -> JpegColorType {
        JpegColorType::Cmyk
    }

    fn width(&self) -> u16 {
        self.1
    }

    fn height(&self) -> u16 {
        self.2
    }

    fn fill_buffers(&self, x: u16, y: u16, buffers: &mut [Vec<u8>; 4]) {
        let offset = (usize::from(y) * usize::from(self.1) + usize::from(x)) * 4;

        buffers[0].push(255 - self.0[offset + 0]);
        buffers[1].push(255 - self.0[offset + 1]);
        buffers[2].push(255 - self.0[offset + 2]);
        buffers[3].push(255 - self.0[offset + 3]);
    }
}

pub(crate) struct CmykAsYcckImage<'a>(pub &'a [u8], pub u16, pub u16);

impl<'a> ImageBuffer for CmykAsYcckImage<'a> {
    fn get_jpeg_color_type(&self) -> JpegColorType {
        JpegColorType::Ycck
    }

    fn width(&self) -> u16 {
        self.1
    }

    fn height(&self) -> u16 {
        self.2
    }

    fn fill_buffers(&self, x: u16, y: u16, buffers: &mut [Vec<u8>; 4]) {
        let offset = (usize::from(y) * usize::from(self.1) + usize::from(x)) * 4;

        let (y, cb, cr, k) = cmyk_to_ycck(
            self.0[offset + 0],
            self.0[offset + 1],
            self.0[offset + 2],
            self.0[offset + 3]);

        buffers[0].push(y);
        buffers[1].push(cb);
        buffers[2].push(cr);
        buffers[3].push(k);
    }
}

pub(crate) struct YcckImage<'a>(pub &'a [u8], pub u16, pub u16);

impl<'a> ImageBuffer for YcckImage<'a> {
    fn get_jpeg_color_type(&self) -> JpegColorType {
        JpegColorType::Ycck
    }

    fn width(&self) -> u16 {
        self.1
    }

    fn height(&self) -> u16 {
        self.2
    }

    fn fill_buffers(&self, x: u16, y: u16, buffers: &mut [Vec<u8>; 4]) {
        let offset = (usize::from(y) * usize::from(self.1) + usize::from(x)) * 4;

        buffers[0].push(self.0[offset + 0]);
        buffers[1].push(self.0[offset + 1]);
        buffers[2].push(self.0[offset + 2]);
        buffers[3].push(self.0[offset + 3]);
    }
}