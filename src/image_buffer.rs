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

pub struct ImageBuffer<'a> {
    pub data: &'a [u8],
    pub width: u32,
    pub height: u32,
}

impl<'a> ImageBuffer<'a> {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn fill_buffers(&self, x: u32, y: u32, buffers: &mut [Vec<u8>; 3]) {
        let x = x.min(self.width as u32 - 1);
        let y = y.min(self.height as u32 - 1);

        let offset = (y * self.width + x) as usize * 3;
        let (y, cb, cr) = rgb_to_ycbcr(
            self.data[offset],
            self.data[offset + 1],
            self.data[offset + 2],
        );

        buffers[0].push(y);
        buffers[1].push(cb);
        buffers[2].push(cr);
    }
}