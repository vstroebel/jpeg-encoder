//! Portable RGB to YCbCr conversion, using the same fixed-point arithmetic as AVX2.

use alloc::vec::Vec;
use fearless_simd::{Level, dispatch, i32x8, prelude::*, u8x16, u32x8};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use fearless_simd::{i32x16, u8x64, u32x16};

use crate::{ImageBuffer, JpegColorType, rgb_to_ycbcr};

fn level() -> Level {
    Level::try_detect().unwrap_or_else(Level::baseline)
}

#[inline(always)]
fn load<S: Simd, const N: usize, const OFFSET: usize>(simd: S, data: &[u8]) -> i32x8<S> {
    // Two loads cover exactly eight pixels, overlapping by eight bytes for RGB.
    // Shuffling within 128-bit blocks avoids AVX2 cross-lane byte permutations.
    let low = u8x16::from_slice(simd, &data[..16]);
    let high = u8x16::from_slice(simd, &data[8 * N - 16..]);
    let pixels = low.combine(high);
    let indices = core::array::from_fn::<_, 32, _>(|i| {
        let offset = if i < 16 { 0 } else { 16 - 4 * N };
        (offset + (i % 16 / 4) * N + OFFSET) as u8
    });
    // Repeat each selected byte across a u32, then mask to zero-extend it.
    // This works on either endianness and lets LLVM fold the mask into PSHUFB.
    let channel: i32x8<S> = pixels.swizzle_dyn_within_blocks(indices).bitcast();
    channel & 255
}

#[inline(always)]
fn pack<S: Simd>(data: i32x8<S>) -> [u8; 8] {
    // Rounded Y/Cb/Cr are all in 0..=255. Keep narrowing in SIMD: extracting
    // lanes and casting them individually generates scalar shifts and inserts.
    let data: u32x8<S> = (data >> 16).bitcast();
    let (low, high) = data.split();
    let words = low.relaxed_narrow(high);
    let bytes = words.relaxed_narrow(words);
    <[u8; 16]>::from(bytes)[..8].try_into().unwrap()
}

// AVX-512 VBMI selects bytes across the entire 64-byte vector, allowing
// sixteen RGB pixels per batch without overlapping loads or lane rearrangement.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn load16<S: Simd, const N: usize, const OFFSET: usize>(simd: S, data: &[u8]) -> i32x16<S> {
    // RGB has only 48 bytes. Copy exactly those bytes and leave the rest zero;
    // never read past the chunk, including at the end of the image. LLVM keeps
    // this in registers (a 32-byte load plus a 16-byte load on x86).
    let mut padded = [0u8; 64];
    padded[..16 * N].copy_from_slice(&data[..16 * N]);
    let pixels = u8x64::from_slice(simd, &padded);
    if N == 4 {
        // RGBA/BGRA already occupy one little-endian u32 per pixel on x86.
        let words: i32x16<S> = pixels.bitcast();
        return (words >> (OFFSET * 8) as u32) & 255;
    }
    // Selecting a zero padding byte for the upper three bytes also widens
    // each color channel to i32, using just one VPERMB per channel.
    let indices = core::array::from_fn::<_, 64, _>(|i| {
        if i % 4 == 0 {
            ((i / 4) * N + OFFSET) as u8
        } else {
            63
        }
    });
    pixels.swizzle_dyn(indices).bitcast()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn pack16<S: Simd>(data: i32x16<S>) -> [u8; 16] {
    let data: u32x16<S> = (data >> 16).bitcast();
    let (low, high) = data.split();
    let words = low.relaxed_narrow(high);
    let (low, high) = words.split();
    low.relaxed_narrow(high).into()
}

// Inline into dispatch!'s target-feature context, as in the portable FDCT.
#[inline(always)]
fn convert<S: Simd, const N: usize, const R: usize, const G: usize, const B: usize>(
    simd: S,
    line: &[u8],
    buffers: &mut [Vec<u8>; 4],
) {
    let [ys, cbs, crs, _] = buffers;
    let width = line.len() / N;
    ys.reserve(width);
    cbs.reserve(width);
    crs.reserve(width);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let line = if simd.level().as_avx512().is_some() {
        let mut chunks = line.chunks_exact(16 * N);
        for data in &mut chunks {
            let r = load16::<_, N, R>(simd, data);
            let g = load16::<_, N, G>(simd, data);
            let b = load16::<_, N, B>(simd, data);
            let y = r * 19595 + g * 38470 + b * 7471 + 0x7FFF;
            let cb = r * -11059 - g * 21709 + b * 32768 + ((128 << 16) + 0x7FFF);
            let cr = r * 32768 - g * 27439 - b * 5329 + ((128 << 16) + 0x7FFF);
            ys.extend_from_slice(&pack16(y));
            cbs.extend_from_slice(&pack16(cb));
            crs.extend_from_slice(&pack16(cr));
        }
        chunks.remainder()
    } else {
        line
    };
    let mut chunks = line.chunks_exact(8 * N);
    for data in &mut chunks {
        let r = load::<_, N, R>(simd, data);
        let g = load::<_, N, G>(simd, data);
        let b = load::<_, N, B>(simd, data);
        let y = r * 19595 + g * 38470 + b * 7471 + 0x7FFF;
        let cb = r * -11059 - g * 21709 + b * 32768 + ((128 << 16) + 0x7FFF);
        let cr = r * 32768 - g * 27439 - b * 5329 + ((128 << 16) + 0x7FFF);
        ys.extend_from_slice(&pack(y));
        cbs.extend_from_slice(&pack(cb));
        crs.extend_from_slice(&pack(cr));
    }
    for data in chunks.remainder().chunks_exact(N) {
        let (y, cb, cr) = rgb_to_ycbcr(data[R], data[G], data[B]);
        ys.push(y);
        cbs.push(cb);
        crs.push(cr);
    }
}

macro_rules! ycbcr_image {
    ($name:ident, $n:expr, $r:expr, $g:expr, $b:expr) => {
        pub struct $name<'a>(pub &'a [u8], pub u16, pub u16);

        impl ImageBuffer for $name<'_> {
            fn get_jpeg_color_type(&self) -> JpegColorType {
                JpegColorType::Ycbcr
            }

            fn width(&self) -> u16 {
                self.1
            }

            fn height(&self) -> u16 {
                self.2
            }

            fn fill_buffers(&self, y: u16, buffers: &mut [Vec<u8>; 4]) {
                let width = usize::from(self.1) * $n;
                let start = usize::from(y) * width;
                let line = &self.0[start..start + width];
                dispatch!(level(), simd => convert::<_, $n, $r, $g, $b>(simd, line, buffers));
            }
        }
    };
}

ycbcr_image!(RgbImageSimd, 3, 0, 1, 2);
ycbcr_image!(RgbaImageSimd, 4, 0, 1, 2);
ycbcr_image!(BgrImageSimd, 3, 2, 1, 0);
ycbcr_image!(BgraImageSimd, 4, 2, 1, 0);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn levels() -> Vec<Level> {
        let detected = level();
        let mut levels = vec![Level::baseline()];
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
        levels.push(detected);
        levels.dedup_by_key(|level| core::mem::discriminant(level));
        levels
    }

    fn check_format<const N: usize, const R: usize, const G: usize, const B: usize>(
        make_image: impl Fn(&[u8], u16, u16, u16, &mut [Vec<u8>; 4]),
    ) {
        let mut state = 42u64;
        for width in (0..=33).chain([127, 128, 129, 515]) {
            // Deliberately unaligned input, distinct rows and arbitrary alpha values.
            let data: Vec<u8> = (0..1 + width * N * 3)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (state >> 32) as u8
                })
                .collect();
            let data = &data[1..];
            let initial = [vec![7, 8], vec![9], vec![], vec![42, 43]];
            let mut expected = initial.clone();
            for pixel in data.chunks_exact(N) {
                let (y, cb, cr) = rgb_to_ycbcr(pixel[R], pixel[G], pixel[B]);
                expected[0].push(y);
                expected[1].push(cb);
                expected[2].push(cr);
            }
            for level in levels() {
                let mut actual = initial.clone();
                for row in 0..3 {
                    let line = &data[row * width * N..(row + 1) * width * N];
                    dispatch!(level, simd => convert::<_, N, R, G, B>(simd, line, &mut actual));
                }
                assert_eq!(actual, expected, "width {width}, level {level:?}");
            }
            let mut actual = initial;
            for row in 0..3 {
                make_image(data, width as u16, 3, row, &mut actual);
            }
            assert_eq!(actual, expected, "dispatched image, width {width}");
        }
    }

    #[test]
    fn formats_rows_tails_and_append() {
        check_format::<3, 0, 1, 2>(|d, w, h, y, out| RgbImageSimd(d, w, h).fill_buffers(y, out));
        check_format::<4, 0, 1, 2>(|d, w, h, y, out| RgbaImageSimd(d, w, h).fill_buffers(y, out));
        check_format::<3, 2, 1, 0>(|d, w, h, y, out| BgrImageSimd(d, w, h).fill_buffers(y, out));
        check_format::<4, 2, 1, 0>(|d, w, h, y, out| BgraImageSimd(d, w, h).fill_buffers(y, out));
    }

    #[test]
    fn every_rgb_matches_scalar() {
        // All 2^24 colors cover fixed-point rounding boundaries and output extrema.
        let mut data = vec![0; 256 * 256 * 3];
        let mut expected: [Vec<u8>; 4] = core::array::from_fn(|_| Vec::new());
        let mut actual = expected.clone();
        for level in levels() {
            for r in 0..=255u8 {
                for (i, pixel) in data.chunks_exact_mut(3).enumerate() {
                    pixel.copy_from_slice(&[r, (i >> 8) as u8, i as u8]);
                }
                for buf in expected.iter_mut().chain(actual.iter_mut()) {
                    buf.clear();
                }
                for pixel in data.chunks_exact(3) {
                    let (y, cb, cr) = rgb_to_ycbcr(pixel[0], pixel[1], pixel[2]);
                    expected[0].push(y);
                    expected[1].push(cb);
                    expected[2].push(cr);
                }
                dispatch!(level, simd => convert::<_, 3, 0, 1, 2>(simd, &data, &mut actual));
                assert_eq!(actual, expected, "r {r}, level {level:?}");
            }
        }
    }
}
