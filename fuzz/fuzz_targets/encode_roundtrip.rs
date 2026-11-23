//! Round-trip target: whatever settings the fuzzer picks, the encoder must
//! either reject the input with an error or produce a JPEG that jpeg-decoder
//! decodes to the same dimensions and pixel format.

#![no_main]

use jpeg_decoder::{Decoder, PixelFormat};
use jpeg_encoder::*;
use libfuzzer_sys::fuzz_target;

const COLOR_TYPES: [ColorType; 9] = [
    ColorType::Luma,
    ColorType::Rgb,
    ColorType::Rgba,
    ColorType::Bgr,
    ColorType::Bgra,
    ColorType::Ycbcr,
    ColorType::Cmyk,
    ColorType::CmykAsYcck,
    ColorType::Ycck,
];

const SAMPLING_FACTORS: [(u8, u8); 8] = [
    (1, 1),
    (1, 2),
    (1, 4),
    (2, 1),
    (2, 2),
    (2, 4),
    (4, 1),
    (4, 2),
];

fuzz_target!(|data: &[u8]| {
    // Header: color type, quality, flags, sampling factor, progressive scans,
    // restart interval (2 bytes), width - 1. Pixels follow.
    let Some((header, pixels)) = data.split_first_chunk::<8>() else {
        return;
    };
    let [color, quality, flags, sampling, scans, restart_lo, restart_hi, width] = *header;

    let color_type = COLOR_TYPES[usize::from(color) % COLOR_TYPES.len()];
    let (h, v) = SAMPLING_FACTORS[usize::from(sampling) % SAMPLING_FACTORS.len()];
    let bytes_per_pixel = match color_type {
        ColorType::Luma => 1,
        ColorType::Rgb | ColorType::Bgr | ColorType::Ycbcr => 3,
        _ => 4,
    };

    let width = u16::from(width) + 1;
    let Ok(height) = u16::try_from(pixels.len() / bytes_per_pixel / usize::from(width)) else {
        return;
    };
    if height == 0 {
        return;
    }

    let mut out = Vec::new();
    let mut encoder = Encoder::new(&mut out, quality);
    encoder.set_sampling_factor(SamplingFactor::from_factors(h, v).unwrap());
    encoder.set_optimized_huffman_tables(flags & 1 != 0);
    encoder.set_exact_quantization(flags & 2 != 0);
    if flags & 4 != 0 {
        encoder.set_chroma_subsampling_method(ChromaSubsamplingMethod::Average);
    }
    if flags & 8 != 0 {
        encoder.set_progressive(true);
        // Valid range is 2..=64.
        encoder.set_progressive_scans(2 + scans % 63);
    }
    encoder.set_restart_interval(u16::from_le_bytes([restart_lo, restart_hi]));

    if encoder.encode(pixels, width, height, color_type).is_err() {
        return;
    }

    let mut decoder = Decoder::new(out.as_slice());
    let decoded = decoder.decode().expect("encoder produced an undecodable JPEG");
    let info = decoder.info().unwrap();

    let expected_format = match color_type {
        ColorType::Luma => PixelFormat::L8,
        ColorType::Cmyk | ColorType::CmykAsYcck | ColorType::Ycck => PixelFormat::CMYK32,
        _ => PixelFormat::RGB24,
    };
    assert_eq!((info.width, info.height), (width, height));
    assert_eq!(info.pixel_format, expected_format);
    assert_eq!(
        decoded.len(),
        usize::from(width) * usize::from(height) * expected_format.pixel_bytes()
    );
});
