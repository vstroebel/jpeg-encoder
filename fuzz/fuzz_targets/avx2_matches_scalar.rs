//! Differential target: the AVX2 FDCT and RGB to YCbCr conversion must produce
//! exactly the same output as their scalar counterparts.
//! Inactive on non-x86 architectures and CPUs without AVX2 support.

#![no_main]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use jpeg_encoder::{fdct, fdct_avx2, AlignedBlock, ImageBuffer, RgbImage, RgbImageAVX2};
use libfuzzer_sys::fuzz_target;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fuzz_target!(|data: &[u8]| {
    // The AVX2 entry points do not check for AVX2 themselves.
    if !std::is_x86_feature_detected!("avx2") || data.len() < 64 {
        return;
    }
    let (block, pixels) = data.split_at(64);

    // Samples are level-shifted bytes, the same range the encoder feeds the FDCT.
    let mut samples = [0i16; 64];
    for (s, &b) in samples.iter_mut().zip(block) {
        *s = i16::from(b) - 128;
    }
    let mut scalar = AlignedBlock::new(samples);
    let mut avx2 = AlignedBlock::new(samples);
    fdct(&mut scalar);
    fdct_avx2(&mut avx2);
    assert_eq!(
        scalar.data, avx2.data,
        "fdct mismatch for input {samples:?}"
    );

    // A single row of arbitrary width exercises both the vector body and the remainder.
    let Ok(width) = u16::try_from(pixels.len() / 3) else {
        return;
    };
    if width == 0 {
        return;
    }
    let rgb = &pixels[..usize::from(width) * 3];
    let mut scalar_buffers: [Vec<u8>; 4] = Default::default();
    let mut avx2_buffers: [Vec<u8>; 4] = Default::default();
    RgbImage(rgb, width, 1).fill_buffers(0, &mut scalar_buffers);
    RgbImageAVX2(rgb, width, 1).fill_buffers(0, &mut avx2_buffers);
    assert_eq!(
        scalar_buffers, avx2_buffers,
        "ycbcr mismatch for input {rgb:?}"
    );
});

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fuzz_target!(|_data: &[u8]| {});
