#![no_main]

use libfuzzer_sys::fuzz_target;

use jpeg_encoder::*;

fuzz_target!(|data: &[u8]| {

    let pixels = data.len() / 4;

    let width = (pixels as f64).sqrt() as u16;
    let height = width;

    if width >0 && width < u16::MAX &&  height >0 && height < u16::MAX  {
        let mut out = Vec::new();
        let mut encoder = Encoder::new(&mut out, 100);
        encoder.set_progressive(true);
        encoder.encode(data, width as u16, height as u16, ColorType::Ycck).unwrap();
    }
});