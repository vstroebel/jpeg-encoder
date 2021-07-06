#![no_main]

use libfuzzer_sys::fuzz_target;
use jpeg_encoder::*;

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 + 3 {

        let q_table = &data[0..64];
        let data = &data[64..];

        let pixels = data.len() / 3;

        let width = (pixels as f64).sqrt() as u16;
        let height = width;

        if width >0 && width < u16::MAX &&  height >0 && height < u16::MAX  {
            let mut table = [0;64];
            for i in 0..64 {
                table[i] = q_table[i].max(1);
            }

            let table = QuantizationTableType::Custom(Box::new(table));

            let mut out = Vec::new();
            let mut encoder = Encoder::new(&mut out, 100);

            encoder.set_quantization_tables(table.clone(), table);

            encoder.encode(data, width as u16, height as u16, ColorType::Rgb).unwrap();
        }
    }
});
