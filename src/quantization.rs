use std::num::NonZeroU8;

static DEFAULT_LUMA_TABLE: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
];

static DEFAULT_CHROMA_TABLE: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
];

///
pub struct QuantizationTable {
    table: [NonZeroU8; 64],
}

impl QuantizationTable {
    pub fn new_with_quality(table: [u8; 64], quality: u8) -> QuantizationTable {
        let quality = quality.max(1).min(100) as u32;

        let scale = if quality < 50 {
            5000 / quality
        } else {
            200 - quality * 2
        };

        let mut q_table = [NonZeroU8::new(1).unwrap(); 64];

        for (i, &v) in table.iter().enumerate() {
            let v = v as u32;

            let v = (v * scale + 50) / 100;

            q_table[i] = NonZeroU8::new(v.max(1).min(255) as u8).unwrap();
        }

        QuantizationTable {
            table: q_table
        }
    }

    pub fn default_luma(quality: u8) -> QuantizationTable {
        Self::new_with_quality(DEFAULT_LUMA_TABLE, quality)
    }

    pub fn default_chroma(quality: u8) -> QuantizationTable {
        Self::new_with_quality(DEFAULT_CHROMA_TABLE, quality)
    }

    #[inline]
    pub fn get(&self, index: usize) -> u8 {
        self.table[index].get()
    }

    #[inline]
    pub fn quantize(&self, value: i16, index: usize) -> i16 {
        let q_value = self.table[index].get() as i16;
        let value = value / q_value;
        (value + 4) / 8
    }
}