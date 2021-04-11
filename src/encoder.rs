use crate::writer::{JfifWriter, ZIGZAG};
use crate::fdct::fdct;
use crate::marker::Marker;
use crate::huffman::{HuffmanTable, CodingClass};
use crate::image_buffer::*;
use crate::quantization::QuantizationTable;
use crate::Density;

use std::io::{Write, Result as IOResult};

pub(crate) struct Component {
    pub id: u8,
    pub quantization_table: u8,
    pub dc_huffman_table: u8,
    pub ac_huffman_table: u8,
}

pub struct JpegEncoder<W: Write> {
    writer: JfifWriter<W>,
    density: Density,

    components: Vec<Component>,
    quantization_tables: [QuantizationTable; 2],
    huffman_tables: [(HuffmanTable, HuffmanTable); 2],

}

impl<W: Write> JpegEncoder<W> {
    pub fn new(w: W, quality: u8) -> JpegEncoder<W> {
        let huffman_tables = [
            (HuffmanTable::default_luma_dc(), HuffmanTable::default_luma_ac()),
            (HuffmanTable::default_chroma_dc(), HuffmanTable::default_chroma_ac())
        ];

        let quantization_tables = [
            QuantizationTable::default_luma(quality),
            QuantizationTable::default_chroma(quality)
        ];

        JpegEncoder {
            writer: JfifWriter::new(w),
            density: Density::None,
            components: vec![],
            quantization_tables,
            huffman_tables,

        }
    }

    pub fn set_density(&mut self, density: Density) {
        self.density = density;
    }

    pub fn encode(
        &mut self,
        data: &[u8],
        width: u16,
        height: u16,
    ) -> IOResult<()> {
        self.components.push(Component {
            id: 0,
            quantization_table: 0,
            dc_huffman_table: 0,
            ac_huffman_table: 0,
        });
        self.components.push(Component {
            id: 1,
            quantization_table: 1,
            dc_huffman_table: 1,
            ac_huffman_table: 1,
        });
        self.components.push(Component {
            id: 2,
            quantization_table: 1,
            dc_huffman_table: 1,
            ac_huffman_table: 1,
        });

        self.writer.write_marker(Marker::SOI)?;

        self.writer.write_header(&self.density)?;

        self.writer.write_frame_header(width, height, &self.components)?;

        self.writer.write_quantization_segment(0, &self.quantization_tables[0])?;
        self.writer.write_quantization_segment(1, &self.quantization_tables[1])?;

        self.writer.write_huffman_segment(
            CodingClass::Dc,
            0,
            &self.huffman_tables[0].0,
        )?;

        self.writer.write_huffman_segment(
            CodingClass::Ac,
            0,
            &self.huffman_tables[0].1,
        )?;

        self.writer.write_huffman_segment(
            CodingClass::Dc,
            1,
            &self.huffman_tables[1].0,
        )?;

        self.writer.write_huffman_segment(
            CodingClass::Ac,
            1,
            &self.huffman_tables[1].1,
        )?;

        self.writer.write_scan_header(&self.components)?;

        self.encode_image(ImageBuffer {
            data,
            width: width as u32,
            height: height as u32,
        })?;

        self.writer.write_bits(0x7F, 7);
        self.writer.flush_bit_buffer()?;
        self.writer.write_marker(Marker::EOI)?;

        Ok(())
    }

    fn encode_image(
        &mut self,
        image: ImageBuffer,
    ) -> IOResult<()> {
        let num_cols = ceil_div(image.width(), 8);
        let num_rows = ceil_div(image.height(), 8);

        let mut prev_dc = [0i16; 3];

        for block_y in 0..num_rows {
            for block_x in 0..num_cols {
                let (mut block_y, mut block_cb, mut block_cr) = get_block(
                    &image,
                    block_x * 8,
                    block_y * 8);

                fdct(&mut block_y);
                fdct(&mut block_cb);
                fdct(&mut block_cr);

                let q_block_y = self.quantize_block(&self.components[0], &block_y);
                let q_block_cb = self.quantize_block(&self.components[1], &block_cb);
                let q_block_cr = self.quantize_block(&self.components[2], &block_cr);

                self.writer.write_block(&q_block_y, prev_dc[0], &self.huffman_tables[0].0, &self.huffman_tables[0].1)?;
                self.writer.write_block(&q_block_cb, prev_dc[1], &self.huffman_tables[1].0, &self.huffman_tables[1].1)?;
                self.writer.write_block(&q_block_cr, prev_dc[2], &self.huffman_tables[1].0, &self.huffman_tables[1].1)?;

                prev_dc[0] = q_block_y[0];
                prev_dc[1] = q_block_cb[0];
                prev_dc[2] = q_block_cr[0];
            }
        }

        Ok(())
    }

    fn quantize_block(&self, component: &Component, block: &[i16; 64]) -> [i16; 64] {
        let mut q_block = [0i16; 64];

        for i in 0..64 {
            q_block[i] = self.quantization_tables[component.quantization_table as usize].quantize(block[ZIGZAG[i] as usize], i);
        }

        q_block
    }
}

fn get_block(image: &ImageBuffer,
             start_x: u32,
             start_y: u32) -> ([i16; 64], [i16; 64], [i16; 64]) {
    let mut block_y = [0i16; 64];
    let mut block_cb = [0i16; 64];
    let mut block_cr = [0i16; 64];

    for x in 0..8 {
        for y in 0..8 {
            let (cy, cb, cr) = image.get_pixel(start_x + x, start_y + y);

            let offset = (y * 8 + x) as usize;
            block_y[offset] = cy as i16;
            block_cb[offset] = cb as i16;
            block_cr[offset] = cr as i16;
        }
    }

    (block_y, block_cb, block_cr)
}

fn ceil_div(value: u32, div: u32) -> u32 {
    value / div + u32::from(value % div != 0)
}
