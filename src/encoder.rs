use crate::writer::{JfifWriter, ZIGZAG};
use crate::fdct::fdct;
use crate::marker::Marker;
use crate::huffman::{HuffmanTable, CodingClass};
use crate::image_buffer::*;
use crate::quantization::QuantizationTable;
use crate::Density;

use std::io::{Write, Result as IOResult, BufWriter};
use std::fs::File;
use std::path::Path;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ColorType {
    Rgb,
    Rgba,
    Bgr,
    Bgra,
    Ycbcr,
}

pub(crate) struct Component {
    pub id: u8,
    pub quantization_table: u8,
    pub dc_huffman_table: u8,
    pub ac_huffman_table: u8,
    pub horizontal_sampling_factor: u8,
    pub vertical_sampling_factor: u8,
}

pub struct JpegEncoder<W: Write> {
    writer: JfifWriter<W>,
    density: Density,

    components: Vec<Component>,
    quantization_tables: [QuantizationTable; 2],
    huffman_tables: [(HuffmanTable, HuffmanTable); 2],

    horizontal_sampling_factor: u8,
    vertical_sampling_factor: u8,
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

        let luma_sampling = if quality < 90 {
            2
        } else {
            1
        };

        JpegEncoder {
            writer: JfifWriter::new(w),
            density: Density::None,
            components: vec![],
            quantization_tables,
            huffman_tables,
            horizontal_sampling_factor: luma_sampling,
            vertical_sampling_factor: luma_sampling,
        }
    }

    pub fn set_density(&mut self, density: Density) {
        self.density = density;
    }

    pub fn set_sampling_factor(&mut self, horizontal_factor: u8, vertical_factor: u8) {
        self.horizontal_sampling_factor = horizontal_factor;
        self.vertical_sampling_factor = vertical_factor;
    }

    pub fn encode(
        &mut self,
        data: &[u8],
        width: u16,
        height: u16,
        color_type: ColorType,
    ) -> IOResult<()> {
        self.components.push(Component {
            id: 0,
            quantization_table: 0,
            dc_huffman_table: 0,
            ac_huffman_table: 0,
            horizontal_sampling_factor: self.horizontal_sampling_factor,
            vertical_sampling_factor: self.vertical_sampling_factor,
        });
        self.components.push(Component {
            id: 1,
            quantization_table: 1,
            dc_huffman_table: 1,
            ac_huffman_table: 1,
            horizontal_sampling_factor: 1,
            vertical_sampling_factor: 1,
        });
        self.components.push(Component {
            id: 2,
            quantization_table: 1,
            dc_huffman_table: 1,
            ac_huffman_table: 1,
            horizontal_sampling_factor: 1,
            vertical_sampling_factor: 1,
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

        match color_type {
            ColorType::Rgb => self.encode_image(RgbImage(data, width as u32, height as u32))?,
            ColorType::Rgba => self.encode_image(RgbaImage(data, width as u32, height as u32))?,
            ColorType::Bgr => self.encode_image(BgrImage(data, width as u32, height as u32))?,
            ColorType::Bgra => self.encode_image(BgraImage(data, width as u32, height as u32))?,
            ColorType::Ycbcr => self.encode_image(YCbCrImage(data, width as u32, height as u32))?,
        }

        self.writer.write_bits(0x7F, 7)?;
        self.writer.flush_bit_buffer()?;
        self.writer.write_marker(Marker::EOI)?;

        Ok(())
    }

    fn get_max_sampling_size(&self) -> (u32, u32) {
        let max_h_sampling = self.components
            .iter()
            .fold(1, |value, component| value.max(component.horizontal_sampling_factor)) as u32;

        let max_v_sampling = self.components
            .iter()
            .fold(1, |value, component| value.max(component.vertical_sampling_factor)) as u32;

        (max_h_sampling, max_v_sampling)
    }

    fn encode_image<I: ImageBuffer>(
        &mut self,
        image: I,
    ) -> IOResult<()> {
        let (max_h_sampling, max_v_sampling) = self.get_max_sampling_size();

        let num_cols = ceil_div(image.width(), 8 * max_h_sampling);
        let num_rows = ceil_div(image.height(), 8 * max_v_sampling);

        let buffer_width = (num_cols * 8 * max_h_sampling) as usize;
        let buffer_size = buffer_width * 8 * max_v_sampling as usize;

        let mut row: [Vec<_>; 3] = [
            Vec::with_capacity(buffer_size),
            Vec::with_capacity(buffer_size),
            Vec::with_capacity(buffer_size),
        ];

        let mut prev_dc = [0i16; 3];

        for block_y in 0..num_rows {
            for r in &mut row {
                r.clear();
            }

            for y in 0..(8 * max_v_sampling) {
                for x in 0..buffer_width as u32 {
                    image.fill_buffers(x, y + block_y * 8 * max_v_sampling, &mut row);
                }
            }

            for block_x in 0..num_cols {
                for (i, component) in self.components.iter().enumerate() {
                    for v_offset in 0..component.vertical_sampling_factor as usize {
                        for h_offset in 0..component.horizontal_sampling_factor as usize {
                            let mut block = get_block(
                                &row[i],
                                (block_x as usize) * 8 * max_h_sampling as usize + (h_offset * 8),
                                v_offset * 8,
                                max_h_sampling as usize / component.horizontal_sampling_factor as usize,
                                max_v_sampling as usize / component.vertical_sampling_factor as usize,
                                buffer_width);


                            fdct(&mut block);

                            let q_block = self.quantize_block(&component, &block);

                            self.writer.write_block(
                                &q_block,
                                prev_dc[i],
                                &self.huffman_tables[component.dc_huffman_table as usize].0,
                                &self.huffman_tables[component.ac_huffman_table as usize].1,
                            )?;

                            prev_dc[i] = q_block[0];
                        }
                    }
                }
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

impl JpegEncoder<BufWriter<File>> {
    pub fn new_file<P: AsRef<Path>>(path: P, quality: u8) -> IOResult<JpegEncoder<BufWriter<File>>> {
        let file = File::create(path)?;
        let buf = BufWriter::new(file);
        Ok(Self::new(buf, quality))
    }
}

fn get_block(data: &[u8],
             start_x: usize,
             start_y: usize,
             col_stride: usize,
             row_stride: usize,
             width: usize) -> [i16; 64] {
    let mut block = [0i16; 64];

    for x in 0..8 {
        for y in 0..8 {
            block[x + y * 8] = data[(x * col_stride) + start_x + (y + start_y) * row_stride * width] as i16;
        }
    }

    block
}

fn ceil_div(value: u32, div: u32) -> u32 {
    value / div + u32::from(value % div != 0)
}
