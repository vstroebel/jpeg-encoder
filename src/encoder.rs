use crate::writer::{JfifWriter, ZIGZAG};
use crate::fdct::fdct;
use crate::marker::Marker;
use crate::huffman::{HuffmanTable, CodingClass};
use crate::image_buffer::*;
use crate::quantization::QuantizationTable;
use crate::Density;

use std::io::{Write, Result as IOResult, Error as IOError, ErrorKind, BufWriter};
use std::fs::File;
use std::path::Path;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum JpegColorType {
    Gray,
    Ycbcr,
    Cmyk,
    Ycck,
}

impl JpegColorType {
    pub(crate) fn get_num_components(&self) -> usize {
        use JpegColorType::*;

        match self {
            Gray => 1,
            Ycbcr => 3,
            Cmyk | Ycck => 4,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ColorType {
    Gray,
    Rgb,
    Rgba,
    Bgr,
    Bgra,
    Ycbcr,
    Cmyk,
    CmykAsYcck,
    Ycck,
}

impl ColorType {
    pub(crate) fn get_bytes_per_pixel(&self) -> usize {
        use ColorType::*;

        match self {
            Gray => 1,
            Rgb | Bgr | Ycbcr => 3,
            Rgba | Bgra | Cmyk | CmykAsYcck | Ycck => 4,
        }
    }

    pub(crate) fn get_jpeg_color_type(&self) -> JpegColorType {
        use ColorType::*;

        match self {
            Gray => JpegColorType::Gray,
            Rgb | Rgba | Bgr | Bgra | Ycbcr => JpegColorType::Ycbcr,
            Cmyk => JpegColorType::Cmyk,
            CmykAsYcck | Ycck => JpegColorType::Ycck,
        }
    }
}

pub(crate) struct Component {
    pub id: u8,
    pub quantization_table: u8,
    pub dc_huffman_table: u8,
    pub ac_huffman_table: u8,
    pub horizontal_sampling_factor: u8,
    pub vertical_sampling_factor: u8,
}

macro_rules! add_component {
    ($components:expr, $id:expr, $dest:expr, $h_sample:expr, $v_sample:expr) => {
        $components.push(Component {
            id: $id,
            quantization_table: $dest,
            dc_huffman_table: $dest,
            ac_huffman_table: $dest,
            horizontal_sampling_factor: $h_sample,
            vertical_sampling_factor: $v_sample,
        });
    }
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
        let required_data_len = width as usize * height as usize * color_type.get_bytes_per_pixel();

        if data.len() < required_data_len {
            return Err(IOError::new(ErrorKind::Other,
                                    format!("Image data too small for dimensions and color_type: {} need at least {}", data.len(), required_data_len)));
        }

        let jpeg_color_type = color_type.get_jpeg_color_type();
        self.init_components(jpeg_color_type);

        let num_components = jpeg_color_type.get_num_components();

        self.writer.write_marker(Marker::SOI)?;

        self.writer.write_header(&self.density)?;

        if color_type == ColorType::CmykAsYcck || color_type == ColorType::Ycck {
            //Set ColorTransform info to YCCK
            let app_14 = b"Adobe\0\0\0\0\0\0\x02";
            self.writer.write_segment(Marker::APP(14), app_14.as_ref())?;
        }

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

        if num_components >= 3 {
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
        }


        match color_type {
            ColorType::Gray => self.encode_image(GrayImage(data, width as u32, height as u32))?,
            ColorType::Rgb => self.encode_image(RgbImage(data, width as u32, height as u32))?,
            ColorType::Rgba => self.encode_image(RgbaImage(data, width as u32, height as u32))?,
            ColorType::Bgr => self.encode_image(BgrImage(data, width as u32, height as u32))?,
            ColorType::Bgra => self.encode_image(BgraImage(data, width as u32, height as u32))?,
            ColorType::Ycbcr => self.encode_image(YCbCrImage(data, width as u32, height as u32))?,
            ColorType::Cmyk => self.encode_image(CmykImage(data, width as u32, height as u32))?,
            ColorType::CmykAsYcck => self.encode_image(CmykAsYcckImage(data, width as u32, height as u32))?,
            ColorType::Ycck => self.encode_image(YcckImage(data, width as u32, height as u32))?,
        }

        self.writer.write_marker(Marker::EOI)?;

        Ok(())
    }

    fn init_components(&mut self, color: JpegColorType) {
        match color {
            JpegColorType::Gray => {
                add_component!(self.components, 0, 0, 1, 1);
            }
            JpegColorType::Ycbcr => {
                add_component!(self.components, 0, 0, self.horizontal_sampling_factor, self.vertical_sampling_factor);
                add_component!(self.components, 1, 1, 1, 1);
                add_component!(self.components, 2, 1, 1, 1);
            }
            JpegColorType::Cmyk => {
                add_component!(self.components, 0, 1, 1, 1);
                add_component!(self.components, 1, 1, 1, 1);
                add_component!(self.components, 2, 1, 1, 1);
                add_component!(self.components, 3, 0, self.horizontal_sampling_factor, self.vertical_sampling_factor);
            }
            JpegColorType::Ycck => {
                add_component!(self.components, 0, 0, self.horizontal_sampling_factor, self.vertical_sampling_factor);
                add_component!(self.components, 1, 1, 1, 1);
                add_component!(self.components, 2, 1, 1, 1);
                add_component!(self.components, 3, 0, self.horizontal_sampling_factor, self.vertical_sampling_factor);
            }
        }
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
        // Interleaved mode is only supported with h/v sampling factors of 1 or 2
        if self.horizontal_sampling_factor > 2 || self.vertical_sampling_factor > 2 {
            self.encode_image_sequential(image)
        } else {
            self.encode_image_interleaved(image)
        }
    }

    fn encode_image_interleaved<I: ImageBuffer>(
        &mut self,
        image: I,
    ) -> IOResult<()> {
        self.writer.write_scan_header(&self.components.iter().collect::<Vec<_>>())?;

        let (max_h_sampling, max_v_sampling) = self.get_max_sampling_size();

        let num_cols = ceil_div(image.width(), 8 * max_h_sampling);
        let num_rows = ceil_div(image.height(), 8 * max_v_sampling);

        let buffer_width = (num_cols * 8 * max_h_sampling) as usize;
        let buffer_size = buffer_width * 8 * max_v_sampling as usize;

        let mut row: [Vec<_>; 4] = self.init_rows(buffer_size);

        let mut prev_dc = [0i16; 4];

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

        self.writer.finalize_bit_buffer()
    }

    fn init_rows(&mut self, buffer_size: usize) -> [Vec<u8>; 4] {

        // To simplify the code and to give the compiler more infos to optimize stuff we always initialize 4 components
        // Resource overhead should be minimal because an empty Vec doesn't allocate

        match self.components.len() {
            1 => [
                Vec::with_capacity(buffer_size),
                Vec::new(),
                Vec::new(),
                Vec::new()
            ],
            3 => [
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::new()
            ],
            4 => [
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size)
            ],
            len => unreachable!("Unsupported component length: {}", len),
        }
    }

    fn quantize_block(&self, component: &Component, block: &[i16; 64]) -> [i16; 64] {
        let mut q_block = [0i16; 64];

        for i in 0..64 {
            q_block[i] = self.quantization_tables[component.quantization_table as usize].quantize(block[ZIGZAG[i] as usize], i);
        }

        q_block
    }

    fn encode_image_sequential<I: ImageBuffer>(
        &mut self,
        image: I,
    ) -> IOResult<()> {
        let num_cols = ceil_div(image.width(), 8);
        let num_rows = ceil_div(image.height(), 8);

        let buffer_width = (num_cols * 8) as usize;
        let buffer_size = (num_cols * num_rows * 64) as usize;

        let mut row: [Vec<_>; 4] = self.init_rows(buffer_size);

        for y in 0..num_rows * 8 {
            for x in 0..num_cols * 8 {
                image.fill_buffers(x, y, &mut row);
            }
        }


        let mut blocks: [Vec<_>; 4] = self.init_block_buffers(buffer_size / 64);

        let (max_h_sampling, max_v_sampling) = self.get_max_sampling_size();

        for (i, component) in self.components.iter().enumerate() {
            let h_scale = max_h_sampling as usize / component.horizontal_sampling_factor as usize;
            let v_scale = max_v_sampling as usize / component.vertical_sampling_factor as usize;

            let cols = num_cols as usize / h_scale;
            let rows = num_rows as usize / v_scale;


            for block_y in 0..rows {
                for block_x in 0..cols {
                    let mut block = get_block(
                        &row[i],
                        block_x * 8 * h_scale,
                        block_y * 8 * v_scale,
                        h_scale,
                        v_scale,
                        buffer_width);

                    fdct(&mut block);

                    let q_block = self.quantize_block(&component, &block);

                    blocks[i].push(q_block);
                }
            }
        }

        for (i, component) in self.components.iter().enumerate() {
            self.writer.write_scan_header(&[component])?;

            let mut prev_dc = 0;

            for block in &blocks[i] {
                self.writer.write_block(
                    &block,
                    prev_dc,
                    &self.huffman_tables[component.dc_huffman_table as usize].0,
                    &self.huffman_tables[component.ac_huffman_table as usize].1,
                )?;

                prev_dc = block[0];
            }

            self.writer.finalize_bit_buffer()?;
        }

        Ok(())
    }

    fn init_block_buffers(&mut self, buffer_size: usize) -> [Vec<[i16; 64]>; 4] {

        // To simplify the code and to give the compiler more infos to optimize stuff we always initialize 4 components
        // Resource overhead should be minimal because an empty Vec doesn't allocate

        match self.components.len() {
            1 => [
                Vec::with_capacity(buffer_size),
                Vec::new(),
                Vec::new(),
                Vec::new()
            ],
            3 => [
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::new()
            ],
            4 => [
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size),
                Vec::with_capacity(buffer_size)
            ],
            len => unreachable!("Unsupported component length: {}", len),
        }
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
            let ix = start_x + (x * col_stride);
            let iy = start_y + (y * row_stride);

            block[y * 8 + x] = data[iy * width + ix] as i16;
        }
    }

    block
}

fn ceil_div(value: u32, div: u32) -> u32 {
    value / div + u32::from(value % div != 0)
}
