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
    Luma,
    Ycbcr,
    Cmyk,
    Ycck,
}

impl JpegColorType {
    pub(crate) fn get_num_components(&self) -> usize {
        use JpegColorType::*;

        match self {
            Luma => 1,
            Ycbcr => 3,
            Cmyk | Ycck => 4,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ColorType {
    Luma,
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
            Luma => 1,
            Rgb | Bgr | Ycbcr => 3,
            Rgba | Bgra | Cmyk | CmykAsYcck | Ycck => 4,
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

    progressive_scans: u8,

    optimize_huffman_table: bool,
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
            progressive_scans: 0,
            optimize_huffman_table: false,
        }
    }

    /// Set pixel density for the image
    ///
    /// By default, this value is None which is equal to "1 pixel per pixel".
    /// # Example
    /// ```no_run
    /// # fn main() -> std::io::Result<()> {
    /// use jpeg_encoder::{JpegEncoder, Density};
    ///
    /// let mut encoder = JpegEncoder::new_file("some.jpeg", 100)?;
    ///
    /// // Set horizontal and vertical density to 72 dpi (dots per inch)
    /// encoder.set_density(Density::Inch{x: 72, y: 72});
    ///
    /// assert_eq!(encoder.density(), Density::Inch{x: 72, y: 72});
    /// # Ok(())
    /// # }
    pub fn set_density(&mut self, density: Density) {
        self.density = density;
    }

    /// Return pixel density
    pub fn density(&self) -> Density {
        self.density
    }

    pub fn set_sampling_factor(&mut self, horizontal_factor: u8, vertical_factor: u8) {
        self.horizontal_sampling_factor = horizontal_factor;
        self.vertical_sampling_factor = vertical_factor;
    }

    /// Controls if progressive encoding is used.
    ///
    /// By default, progressive encoding uses 4 scans.<br>
    /// Use [set_progressive_scans](JpegEncoder::set_progressive_scans) to use a different number of scans
    ///
    /// # Example
    /// ```no_run
    /// # pub fn main() -> std::io::Result<()> {
    /// use jpeg_encoder::JpegEncoder;
    ///
    /// let mut encoder = JpegEncoder::new_file("some.jpeg", 100)?;
    ///
    /// encoder.set_progressive(true);
    ///
    /// assert_eq!(encoder.progressive_scans(), Some(4));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_progressive(&mut self, progressive: bool) {
        self.set_progressive_scans(if progressive {
            4
        } else {
            0
        });
    }

    /// Set number of scans per component for progressive encoding
    ///
    /// Number of scans must be between 2 and 64.
    /// There is at least one scan for the DC coefficients and one for the remaining 63 AC coefficients.
    ///
    /// # Panics
    /// If number of scans is not within valid range
    pub fn set_progressive_scans(&mut self, scans: u8) {
        assert!(scans >= 2 && scans <= 64, "Invalid number of scans: {}", scans);
        self.progressive_scans = scans;
    }

    /// Return number of progressive scans if progressive encoding is enabled
    pub fn progressive_scans(&self) -> Option<u8> {
        match self.progressive_scans {
            0 => None,
            scans => Some(scans)
        }
    }

    pub fn set_optimized_huffman_tables(&mut self, optimize_huffman_table: bool) {
        self.optimize_huffman_table = optimize_huffman_table;
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

        match color_type {
            ColorType::Luma => self.encode_image(GrayImage(data, width as u32, height as u32))?,
            ColorType::Rgb => self.encode_image(RgbImage(data, width as u32, height as u32))?,
            ColorType::Rgba => self.encode_image(RgbaImage(data, width as u32, height as u32))?,
            ColorType::Bgr => self.encode_image(BgrImage(data, width as u32, height as u32))?,
            ColorType::Bgra => self.encode_image(BgraImage(data, width as u32, height as u32))?,
            ColorType::Ycbcr => self.encode_image(YCbCrImage(data, width as u32, height as u32))?,
            ColorType::Cmyk => self.encode_image(CmykImage(data, width as u32, height as u32))?,
            ColorType::CmykAsYcck => self.encode_image(CmykAsYcckImage(data, width as u32, height as u32))?,
            ColorType::Ycck => self.encode_image(YcckImage(data, width as u32, height as u32))?,
        }

        Ok(())
    }

    fn init_components(&mut self, color: JpegColorType) {
        match color {
            JpegColorType::Luma => {
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

    pub fn encode_image<I: ImageBuffer>(
        &mut self,
        image: I,
    ) -> IOResult<()> {
        let jpeg_color_type = image.get_jpeg_color_type();
        self.init_components(jpeg_color_type);

        self.writer.write_marker(Marker::SOI)?;

        self.writer.write_header(&self.density)?;

        if jpeg_color_type == JpegColorType::Cmyk {
            //Set ColorTransform info to "Unknown"
            let app_14 = b"Adobe\0\0\0\0\0\0\0";
            self.writer.write_segment(Marker::APP(14), app_14.as_ref())?;
        } else if jpeg_color_type == JpegColorType::Ycck {
            //Set ColorTransform info to YCCK
            let app_14 = b"Adobe\0\0\0\0\0\0\x02";
            self.writer.write_segment(Marker::APP(14), app_14.as_ref())?;
        }

        if self.progressive_scans != 0 {
            self.encode_image_progressive(image)?;
        } else if self.optimize_huffman_table || self.horizontal_sampling_factor > 2 || self.vertical_sampling_factor > 2 {
            // Interleaved mode is only supported with h/v sampling factors of 1 or 2
            self.encode_image_sequential(image)?;
        } else {
            self.encode_image_interleaved(image)?;
        }

        self.writer.write_marker(Marker::EOI)?;

        Ok(())
    }

    fn write_frame_header<I: ImageBuffer>(&mut self, image: &I) -> IOResult<()> {
        self.writer.write_frame_header(image.width() as u16, image.height() as u16, &self.components, self.progressive_scans != 0)?;

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

        if image.get_jpeg_color_type().get_num_components() >= 3 {
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

        Ok(())
    }

    fn encode_image_interleaved<I: ImageBuffer>(
        &mut self,
        image: I,
    ) -> IOResult<()> {
        self.write_frame_header(&image)?;
        self.writer.write_scan_header(&self.components.iter().collect::<Vec<_>>(), None)?;

        let (max_h_sampling, max_v_sampling) = self.get_max_sampling_size();

        let width = image.width();
        let height = image.height();

        let num_cols = ceil_div(width, 8 * max_h_sampling);
        let num_rows = ceil_div(height, 8 * max_v_sampling);

        let buffer_width = (num_cols * 8 * max_h_sampling) as usize;
        let buffer_size = buffer_width * 8 * max_v_sampling as usize;

        let mut row: [Vec<_>; 4] = self.init_rows(buffer_size);

        let mut prev_dc = [0i16; 4];

        for block_y in 0..num_rows {
            for r in &mut row {
                r.clear();
            }

            for y in 0..(8 * max_v_sampling) {
                let y = y + block_y * 8 * max_v_sampling;
                let y = y.min(height - 1);

                for x in 0..image.width() as u32 {
                    image.fill_buffers(x, y, &mut row);
                }
                for _ in width..buffer_width as u32 {
                    image.fill_buffers(width - 1, y, &mut row);
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
        let blocks = self.encode_blocks(&image);

        if self.optimize_huffman_table {
            self.optimize_huffman_table(&blocks);
        }

        self.write_frame_header(&image)?;

        for (i, component) in self.components.iter().enumerate() {
            self.writer.write_scan_header(&[component], None)?;

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

    fn encode_image_progressive<I: ImageBuffer>(
        &mut self,
        image: I,
    ) -> IOResult<()> {
        let blocks = self.encode_blocks(&image);

        if self.optimize_huffman_table {
            self.optimize_huffman_table(&blocks);
        }

        self.write_frame_header(&image)?;

        for (i, component) in self.components.iter().enumerate() {
            self.writer.write_scan_header(&[component], Some((0, 0)))?;

            let mut prev_dc = 0;

            for block in &blocks[i] {
                self.writer.write_dc(
                    block[0],
                    prev_dc,
                    &self.huffman_tables[component.dc_huffman_table as usize].0,
                )?;

                prev_dc = block[0];
            }

            self.writer.finalize_bit_buffer()?;
        }

        let scans = self.progressive_scans as usize - 1;

        let values_per_scan = 64 / scans;

        for scan in 0..scans {
            let start = (scan * values_per_scan).max(1);
            let end = if scan == scans - 1 {
                // Due to rounding we might need to transfer more than values_per_scan values in the last scan
                64
            } else {
                (scan + 1) * values_per_scan
            };

            for (i, component) in self.components.iter().enumerate() {
                self.writer.write_scan_header(&[component], Some((start as u8, end as u8 - 1)))?;

                for block in &blocks[i] {
                    self.writer.write_ac_block(
                        &block,
                        start,
                        end,
                        &self.huffman_tables[component.ac_huffman_table as usize].1,
                    )?;
                }

                self.writer.finalize_bit_buffer()?;
            }
        }

        Ok(())
    }

    fn encode_blocks<I: ImageBuffer>(&mut self, image: &I) -> [Vec<[i16; 64]>; 4] {
        let width = image.width();
        let height = image.height();

        let num_cols = ceil_div(width, 8);
        let num_rows = ceil_div(height, 8);

        let buffer_width = (num_cols * 8) as usize;
        let buffer_size = (num_cols * num_rows * 64) as usize;

        let mut row: [Vec<_>; 4] = self.init_rows(buffer_size);

        for y in 0..num_rows * 8 {
            let y = y.min(height - 1);

            for x in 0..width {
                image.fill_buffers(x, y, &mut row);
            }
            for _ in width..num_cols * 8 {
                image.fill_buffers(width - 1, y, &mut row);
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
        blocks
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

    fn optimize_huffman_table(&mut self, blocks: &[Vec<[i16; 64]>; 4]) {
        let max_tables = self.components.len().min(2) as u8;

        for table in 0..max_tables {
            let mut dc_freq = [0u32; 257];
            dc_freq[256] = 1;
            let mut ac_freq = [0u32; 257];
            ac_freq[256] = 1;

            for (i, component) in self.components.iter().enumerate() {
                if component.dc_huffman_table == table {
                    let mut prev_dc = 0;

                    for block in &blocks[i] {
                        let value = block[0];
                        let diff = value - prev_dc;
                        let num_bits = get_num_bits(diff);

                        dc_freq[num_bits as usize] += 1;

                        prev_dc = value;
                    }
                }

                if component.ac_huffman_table == table {
                    if self.progressive_scans > 0 {
                        let scans = self.progressive_scans as usize - 1;

                        let values_per_scan = 64 / scans;

                        for scan in 0..scans {
                            let start = (scan * values_per_scan).max(1);
                            let end = if scan == scans - 1 {
                                // Due to rounding we might need to transfer more than values_per_scan values in the last scan
                                64
                            } else {
                                (scan + 1) * values_per_scan
                            };

                            for block in &blocks[i] {
                                let mut zero_run = 0;

                                for &value in &block[start..end] {
                                    if value == 0 {
                                        zero_run += 1;
                                    } else {
                                        while zero_run > 15 {
                                            ac_freq[0xF0] += 1;
                                            zero_run -= 16;
                                        }
                                        let num_bits = get_num_bits(value);
                                        let symbol = (zero_run << 4) | num_bits;

                                        ac_freq[symbol as usize] += 1;

                                        zero_run = 0;
                                    }
                                }

                                if zero_run > 0 {
                                    ac_freq[0] += 1;
                                }
                            }
                        }
                    } else {
                        for block in &blocks[i] {
                            let mut zero_run = 0;

                            for &value in &block[1..] {
                                if value == 0 {
                                    zero_run += 1;
                                } else {
                                    while zero_run > 15 {
                                        ac_freq[0xF0] += 1;
                                        zero_run -= 16;
                                    }
                                    let num_bits = get_num_bits(value);
                                    let symbol = (zero_run << 4) | num_bits;

                                    ac_freq[symbol as usize] += 1;

                                    zero_run = 0;
                                }
                            }

                            if zero_run > 0 {
                                ac_freq[0] += 1;
                            }
                        }
                    }
                }
            }

            self.huffman_tables[table as usize] = (
                HuffmanTable::new_optimized(dc_freq),
                HuffmanTable::new_optimized(ac_freq)
            );
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

fn get_num_bits(mut value: i16) -> u8 {
    if value < 0 {
        value = -value;
    }

    let mut num_bits = 0;

    while value > 0 {
        num_bits += 1;
        value >>= 1;
    }

    num_bits
}

#[cfg(test)]
mod tests {
    use crate::encoder::get_num_bits;
    use crate::writer::get_code;

    #[test]
    fn test_get_num_bits() {
        let min_max = 2i16.pow(13);

        for value in -min_max..=min_max {
            let num_bits1 = get_num_bits(value);
            let (num_bits2, _) = get_code(value);

            assert_eq!(num_bits1, num_bits2, "Difference in num bits for value {}: {} vs {}", value, num_bits1, num_bits2);
        }
    }
}