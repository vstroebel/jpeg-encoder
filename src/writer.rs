use crate::marker::Marker;

use std::io::{Write, Result as IOResult};

#[derive(Debug)]
pub enum Density {
    None,
    Inch { x: u16, y: u16 },
    Centimeter { x: u16, y: u16 },
}

pub(crate) struct JfifWriter<W: Write> {
    w: W,
}

impl<W: Write> JfifWriter<W> {
    pub fn new(w: W) -> Self {
        JfifWriter {
            w,
        }
    }

    pub fn write(&mut self, buf: &[u8]) -> IOResult<()> {
        self.w.write_all(buf)
    }

    pub fn write_u8(&mut self, value: u8) -> IOResult<()> {
        self.w.write_all(&[value])
    }

    pub fn write_u16(&mut self, value: u16) -> IOResult<()> {
        self.w.write_all(&value.to_be_bytes())
    }

    pub fn write_marker(&mut self, marker: Marker) -> IOResult<()> {
        self.write(&[0xFF, marker.into()])
    }

    pub fn write_segment(&mut self, marker: Marker, data: &[u8]) -> IOResult<()> {
        self.write_marker(marker)?;
        self.write_u16(data.len() as u16 + 2)?;
        self.write(data)?;

        Ok(())
    }

    pub fn write_header(&mut self, density: &Density) -> IOResult<()> {
        self.write_marker(Marker::APP(0))?;
        self.write_u16(16)?;

        self.write(b"JFIF\0")?;
        self.write(&[0x01, 0x02])?;

        match *density {
            Density::None => {
                self.write_u8(0x00)?;
                self.write_u16(1)?;
                self.write_u16(1)?;
            }
            Density::Inch { x, y } => {
                self.write_u8(0x01)?;
                self.write_u16(x)?;
                self.write_u16(y)?;
            }
            Density::Centimeter { x, y } => {
                self.write_u8(0x02)?;
                self.write_u16(x)?;
                self.write_u16(y)?;
            }
        }

        self.write(&[0x00, 0x00])
    }
}

