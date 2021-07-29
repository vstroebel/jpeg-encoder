mod ycbcr;
mod fdct;

pub(crate) use ycbcr::*;
pub(crate) use fdct::fdct_avx2;
use crate::encoder::Operations;

pub(crate) struct AVX2Operations;

impl Operations for AVX2Operations {
    #[inline(always)]
    fn fdct(data: &mut [i16; 64]) {
        fdct_avx2(data)
    }
}