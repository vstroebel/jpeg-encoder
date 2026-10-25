mod fdct;
mod ycbcr;

#[cfg(not(feature = "fearless_simd"))]
use crate::encoder::{AlignedBlock, Operations};
pub use fdct::fdct_avx2;
pub use ycbcr::*;

#[cfg(not(feature = "fearless_simd"))]
pub(crate) struct AVX2Operations;

#[cfg(not(feature = "fearless_simd"))]
impl Operations for AVX2Operations {
    #[inline(always)]
    fn fdct(data: &mut AlignedBlock) {
        fdct_avx2(data);
    }
}
