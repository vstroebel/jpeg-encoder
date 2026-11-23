mod fdct;
mod ycbcr;

use crate::encoder::{AlignedBlock, Operations};
pub use fdct::fdct_avx2;
pub use ycbcr::*;

pub(crate) struct AVX2Operations;

impl Operations for AVX2Operations {
    #[inline(always)]
    fn fdct(data: &mut AlignedBlock) {
        // SAFETY: the encoder only selects AVX2Operations after is_x86_feature_detected!("avx2").
        unsafe { fdct_avx2(data) };
    }
}
