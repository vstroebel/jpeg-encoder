mod fdct;
mod ycbcr;

#[cfg(not(feature = "use_wide"))]
use crate::encoder::{AlignedBlock, Operations};
pub use fdct::fdct_avx2;
pub use ycbcr::*;

#[cfg(not(feature = "use_wide"))]
pub(crate) struct AVX2Operations;

#[cfg(not(feature = "use_wide"))]
impl Operations for AVX2Operations {
    #[inline(always)]
    fn fdct(data: &mut AlignedBlock) {
        fdct_avx2(data);
    }
}
