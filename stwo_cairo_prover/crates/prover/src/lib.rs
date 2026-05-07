#![feature(portable_simd, iter_array_chunks, array_chunks, raw_slice_split)]
#![allow(clippy::too_many_arguments)]

pub use stwo;

pub mod debug_tools;
pub mod prover;
pub mod utils;
pub mod witness;

#[cfg(feature = "cuda-backend")]
pub mod cuda_prover;
#[cfg(feature = "cuda-backend")]
pub use cuda_prover::prove_cairo_cuda;
