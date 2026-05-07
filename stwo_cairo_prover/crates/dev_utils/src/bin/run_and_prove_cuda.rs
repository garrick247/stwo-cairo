//! CUDA-backed run_and_prove binary.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_dev_utils::vm_utils::{run_and_adapt, ProgramType};
use stwo_cairo_prover::prove_cairo_cuda;
use stwo_cairo_prover::prover::{ChannelHash, ProverParameters};
use cairo_vm::types::layout_name::LayoutName;
use tracing::{span, Level};
use tracing_subscriber::fmt::format::FmtSpan;

#[derive(Parser, Debug)]
struct Args {
    #[structopt(long = "program")]
    program: PathBuf,
    #[arg(long = "program_type", default_value = "json")]
    program_type: ProgramType,
    #[arg(long = "program_arguments_file")]
    program_arguments_file: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::ENTER | FmtSpan::CLOSE)
        .init();
    let _span = span!(Level::INFO, "run_and_prove_cuda").entered();

    vortexstark::cuda::ffi::init_memory_pool();

    let t0 = Instant::now();
    let prover_input = run_and_adapt(
        &args.program,
        args.program_type,
        LayoutName::all_cairo_stwo,
        args.program_arguments_file.as_ref(),
    )?;
    let t_adapt = t0.elapsed();
    eprintln!("[time] VM run + adapt: {:?}", t_adapt);

    let prover_params = ProverParameters {
        channel_hash: ChannelHash::Blake2s,
        channel_salt: 0,
        pcs_config: PcsConfig {
            pow_bits: 26,
            fri_config: FriConfig {
                log_last_layer_degree_bound: 0,
                log_blowup_factor: 1,
                n_queries: 70,
                fold_step: 1,
            },
            lifting_log_size: None,
        },
        preprocessed_trace: PreProcessedTraceVariant::CanonicalWithoutPedersen,
        store_polynomials_coefficients: false,
        include_all_preprocessed_columns: false,
        opt_n_id_to_big_components: None,
    };

    let t0 = Instant::now();
    let _proof = prove_cairo_cuda::<Blake2sMerkleChannel>(prover_input, prover_params)
        .expect("prove_cairo_cuda failed");
    let t_prove = t0.elapsed();
    eprintln!();
    eprintln!("============================================================");
    eprintln!("[time] prove_cairo_cuda: {:?}", t_prove);
    eprintln!("[time] total (adapt + prove): {:?}", t_adapt + t_prove);
    eprintln!("============================================================");

    Ok(())
}