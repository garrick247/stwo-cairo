//! CUDA-backed run_and_prove binary.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use cairo_air::verifier::verify_cairo_ex;
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
    #[clap(long = "verify", action)]
    verify: bool,
    /// Re-prove this many times in the same process (cold = first, warm = rest).
    #[clap(long = "iterations", default_value_t = 1)]
    iterations: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::ENTER | FmtSpan::CLOSE)
        .init();
    let _span = span!(Level::INFO, "run_and_prove_cuda").entered();

    vortexstark::cuda::ffi::init_memory_pool();

    let t0 = Instant::now();
    let _warmup_pi = run_and_adapt(
        &args.program,
        args.program_type.clone(),
        LayoutName::all_cairo_stwo,
        args.program_arguments_file.as_ref(),
    )?;
    let t_adapt = t0.elapsed();
    eprintln!("[time] VM run + adapt: {:?}", t_adapt);
    drop(_warmup_pi);

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
        preprocessed_trace: PreProcessedTraceVariant::Canonical,
        store_polynomials_coefficients: true,
        include_all_preprocessed_columns: false,
        opt_n_id_to_big_components: None,
    };

    let mut t_prove_first = std::time::Duration::ZERO;
    let mut t_prove_last = std::time::Duration::ZERO;
    let mut last_proof = None;
    for iter in 0..args.iterations {
        let pi = run_and_adapt(
            &args.program,
            args.program_type.clone(),
            LayoutName::all_cairo_stwo,
            args.program_arguments_file.as_ref(),
        )?;
        let t0 = Instant::now();
        let proof = prove_cairo_cuda::<Blake2sMerkleChannel>(pi, prover_params.clone())
            .expect("prove_cairo_cuda failed");
        let t = t0.elapsed();
        let (eap_calls, eap_ns) = stwo_cairo_prover::eval_at_point_stats_take();
        let (cache_hits, cache_misses) = stwo_cairo_prover::preproc_cache_stats_take();
        let (gpu_bc, gpu_bc_ns, cpu_bc, cpu_bc_ns) = stwo_cairo_prover::bytecode_kernel_stats_take();
        let (an_c, an_ns, qc_c, qc_ns) = stwo_cairo_prover::quotient_stats_take();
        let (fl_c, fl_ns, fc_c, fc_ns) = stwo_cairo_prover::fri_stats_take();
        let (nt_c, nt_ns, ift_c, ift_ns) = stwo_cairo_prover::ntt_stats_take();
        let (mk_c, mk_ns) = stwo_cairo_prover::merkle_stats_take();
        eprintln!(
            "[time] iter {} prove: {:?}  [stats] eap={}({:.0}ms) cache={}h/{}m bc={}g({:.0}ms)/{}c({:.0}ms) qo=an{}({:.0}ms)/qc{}({:.0}ms) fri=fl{}({:.0}ms)/fc{}({:.0}ms) ntt=fwd{}({:.0}ms)/ifft{}({:.0}ms) mk={}({:.0}ms)",
            iter, t, eap_calls, eap_ns as f64 / 1e6, cache_hits, cache_misses, gpu_bc, gpu_bc_ns as f64 / 1e6, cpu_bc, cpu_bc_ns as f64 / 1e6, an_c, an_ns as f64 / 1e6, qc_c, qc_ns as f64 / 1e6, fl_c, fl_ns as f64 / 1e6, fc_c, fc_ns as f64 / 1e6, nt_c, nt_ns as f64 / 1e6, ift_c, ift_ns as f64 / 1e6, mk_c, mk_ns as f64 / 1e6,
        );
        if iter == 0 { t_prove_first = t; }
        t_prove_last = t;
        last_proof = Some(proof);
    }
    let t_prove = t_prove_last;
    let proof = last_proof.expect("no iterations ran");

    if args.verify {
        let t0 = Instant::now();
        verify_cairo_ex::<Blake2sMerkleChannel>(proof.into(), false)
            .expect("verify_cairo failed");
        let t_verify = t0.elapsed();
        eprintln!("[time] verify_cairo: {:?}", t_verify);
        eprintln!("[result] PROOF VALID");
    } else {
        let _ = proof;
    }

    eprintln!("============================================================");
    eprintln!("[summary] adapt: {:?} prove_cold: {:?} prove_warm: {:?}", t_adapt, t_prove_first, t_prove);
    eprintln!("============================================================");

    Ok(())
}