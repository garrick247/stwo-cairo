//! Load a serialized ProverInput and prove via CudaBackend.

use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use serde_json::from_reader;
use stwo::core::fri::FriConfig;
use stwo::core::pcs::PcsConfig;
use stwo::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo_cairo_prover::prove_cairo_cuda;
use stwo_cairo_prover::prover::{ChannelHash, ProverParameters};
use cairo_air::verifier::verify_cairo_ex;
use tracing::{span, Level};
use tracing_subscriber::fmt::format::FmtSpan;

#[derive(Parser, Debug)]
struct Args {
    #[clap(long = "prover_input_path")]
    prover_input_path: PathBuf,
    #[clap(long = "verify", action)]
    verify: bool,
    /// Number of prove iterations in this process (for measuring cache warmup).
    #[clap(long = "iterations", default_value_t = 1)]
    iterations: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::ENTER | FmtSpan::CLOSE)
        .init();
    let _span = span!(Level::INFO, "prove_cuda_from_input").entered();

    vortexstark::cuda::ffi::init_memory_pool();

    let t0 = Instant::now();
    let prover_input: ProverInput = from_reader(File::open(&args.prover_input_path)?)?;
    let t_load = t0.elapsed();
    eprintln!("[time] load_input: {:?}", t_load);

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

    let mut last_proof = None;
    let mut t_prove_first = std::time::Duration::ZERO;
    let mut t_prove_last = std::time::Duration::ZERO;
    for iter in 0..args.iterations {
        // Re-read prover_input each iteration since prove_cairo_cuda consumes it.
        let pi: ProverInput = from_reader(File::open(&args.prover_input_path)?)?;
        let t0 = Instant::now();
        let proof = prove_cairo_cuda::<Blake2sMerkleChannel>(pi, prover_params)
            .expect("prove_cairo_cuda failed");
        let t_prove = t0.elapsed();
        let (n_calls, ns) = stwo_cairo_prover::eval_at_point_stats_take();
        let (cache_hits, cache_misses) = stwo_cairo_prover::preproc_cache_stats_take();
        eprintln!(
            "[iter {}] prove={:?} eval_at_point_calls={} cache_hits={} cache_misses={}",
            iter, t_prove, n_calls, cache_hits, cache_misses,
        );
        if iter == 0 {
            t_prove_first = t_prove;
        }
        t_prove_last = t_prove;
        last_proof = Some(proof);
    }
    eprintln!("[time] prove_cairo_cuda: {:?}", t_prove_last);
    let proof = last_proof.expect("at least one iteration");

    let proof_json_path = args.prover_input_path.with_extension("cuda_proof.json");
    use cairo_air::utils::{serialize_proof_to_file, ProofFormat};
    serialize_proof_to_file(&proof, &proof_json_path, ProofFormat::Json)?;
    eprintln!("[proof] wrote {}", proof_json_path.display());

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

    eprintln!(
        "[summary] load: {:?} prove_first: {:?} prove_last: {:?}",
        t_load, t_prove_first, t_prove_last
    );
    Ok(())
}