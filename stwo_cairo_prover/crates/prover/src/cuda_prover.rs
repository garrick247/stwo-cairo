//! CUDA-backed prove path for stwo-cairo.

use std::sync::Arc;

use cairo_air::cairo_components::CairoComponents;
use cairo_air::claims::lookup_sum;
use cairo_air::relations::CommonLookupElements;
use cairo_air::verifier::INTERACTION_POW_BITS;
use cairo_air::CairoProof;
use num_traits::Zero;
use stwo::core::channel::{Channel, MerkleChannel};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::fields::ExtensionOf;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::proof_of_work::GrindOps;
use stwo::core::utils::MaybeOwned;
use stwo::core::vcs_lifted::merkle_hasher::MerkleHasherLifted;
use stwo::prover::backend::{BackendForChannel, Column, ColumnOps};
use stwo::prover::mempool::BaseColumnPool;
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove_ex, CommitmentSchemeProver, CommitmentTreeProver, ComponentProver, ProvingError};
use stwo_cairo_adapter::ProverInput;
use stwo_cairo_serialize::CairoSerialize;
use tracing::{event, span, Level};
use vortex_cuda_backend::{CudaBackend, CudaColumn, CudaFrameworkComponentRef};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedTraceVariant;
use stwo::core::poly::circle::CircleDomain;
use stwo_cairo_common::preprocessed_columns::pedersen::{
    PedersenPoints, PEDERSEN_TABLE_18, PEDERSEN_TABLE_N_COLUMNS,
};
use stwo_cairo_common::preprocessed_columns::preprocessed_trace::PreProcessedColumn;
use vortexstark::device::DeviceBuffer;

/// Pre-uploaded GPU buffers for the WINDOW_BITS=18 Pedersen point table.
/// LazyLock evaluation (~1.0s on first access) + 56 H->D uploads happen
/// once per process; subsequent calls D->D clone (a few ms total).
/// Force initialization of the GPU-resident Pedersen point table. Call once
/// at process startup (before any prove) to amortize the ~1 s LazyLock init
/// + H->D upload out of the first prove's critical path.
pub fn prewarm_pedersen_gpu() {
    let _ = pedersen_gpu_data();
}

fn pedersen_gpu_data() -> &'static Vec<DeviceBuffer<u32>> {
    static CACHE: OnceLock<Vec<DeviceBuffer<u32>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        // Force LazyLock init.
        let _ = &*PEDERSEN_TABLE_18;
        (0..PEDERSEN_TABLE_N_COLUMNS)
            .map(|i| {
                let pp = PedersenPoints::<18>::new(i);
                let raw: Vec<u32> = pp.get_data().iter().map(|f| f.0).collect();
                DeviceBuffer::from_host(&raw)
            })
            .collect()
    })
}

/// CUDA-native column generation. For Pedersen columns we skip the SimdBackend
/// `BaseColumn::from_cpu` SIMD pack and the `simd_to_cuda_eval` round-trip and
/// upload directly from the LazyLock-cached `Vec<M31>`. All other column types
/// fall through to `gen_column_simd` + `simd_to_cuda_eval`.
fn gen_column_cuda(
    c: &dyn PreProcessedColumn,
) -> stwo::prover::poly::circle::CircleEvaluation<CudaBackend, BaseField, stwo::prover::poly::BitReversedOrder>
{
    let id = c.id().id;
    if let Some(idx_str) = id.strip_prefix("pedersen_points_") {
        if let Ok(idx) = idx_str.parse::<usize>() {
            if idx < PEDERSEN_TABLE_N_COLUMNS {
                let cached = &pedersen_gpu_data()[idx];
                let buf = cached.clone();
                let len = 1usize << c.log_size();
                use vortex_cuda_backend::CudaColumn;
                let col = CudaColumn::<BaseField>::from_device_buffer(buf, len);
                let domain: CircleDomain = CanonicCoset::new(c.log_size()).circle_domain();
                return stwo::prover::poly::circle::CircleEvaluation::new(domain, col);
            }
        }
    }
    simd_to_cuda_eval(c.gen_column_simd())
}


// ---------------------------------------------------------------------------
// Preprocessed-trace polynomial cache
// ---------------------------------------------------------------------------
//
// gen_preproc_trace + SIMD->CUDA convert + interpolate_columns together
// take ~2.1s and are deterministic for a given PreProcessedTraceVariant.
// Cache the post-interpolate polys; subsequent proves deep-clone (D2D
// memcpy ~few ms) instead of recomputing.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

fn variant_key(v: PreProcessedTraceVariant) -> u8 {
    match v {
        PreProcessedTraceVariant::Canonical => 0,
        PreProcessedTraceVariant::CanonicalWithoutPedersen => 1,
        PreProcessedTraceVariant::CanonicalSmall => 2,
    }
}

#[allow(clippy::type_complexity)]
fn preproc_polys_cache() -> &'static Mutex<HashMap<u8, Arc<Vec<stwo::prover::poly::circle::CircleCoefficients<CudaBackend>>>>> {
    static CACHE: OnceLock<Mutex<HashMap<u8, Arc<Vec<stwo::prover::poly::circle::CircleCoefficients<CudaBackend>>>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub static PREPROC_CACHE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PREPROC_CACHE_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn preproc_cache_stats_take() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let hits = PREPROC_CACHE_HITS.swap(0, Relaxed);
    let misses = PREPROC_CACHE_MISSES.swap(0, Relaxed);
    (hits, misses)
}


use crate::prover::ProverParameters;
use crate::witness::cairo::create_cairo_claim_generator;
use crate::witness::utils::witness_trace_cells;

macro_rules! push_if_some {
    ($vec:ident, $components:ident, $field:ident) => {
        if let Some(c) = &$components.$field {
            $vec.push(Box::new(CudaFrameworkComponentRef(c))
                as Box<dyn ComponentProver<CudaBackend>>);
        }
    };
}

fn cuda_cairo_provers<'a>(
    components: &'a CairoComponents,
) -> Vec<Box<dyn ComponentProver<CudaBackend> + 'a>> {
    let mut wrappers: Vec<Box<dyn ComponentProver<CudaBackend>>> = vec![];
    push_if_some!(wrappers, components, add_opcode);
    push_if_some!(wrappers, components, add_opcode_small);
    push_if_some!(wrappers, components, add_ap_opcode);
    push_if_some!(wrappers, components, assert_eq_opcode);
    push_if_some!(wrappers, components, assert_eq_opcode_imm);
    push_if_some!(wrappers, components, assert_eq_opcode_double_deref);
    push_if_some!(wrappers, components, blake_compress_opcode);
    push_if_some!(wrappers, components, call_opcode_abs);
    push_if_some!(wrappers, components, call_opcode_rel_imm);
    push_if_some!(wrappers, components, generic_opcode);
    push_if_some!(wrappers, components, jnz_opcode_non_taken);
    push_if_some!(wrappers, components, jnz_opcode_taken);
    push_if_some!(wrappers, components, jump_opcode_abs);
    push_if_some!(wrappers, components, jump_opcode_double_deref);
    push_if_some!(wrappers, components, jump_opcode_rel);
    push_if_some!(wrappers, components, jump_opcode_rel_imm);
    push_if_some!(wrappers, components, mul_opcode);
    push_if_some!(wrappers, components, mul_opcode_small);
    push_if_some!(wrappers, components, qm_31_add_mul_opcode);
    push_if_some!(wrappers, components, ret_opcode);
    push_if_some!(wrappers, components, verify_instruction);
    push_if_some!(wrappers, components, blake_round);
    push_if_some!(wrappers, components, blake_g);
    push_if_some!(wrappers, components, blake_round_sigma);
    push_if_some!(wrappers, components, triple_xor_32);
    push_if_some!(wrappers, components, verify_bitwise_xor_12);
    push_if_some!(wrappers, components, add_mod_builtin);
    push_if_some!(wrappers, components, bitwise_builtin);
    push_if_some!(wrappers, components, mul_mod_builtin);
    push_if_some!(wrappers, components, pedersen_builtin);
    push_if_some!(wrappers, components, pedersen_builtin_narrow_windows);
    push_if_some!(wrappers, components, poseidon_builtin);
    push_if_some!(wrappers, components, range_check96_builtin);
    push_if_some!(wrappers, components, range_check_builtin);
    push_if_some!(wrappers, components, ec_op_builtin);
    push_if_some!(wrappers, components, partial_ec_mul_generic);
    push_if_some!(wrappers, components, pedersen_aggregator_window_bits_18);
    push_if_some!(wrappers, components, partial_ec_mul_window_bits_18);
    push_if_some!(wrappers, components, pedersen_points_table_window_bits_18);
    push_if_some!(wrappers, components, pedersen_aggregator_window_bits_9);
    push_if_some!(wrappers, components, partial_ec_mul_window_bits_9);
    push_if_some!(wrappers, components, pedersen_points_table_window_bits_9);
    push_if_some!(wrappers, components, poseidon_aggregator);
    push_if_some!(wrappers, components, poseidon_3_partial_rounds_chain);
    push_if_some!(wrappers, components, poseidon_full_round_chain);
    push_if_some!(wrappers, components, cube_252);
    push_if_some!(wrappers, components, poseidon_round_keys);
    push_if_some!(wrappers, components, range_check_252_width_27);
    push_if_some!(wrappers, components, memory_address_to_id);
    for c in &components.memory_id_to_big {
        wrappers.push(Box::new(CudaFrameworkComponentRef(c)) as Box<dyn ComponentProver<CudaBackend>>);
    }
    push_if_some!(wrappers, components, memory_id_to_small);
    push_if_some!(wrappers, components, range_check_6);
    push_if_some!(wrappers, components, range_check_8);
    push_if_some!(wrappers, components, range_check_11);
    push_if_some!(wrappers, components, range_check_12);
    push_if_some!(wrappers, components, range_check_18);
    push_if_some!(wrappers, components, range_check_20);
    push_if_some!(wrappers, components, range_check_4_3);
    push_if_some!(wrappers, components, range_check_4_4);
    push_if_some!(wrappers, components, range_check_9_9);
    push_if_some!(wrappers, components, range_check_7_2_5);
    push_if_some!(wrappers, components, range_check_3_6_6_3);
    push_if_some!(wrappers, components, range_check_4_4_4_4);
    push_if_some!(wrappers, components, range_check_3_3_3_3_3);
    push_if_some!(wrappers, components, verify_bitwise_xor_4);
    push_if_some!(wrappers, components, verify_bitwise_xor_7);
    push_if_some!(wrappers, components, verify_bitwise_xor_8);
    push_if_some!(wrappers, components, verify_bitwise_xor_9);
    wrappers
}

fn simd_to_cuda_eval<F>(
    e: CircleEvaluation<stwo::prover::backend::simd::SimdBackend, F, BitReversedOrder>,
) -> CircleEvaluation<CudaBackend, F, BitReversedOrder>
where
    F: Copy + Send + Sync + 'static + ExtensionOf<BaseField>,
    stwo::prover::backend::simd::SimdBackend: ColumnOps<F>,
    CudaBackend: ColumnOps<F, Column = CudaColumn<F>>,
    CudaColumn<F>: FromIterator<F>,
{
    let domain = e.domain;
    let cuda_col: CudaColumn<F> = e.values.to_cpu().into_iter().collect();
    CircleEvaluation::new(domain, cuda_col)
}

pub fn prove_cairo_cuda<MC: MerkleChannel>(
    input: ProverInput,
    prover_params: ProverParameters,
) -> Result<CairoProof<MC::H>, ProvingError>
where
    CudaBackend: BackendForChannel<MC>,
    MC::H: MerkleHasherLifted,
    <MC::H as MerkleHasherLifted>::Hash: CairoSerialize,
{
    let _span = span!(Level::INFO, "prove_cairo_cuda").entered();
    let ProverParameters {
        channel_hash: _,
        channel_salt,
        pcs_config,
        preprocessed_trace: preprocessed_trace_variant,
        store_polynomials_coefficients,
        include_all_preprocessed_columns,
        opt_n_id_to_big_components,
    } = prover_params;

    let preprocessed_trace = Arc::new(preprocessed_trace_variant.to_preprocessed_trace());

    let cairo_claim_generator = create_cairo_claim_generator(input, preprocessed_trace.clone());
    let span = span!(Level::INFO, "Write Base trace").entered();
    let (simd_trace_evals, claim, interaction_generator) =
        cairo_claim_generator.write_trace(opt_n_id_to_big_components);
    span.exit();

    let span = span!(Level::INFO, "SIMD->CUDA trace transfer").entered();
    let trace_evals: Vec<CircleEvaluation<CudaBackend, _, _>> =
        simd_trace_evals.into_iter().map(simd_to_cuda_eval).collect();
    span.exit();

    let max_log_trace_size = claim.log_sizes().iter().flatten().fold(
        preprocessed_trace_variant.max_log_trace_size(),
        |max, &size| max.max(size),
    );
    let cairo_air_log_degree_bound = 1u32;
    let mut max_domain_log_size = max_log_trace_size
        + std::cmp::max(
            cairo_air_log_degree_bound,
            pcs_config.fri_config.log_blowup_factor,
        );
    if let Some(lifting_log_size) = pcs_config.lifting_log_size {
        if lifting_log_size < max_domain_log_size {
            return Err(ProvingError::InvalidLiftingLogSize(
                stwo::core::pcs::utils::InvalidLiftingLogSizeError {
                    lifting_log_size,
                    min_log_size: max_domain_log_size,
                },
            ));
        }
        max_domain_log_size = lifting_log_size;
    }

    let span = span!(Level::INFO, "Precompute Twiddles (CUDA)").entered();
    let twiddles = CudaBackend::precompute_twiddles(
        CanonicCoset::try_new(max_domain_log_size)?
            .circle_domain()
            .half_coset,
    );
    span.exit();

    let span = span!(Level::INFO, "Preprocessed trace commit (CUDA)").entered();
    use crate::witness::preprocessed_trace::gen_trace as gen_preproc_trace;
    let preprocessed_trace_polys: Vec<stwo::prover::poly::circle::CircleCoefficients<CudaBackend>> = {
        let key = variant_key(preprocessed_trace_variant);
        let cached_arc = {
            let cache = preproc_polys_cache().lock().unwrap();
            cache.get(&key).cloned()
        };
        if let Some(arc_polys) = cached_arc {
            // Hit: deep-clone each poly (D2D memcpy ~few ms total).
            let s = span!(Level::INFO, "Preproc: cache hit (D2D clone)").entered();
            PREPROC_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let cloned: Vec<_> = arc_polys.iter().cloned().collect();
            s.exit();
            cloned
        } else {
            PREPROC_CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let s1 = span!(Level::INFO, "Preproc: gen_trace_cuda (Pedersen direct upload)").entered();
            let preproc_cuda: Vec<CircleEvaluation<CudaBackend, _, _>> = preprocessed_trace
                .columns
                .iter()
                .map(|c| gen_column_cuda(c.as_ref()))
                .collect();
            s1.exit();
            let s3 = span!(Level::INFO, "Preproc: interpolate_columns (CUDA)").entered();
            let polys = CudaBackend::interpolate_columns(preproc_cuda, &twiddles);
            s3.exit();
            // Store a clone in the cache so we can return owned polys.
            let s4 = span!(Level::INFO, "Preproc: cache store (D2D clone)").entered();
            let cached_clone: Vec<_> = polys.iter().cloned().collect();
            preproc_polys_cache()
                .lock()
                .unwrap()
                .insert(key, Arc::new(cached_clone));
            s4.exit();
            polys
        }
    };

    let base_column_pool = BaseColumnPool::<CudaBackend>::new();
    let s4 = span!(Level::INFO, "Preproc: CommitmentTreeProver::new").entered();
    let preprocessed_tree = MaybeOwned::Owned(CommitmentTreeProver::<CudaBackend, MC>::new(
        preprocessed_trace_polys,
        pcs_config.fri_config.log_blowup_factor,
        &twiddles,
        store_polynomials_coefficients,
        pcs_config.lifting_log_size,
        &base_column_pool,
    ));
    s4.exit();
    span.exit();

    let channel = &mut <MC::C as Default>::default();
    channel.mix_felts(&[channel_salt.into()]);
    pcs_config.mix_into(channel);
    let mut commitment_scheme = CommitmentSchemeProver::<CudaBackend, MC>::with_memory_pool(
        pcs_config,
        &twiddles,
        &base_column_pool,
    );
    if store_polynomials_coefficients {
        commitment_scheme.set_store_polynomials_coefficients();
    }
    commitment_scheme.commit_tree(preprocessed_tree, channel);

    claim.mix_into::<MC>(channel);
    let span = span!(Level::INFO, "Base trace commit (CUDA)").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(trace_evals);
    tree_builder.commit(channel);
    span.exit();

    let interaction_pow = CudaBackend::grind(channel, INTERACTION_POW_BITS);
    channel.mix_u64(interaction_pow);
    let interaction_elements = CommonLookupElements::draw(channel);

    let span = span!(Level::INFO, "Interaction trace + SIMD->CUDA").entered();
    let (simd_inter_evals, interaction_claim) =
        interaction_generator.write_interaction_trace(&interaction_elements);
    let interaction_trace_evals: Vec<CircleEvaluation<CudaBackend, _, _>> =
        simd_inter_evals.into_iter().map(simd_to_cuda_eval).collect();
    span.exit();

    tracing::info!(
        "Witness trace cells: {:?}",
        witness_trace_cells(&claim, &preprocessed_trace)
    );
    debug_assert_eq!(
        lookup_sum(&claim, &interaction_elements, &interaction_claim),
        SecureField::zero()
    );
    interaction_claim.mix_into(channel);

    let span = span!(Level::INFO, "Interaction trace commit (CUDA)").entered();
    let mut tree_builder = commitment_scheme.tree_builder();
    tree_builder.extend_evals(interaction_trace_evals);
    tree_builder.commit(channel);
    span.exit();

    let component_builder = CairoComponents::new(
        &claim,
        &interaction_elements,
        &interaction_claim,
        &preprocessed_trace.ids(),
    );
    let cuda_provers_owned = cuda_cairo_provers(&component_builder);
    let cuda_provers_refs: Vec<&dyn ComponentProver<CudaBackend>> =
        cuda_provers_owned.iter().map(|b| &**b).collect();

    let span = span!(Level::INFO, "Prove STARKs (CUDA)").entered();
    let proof = prove_ex::<CudaBackend, _>(
        &cuda_provers_refs,
        channel,
        commitment_scheme,
        include_all_preprocessed_columns,
    )?;
    span.exit();

    event!(name: "component_info", Level::DEBUG, "Components: {}", component_builder);

    Ok(CairoProof {
        claim,
        interaction_pow,
        interaction_claim,
        extended_stark_proof: proof,
        channel_salt,
        preprocessed_trace_variant,
    })
}