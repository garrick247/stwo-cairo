# ⚡ S-two Cairo ⚡

Prove Cairo programs with the blazing-fast [S-Two prover](https://github.com/starkware-libs/stwo), powered by the cryptographic breakthrough of [Circle STARKs](https://eprint.iacr.org/2024/278).

* [Prerequisites](#prerequisites)
* [`scarb-prove`](#scarb-prove)

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install/)
- [Scarb](https://docs.swmansion.com/scarb/download.html)
  - The recommended installation method is using [asdf](https://asdf-vm.com/)
  - Make sure to use version 2.10.0 and onwards, and preferably the latest nightly version.
  
    To use the latest nightly version, run:
    
    ```
    asdf set -u scarb latest:nightly
    ```

## Installation

This repository now focuses on the prover and verifier crates under `stwo_cairo_prover/` and `stwo_cairo_verifier/`. The former `cairo-prove` CLI has been removed. The equivalent utility is now provided in `proving-utils`: https://github.com/starkware-libs/proving-utils

## `scarb prove`

As of Scarb version 2.10.0, `scarb prove` can be used instead of manually building and running `stwo-cairo`.

However, `scarb prove` is still a work in progress, and using `stwo-cairo` directly is preferable for now.

---

## Experimental CUDA backend (this branch only)

This `cuda-backend-poc` branch routes `prove_cairo` through a GPU backend
from [garrick247/VortexSTARK](https://github.com/garrick247/VortexSTARK).
The proof output is **byte-identical** to the upstream SIMD backend and
verifies through the standard `verify_cairo_ex`.

### Prerequisites

- CUDA toolkit 12+ (`nvcc` on PATH)
- NVIDIA GPU with compute capability sm_120 (Blackwell — RTX 5090) or
  sm_89 (Ada Lovelace — RTX 4090). The kernels currently target `sm_120`;
  to build for sm_89 set `RUSTFLAGS="--cfg compute=89"` and adjust the
  `-arch` flag in VortexSTARK's `build.rs`.
- ~10 GB free VRAM for the test_data programs; large Cairo programs
  need more (Pedersen-heavy traces use ~20 GB at the sizes tested).
- Rust 1.85+ (stable).

### Build and run

```bash
cd stwo_cairo_prover
RUSTFLAGS="-C target-cpu=native"   cargo build --release --bin run_and_prove_cuda --features cuda-backend

# Smallest test program (~1 sec warm prove on RTX 5090):
./target/release/run_and_prove_cuda   --program test_data/test_prove_verify_ret_opcode/compiled.json   --iterations 3 --verify
```

`--iterations N` runs prove N times in the same process — the first is
cold, the rest are warm (caches populated). `--verify` runs the standard
CPU verifier on the produced proof; it should print `PROOF VALID`.

### Measured performance

RTX 5090 + Core Ultra 9 285K, all four `test_data/test_prove_verify_*`
programs, `store_polynomials_coefficients=true` (the binary now sets this
by default), warm prove vs CPU SimdBackend `run_and_prove`:

| Program | CUDA warm | CPU | CUDA / CPU |
|---|---:|---:|---:|
| ret_opcode | 0.73s | 4.17s | **5.7x faster** |
| range_check_bits_128 | 1.18s | 4.24s | **3.6x faster** |
| bitwise_builtin | 1.24s | 4.32s | **3.5x faster** |
| pedersen_builtin | 5.37s | 7.65s | **1.4x faster** |

Cold-prove on the first iteration is slower because several deterministic
GPU buffers are computed and cached on first use. The cache is keyed by
domain log_size, so reusing the same prover process across many proofs
amortizes this.

### Status

Active proof-of-concept. The Rust API exposed by the `cuda-backend`
feature is unstable and may change. Issues and benchmarks-on-your-circuit
welcome — see [garrick247/VortexSTARK](https://github.com/garrick247/VortexSTARK).
