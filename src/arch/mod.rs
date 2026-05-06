//! Per-architecture kernels.
//!
//! Each backend submodule (`scalar`, `neon`, `x86_64`) re-exports the
//! same set of inner modules — `color`, `dct`, `quant`, `huffman` —
//! with bit-exact-equivalent function signatures. The crate as a whole
//! talks to `arch::backend::*`, which is an alias for the active
//! backend selected at compile time.
//!
//! Selection rules:
//!
//! - `aarch64` + `not(force-scalar)`  → `neon`
//! - `x86_64`  + `not(force-scalar)`  → `x86_64` (AVX2; runtime
//!   `is_x86_feature_detected!` falls back to scalar on non-AVX2 CPUs)
//! - everything else                  → `scalar`
//!
//! `arch::x86_64::huffman` intentionally stays scalar — its AC zero-scan
//! helper autovectorizes well in the trivial scalar form, and the
//! entropy loop is too branchy for SIMD to win. See BENCH.md.
//!
//! ## Adding a new backend
//!
//! 1. Create `arch/<name>.rs` with four inline modules (`color`, `dct`,
//!    `quant`, `huffman`), each exposing the kernel functions named in
//!    `arch::scalar` (use `pub use crate::arch::scalar::<kernel>::*;`
//!    for any kernel you don't override).
//! 2. Declare the module here under the appropriate `#[cfg(target_arch
//!    = "...")]` gate.
//! 3. Add a `pub use <name> as backend;` cfg arm so it gets selected.
//! 4. Update `bin/bench.rs`'s `arch` label to print the right string.
//! 5. Mirror the cross-check tests pattern from `arch::neon::tests` /
//!    `arch::x86_64::tests` (compare each kernel against scalar on a
//!    panel of inputs).

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
pub use neon as backend;

#[cfg(all(target_arch = "x86_64", not(feature = "force-scalar")))]
pub use x86_64 as backend;

#[cfg(any(
    feature = "force-scalar",
    not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
pub use scalar as backend;
