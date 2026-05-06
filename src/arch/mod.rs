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
//! - everything else                  → `scalar`
//!
//! `x86_64` has its own module reserved for an upcoming AVX2 port; it
//! is currently empty and not wired into the dispatch.

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
pub use neon as backend;

#[cfg(not(all(target_arch = "aarch64", not(feature = "force-scalar"))))]
pub use scalar as backend;
