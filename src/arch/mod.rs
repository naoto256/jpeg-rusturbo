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
//! - `x86_64`  + `not(force-scalar)`  → `x86_64` (AVX2 with scalar fallback
//!                                                at runtime when AVX2 absent)
//! - everything else                  → `scalar`
//!
//! On x86_64, individual kernels in `arch::x86_64` may currently delegate
//! to scalar pending their AVX2 port (incremental Step 3 work).

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
