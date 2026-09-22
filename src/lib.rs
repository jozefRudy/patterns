//! Personal pattern library: reusable, strictly-linted building blocks
//! shared across projects via a pinned git dependency.

// Allow `#[derive(SystemOne)]`-generated `::patterns::` paths to resolve when
// the derive is used inside this crate's own tests.
extern crate self as patterns;

/// Re-exported so macro-generated derives resolve without a direct `serde` dep.
pub use serde;

#[cfg(feature = "embed")]
pub mod embed;
#[cfg(feature = "embed_api")]
pub mod embed_api;
#[cfg(feature = "language")]
pub mod language;
pub mod limits;
#[cfg(feature = "llm_cli")]
pub mod llm_cli;
#[cfg(feature = "systemone")]
pub mod systemone;
// Reserved: pub mod lance_store;

/// Re-exported so embed consumers don't declare fastembed/ort separately
/// (single version, enforced).
#[cfg(feature = "embed")]
pub use fastembed;
#[cfg(feature = "embed")]
pub use ort;

// Guard: the ONNX Runtime C API level `ort` asks for is the *union* of every
// crate's `api-*` features — the highest one wins, and features can only be
// added, never subtracted. This crate links the system onnxruntime (via
// `ORT_LIB_LOCATION`) instead of ort's bundled one, so a silent bump past its
// supported API version only fails at *runtime* (`GetApi` returns null ->
// panic). Assert the known-good level here so any dependency bump fails the
// build instead.
#[cfg(feature = "embed")]
const _: () = assert!(
    ort::sys::ORT_API_VERSION == 24,
    "ort API level changed: re-check the linked onnxruntime version (deploy pins 1.26, which supports API <= 26)"
);

/// Re-exported so `#[derive(Extractable)]`/`#[derive(SystemOne)]` consumers
/// don't need own askama deps (shared `templating` sub-feature).
#[cfg(feature = "templating")]
pub use askama;

/// `#[extract(..)]`-annotated extraction structs (see `llm_cli`).
#[cfg(feature = "llm_cli")]
pub use patterns_macros::Extractable;

/// `#[derive(SystemOne)]` for annotated question structs (see `systemone`).
#[cfg(feature = "systemone")]
pub use patterns_macros::SystemOne;
