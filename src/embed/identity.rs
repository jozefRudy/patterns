//! Model identity: the complete, comparison-ready description of what produced a set of vectors.
//!
//! `id()` is keyed only on what can reshape the token stream — the verified bytes, the spec choices,
//! the stored width and the tokenizer version. The revision pin, the repo/artifact name and the
//! execution crates (`fastembed`, `ort`, `ndarray`) are deliberately excluded: they cannot change the
//! vectors beyond float-level noise, so a bump must not force a re-embed.

use std::fmt;
use std::fmt::Write;

use sha2::{Digest, Sha256};

use super::spec::{ModelSpec, Pooling, Quantization, TruncatedDims};
use crate::prefixes::Prefixes;

/// Engine crate versions, baked at build time from `Cargo.lock` (see `build.rs`); inspection only.
const ENGINE_FINGERPRINT: &str = env!("PATTERNS_ENGINE_FINGERPRINT");

/// Tokenizer version: the one engine crate in `id()`, since it decides the token stream.
const TOKENIZER_FINGERPRINT: &str = env!("PATTERNS_TOKENIZER_FINGERPRINT");

/// Hex characters of the specification digest kept in the identity (32 bits).
const SPEC_DIGEST_HEX: usize = 8;

/// Hex characters of the compact model id (128 bits).
const MODEL_ID_HEX: usize = 32;

/// Which source authorised a truncated (MRL) width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MrlSource {
    /// The artifact's own `matryoshka_dimensions` declared the width.
    Metadata,
    /// The request itself was the claim; there was nothing to cross-check.
    Spec,
}

impl MrlSource {
    const fn key(self) -> &'static str {
        match self {
            Self::Metadata => "meta",
            Self::Spec => "spec",
        }
    }

    const fn suffix(self) -> &'static str {
        match self {
            Self::Metadata => "/meta",
            Self::Spec => "/spec",
        }
    }
}

/// Complete identity of a loaded model: everything that decides whether stored vectors are
/// comparable with a new run, plus `revision` for humans (not part of `id()`).
///
/// `id()` is a compact key derived from all of the above except `revision`; `Display` is the
/// verbose human form. Two runs are safe to reuse interchangeably exactly when their `id()` match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentity {
    repo: String,
    file: String,
    revision: String,
    content_hash: String,
    dim: usize,
    spec_digest: String,
    /// `tokenizers` version — the only engine crate in `id()`.
    tokenizer_digest: String,
    /// `fastembed`/`ort`/`ndarray` versions — inspection only, not in `id()`.
    engine_digest: String,
    mrl_source: Option<MrlSource>,
    id: String,
}

impl ModelIdentity {
    /// Identity of `spec`'s artifact, given the verified `content_hash` of its files and whether a
    /// truncated width came from the artifact metadata. `revision` is carried but excluded from `id`.
    pub(super) fn new(spec: &ModelSpec, content_hash: String, mrl_from_metadata: bool) -> Self {
        let dim = spec
            .truncate_to()
            .map_or_else(|| spec.native_dim(), TruncatedDims::get);
        let mrl_source = spec.truncate_to().map(|_| {
            if mrl_from_metadata {
                MrlSource::Metadata
            } else {
                MrlSource::Spec
            }
        });
        let mut identity = Self {
            repo: spec.repo().to_string(),
            file: spec.file().to_string(),
            revision: spec.revision().to_string(),
            content_hash,
            dim,
            spec_digest: spec_digest(spec),
            tokenizer_digest: TOKENIZER_FINGERPRINT.to_string(),
            engine_digest: ENGINE_FINGERPRINT.to_string(),
            mrl_source,
            id: String::new(),
        };
        identity.id = identity.compute_id();
        identity
    }

    /// Synthetic identity for [`super::Embedder::fake`]; not derived from any artifact.
    pub(super) fn fake(dim: usize, prefixes: &Prefixes) -> Self {
        let mut identity = Self {
            repo: "fake".to_string(),
            file: "fake".to_string(),
            revision: "0".repeat(40),
            content_hash: "fake".to_string(),
            dim,
            spec_digest: digest_parts("none", "none", "none", &prefixes.query, &prefixes.document),
            tokenizer_digest: TOKENIZER_FINGERPRINT.to_string(),
            engine_digest: ENGINE_FINGERPRINT.to_string(),
            mrl_source: None,
            id: String::new(),
        };
        identity.id = identity.compute_id();
        identity
    }

    /// Hub repository, e.g. `Snowflake/snowflake-arctic-embed-m-v1.5`.
    #[must_use]
    pub fn repo(&self) -> &str {
        &self.repo
    }

    /// Artifact path within the repo, e.g. `onnx/model_quantized.onnx`.
    #[must_use]
    pub fn file(&self) -> &str {
        &self.file
    }

    /// Pinned commit the bytes were fetched from; for humans only, not part of `id()`, so a revision
    /// bump over identical bytes does not force a re-embed.
    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// sha256 over the verified files (the artifact, `tokenizer.json`, `config.json`,
    /// `tokenizer_config.json`, `special_tokens_map.json`, `1_Pooling/config.json`), independent of
    /// the hub's hash scheme.
    #[must_use]
    pub fn content_hash(&self) -> &str {
        &self.content_hash
    }

    /// Effective (post-truncation) width stored.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Digest of the vector-shaping choices: output, pooling, quantization and prefixes.
    #[must_use]
    pub fn spec_digest(&self) -> &str {
        &self.spec_digest
    }

    /// Digest of the tokenizer version, part of `id()` (a bump can change the token stream).
    #[must_use]
    pub fn tokenizer_digest(&self) -> &str {
        &self.tokenizer_digest
    }

    /// Digest of the execution crate versions, for inspection; not part of `id()`.
    #[must_use]
    pub fn engine_digest(&self) -> &str {
        &self.engine_digest
    }

    /// Source of the MRL width, when the model is truncated.
    #[must_use]
    pub const fn mrl_source(&self) -> Option<MrlSource> {
        self.mrl_source
    }

    /// Compact key: equal exactly when the produced vectors are comparable. Use as a directory or
    /// column key; it is filesystem- and SQL-safe and of fixed length.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    fn compute_id(&self) -> String {
        let mrl = self.mrl_source.map_or("native", MrlSource::key);
        // repo/file and the revision are labels, not determinants; the artifact path is already
        // inside `content_hash`. The execution crates are excluded on purpose.
        let canonical = format!(
            "{}\n{}\n{}\n{}\n{mrl}",
            self.content_hash, self.dim, self.spec_digest, self.tokenizer_digest
        );
        let digest = Sha256::digest(canonical.as_bytes());
        hex(&digest).chars().take(MODEL_ID_HEX).collect()
    }
}

impl fmt::Display for ModelIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let suffix = self.mrl_source.map_or("", MrlSource::suffix);
        write!(
            f,
            "{}/{}@{}#d{}+c{}+e{}{suffix}",
            self.repo, self.file, self.revision, self.dim, self.spec_digest, self.engine_digest
        )
    }
}

/// Digest of the vector-shaping spec choices; unset output/pooling render as `none`.
fn spec_digest(spec: &ModelSpec) -> String {
    let pooling = match spec.pooling() {
        None => "none",
        Some(Pooling::Cls) => "cls",
        Some(Pooling::Mean) => "mean",
    };
    let quantization = match spec.quantization() {
        Quantization::None => "none",
        Quantization::Dynamic => "dynamic",
    };
    digest_parts(
        spec.output().unwrap_or("none"),
        pooling,
        quantization,
        &spec.prefixes().query,
        &spec.prefixes().document,
    )
}

/// First [`SPEC_DIGEST_HEX`] hex chars of the sha256 over `parts` (each `0xFF`-separated).
fn digest_parts(
    output: &str,
    pooling: &str,
    quantization: &str,
    query: &str,
    document: &str,
) -> String {
    let mut hasher = Sha256::new();
    for part in [output, pooling, quantization, query, document] {
        hasher.update(part.as_bytes());
        hasher.update([0xff]);
    }
    hex(&hasher.finalize())
        .chars()
        .take(SPEC_DIGEST_HEX)
        .collect()
}

/// Lowercase hex of `bytes`.
fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        write!(out, "{byte:02x}").expect("writing to a String is infallible");
    }
    out
}
