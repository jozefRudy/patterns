//! Model identity for [`crate::embed::Embedder`].
//!
//! Holds the single spec type plus patterns-owned mirrors of the fastembed enums, so consumers
//! never name fastembed types. No I/O: validation and (crate-internal) conversions only.

use std::num::NonZeroUsize;

use anyhow::{Result, anyhow, ensure};

/// Minimum MRL width accepted (mirrors `embed_api`'s `MIN_DIMS`).
pub(crate) const MIN_TRUNCATED_DIMS: usize = 32;

/// How a 3-D (token-level) graph output is reduced to one vector per text.
///
/// Irrelevant for a pre-pooled 2-D output: fastembed returns such a tensor unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    Cls,
    Mean,
}

/// Input to fastembed's batching rule: `Dynamic` means one call must not be split into batches
/// (per-batch activation ranges differ, so embeddings would not be comparable across batches).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Quantization {
    #[default]
    None,
    Static,
    Dynamic,
}

/// MRL width to store; constructor-validated so [`ModelSpec`] always holds a valid one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncatedDims(NonZeroUsize);

impl TruncatedDims {
    /// A stored width of `dims` dimensions.
    ///
    /// # Errors
    /// `dims < MIN_TRUNCATED_DIMS`, or `dims % 8 != 0` (sign bits pack 8 per byte).
    pub fn new(dims: usize) -> Result<Self> {
        let value = NonZeroUsize::new(dims).ok_or_else(|| anyhow!("truncated dims must be > 0"))?;
        ensure!(
            dims >= MIN_TRUNCATED_DIMS,
            "truncated dims {dims} below minimum {MIN_TRUNCATED_DIMS}"
        );
        ensure!(
            dims.is_multiple_of(8),
            "truncated dims {dims} must be a multiple of 8 (one sign bit per dim, 8 per byte)"
        );
        Ok(Self(value))
    }

    pub(crate) const fn get(self) -> usize {
        self.0.get()
    }
}

/// Pooling flags parsed from a repo's `1_Pooling/config.json`. `None` for the whole struct means
/// the file was absent in both the artifact and any declared metadata repo.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors the 1_Pooling/config.json flag set verbatim"
)]
pub(crate) struct PoolingMeta {
    pub(crate) cls: bool,
    pub(crate) mean: bool,
    pub(crate) max: bool,
    pub(crate) weightedmean: bool,
    pub(crate) mean_sqrt_len: bool,
    pub(crate) lasttoken: bool,
    /// Missing in some repos (e.g. bge) — treat `None` as `true`.
    pub(crate) include_prompt: Option<bool>,
}

/// Secondary repo consulted only for `1_Pooling/config.json` (converted ONNX exports drop it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetadataSource {
    pub(crate) repo: String,
    pub(crate) revision: String,
}

/// The only model interface: everything needed to fetch, verify and load one embedding artifact.
///
/// Fields are private: invalid states (an unpinned revision, a zero width, an unaligned MRL width)
/// are rejected by [`ModelSpec::new`] and the `with_*` builders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    /// Hub repository, e.g. `mixedbread-ai/mxbai-embed-large-v1`.
    pub(crate) repo: String,
    /// Full commit sha — never a branch or tag, so the fetched bytes cannot change.
    pub(crate) revision: String,
    /// Artifact path within the repo, e.g. `onnx/model_quantized.onnx`.
    pub(crate) file: String,
    /// External-initializer files (`*.onnx_data`) that `file` references, if any.
    pub(crate) additional: Vec<String>,
    /// Graph output to read. `None` requires the graph to have exactly one output.
    pub(crate) output: Option<&'static str>,
    /// How to reduce a 3-D output; `None` means the selected output is already pooled.
    pub(crate) pooling: Option<Pooling>,
    /// Decides fastembed's batching rule; `Dynamic` forbids splitting one call into batches.
    pub(crate) quantization: Quantization,
    /// Repo to read `1_Pooling/config.json` from when the artifact's own repo has none.
    pub(crate) pooling_metadata_from: Option<MetadataSource>,
    /// The model's own width, before any MRL truncation.
    pub(crate) native_dim: NonZeroUsize,
    /// MRL width to store instead of the native width; requesting one *is* the MRL claim.
    pub(crate) truncate_to: Option<TruncatedDims>,
}

impl ModelSpec {
    /// A spec for `repo`'s `file` at a pinned `revision`, whose native width is `dim`.
    ///
    /// # Errors
    /// `revision` is not a 40-hex commit sha (branches, tags and short shas are rejected so the
    /// fetched bytes are reproducible), or `dim == 0`.
    pub fn new(
        repo: impl Into<String>,
        revision: impl Into<String>,
        file: impl Into<String>,
        dim: usize,
    ) -> Result<Self> {
        let (repo, revision, file) = (repo.into(), revision.into(), file.into());
        ensure!(!repo.trim().is_empty(), "repo must not be empty");
        ensure!(!file.trim().is_empty(), "file must not be empty");
        ensure!(
            is_pinned_sha(&revision),
            "revision `{revision}` is not a 40-character lowercase hex commit sha; pin a commit \
             (branches, tags and short shas are not reproducible)"
        );
        let native_dim = NonZeroUsize::new(dim).ok_or_else(|| anyhow!("dim must be > 0"))?;
        Ok(Self {
            repo,
            revision,
            file,
            additional: Vec::new(),
            output: None,
            pooling: None,
            quantization: Quantization::default(),
            pooling_metadata_from: None,
            native_dim,
            truncate_to: None,
        })
    }

    /// External-initializer files (`*.onnx_data`) fetched alongside `file`.
    #[must_use]
    pub fn with_additional(mut self, files: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.additional = files.into_iter().map(Into::into).collect();
        self
    }

    /// Force a graph output by name (read off the export, e.g. `"sentence_embedding"`).
    ///
    /// Left unset, the graph must have exactly one output — fastembed's own precedence list is
    /// deliberately not used, since it ranks `last_hidden_state` above `sentence_embedding`.
    #[must_use]
    pub const fn with_output(mut self, name: &'static str) -> Self {
        self.output = Some(name);
        self
    }

    /// How *we* reduce a 3-D output; required in that case, ignored for a pooled 2-D output.
    #[must_use]
    pub const fn with_pooling(mut self, pooling: Pooling) -> Self {
        self.pooling = Some(pooling);
        self
    }

    #[must_use]
    pub const fn with_quantization(mut self, mode: Quantization) -> Self {
        self.quantization = mode;
        self
    }

    /// Verify pooling against this repo's `1_Pooling/config.json` when the artifact's own repo
    /// has none (typical for converted ONNX exports).
    #[must_use]
    pub fn with_pooling_metadata_from(
        mut self,
        repo: impl Into<String>,
        revision: impl Into<String>,
    ) -> Self {
        self.pooling_metadata_from = Some(MetadataSource {
            repo: repo.into(),
            revision: revision.into(),
        });
        self
    }

    /// Requesting a width *is* the MRL claim; the artifact's `matryoshka_dimensions` (when
    /// declared) constrains it at load.
    #[must_use]
    pub const fn with_truncated_dims(mut self, dims: TruncatedDims) -> Self {
        self.truncate_to = Some(dims);
        self
    }

    pub(crate) fn repo(&self) -> &str {
        &self.repo
    }

    pub(crate) fn revision(&self) -> &str {
        &self.revision
    }

    pub(crate) fn file(&self) -> &str {
        &self.file
    }

    pub(crate) const fn additional(&self) -> &[String] {
        self.additional.as_slice()
    }

    pub(crate) const fn output(&self) -> Option<&'static str> {
        self.output
    }

    pub(crate) const fn pooling(&self) -> Option<Pooling> {
        self.pooling
    }

    pub(crate) const fn quantization(&self) -> Quantization {
        self.quantization
    }

    pub(crate) const fn pooling_metadata_from(&self) -> Option<&MetadataSource> {
        self.pooling_metadata_from.as_ref()
    }

    pub(crate) const fn native_dim(&self) -> usize {
        self.native_dim.get()
    }

    pub(crate) const fn truncate_to(&self) -> Option<TruncatedDims> {
        self.truncate_to
    }
}

/// True only for a full, lowercase-hex commit sha: anything else (a branch, tag, or abbreviated
/// sha) lets the fetched bytes change between loads under the same identity.
fn is_pinned_sha(revision: &str) -> bool {
    revision.len() == 40
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

impl From<Pooling> for fastembed::Pooling {
    fn from(pooling: Pooling) -> Self {
        match pooling {
            Pooling::Cls => Self::Cls,
            Pooling::Mean => Self::Mean,
        }
    }
}

impl From<Quantization> for fastembed::QuantizationMode {
    fn from(quantization: Quantization) -> Self {
        match quantization {
            Quantization::None => Self::None,
            Quantization::Static => Self::Static,
            Quantization::Dynamic => Self::Dynamic,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

    fn spec() -> ModelSpec {
        ModelSpec::new("org/repo", SHA, "onnx/model.onnx", 768).expect("valid spec")
    }

    #[test]
    fn pooling_mirror_maps_every_variant() {
        assert_eq!(
            fastembed::Pooling::from(Pooling::Cls),
            fastembed::Pooling::Cls
        );
        assert_eq!(
            fastembed::Pooling::from(Pooling::Mean),
            fastembed::Pooling::Mean
        );
    }

    #[test]
    fn quantization_mirror_maps_every_variant() {
        assert_eq!(
            fastembed::QuantizationMode::from(Quantization::None),
            fastembed::QuantizationMode::None
        );
        assert_eq!(
            fastembed::QuantizationMode::from(Quantization::Static),
            fastembed::QuantizationMode::Static
        );
        assert_eq!(
            fastembed::QuantizationMode::from(Quantization::Dynamic),
            fastembed::QuantizationMode::Dynamic
        );
        assert_eq!(Quantization::default(), Quantization::None);
    }

    #[test]
    fn truncated_dims_accepts_mrl_widths() {
        assert_eq!(TruncatedDims::new(32).expect("32 ok").get(), 32);
        assert_eq!(TruncatedDims::new(256).expect("256 ok").get(), 256);
        assert_eq!(TruncatedDims::new(768).expect("768 ok").get(), 768);
    }

    #[test]
    fn truncated_dims_rejects_zero_short_and_unaligned() {
        assert!(TruncatedDims::new(0).is_err(), "zero");
        assert!(TruncatedDims::new(24).is_err(), "below the minimum");
        assert!(TruncatedDims::new(300).is_err(), "not byte-aligned");
    }

    #[test]
    fn model_spec_rejects_unpinned_or_malformed_revisions() {
        for revision in [
            "main",
            "v1.5",
            "0123456789abcdef0123456789abcdef0123456", // 39 chars
            "0123456789ABCDEF0123456789abcdef01234567", // uppercase
            "0123456789abcdef0123456789abcdef0123456z", // non-hex
        ] {
            assert!(
                ModelSpec::new("org/repo", revision, "onnx/model.onnx", 768).is_err(),
                "revision `{revision}` must be rejected"
            );
        }
    }

    #[test]
    fn model_spec_rejects_empty_repo_or_file_and_zero_dim() {
        for (repo, file, dim) in [
            ("", "onnx/model.onnx", 768),
            ("  ", "onnx/model.onnx", 768),
            ("org/repo", "", 768),
            ("org/repo", "onnx/model.onnx", 0),
        ] {
            ModelSpec::new(repo, SHA, file, dim)
                .expect_err("empty repo/file and dim 0 must be rejected");
        }
    }

    #[test]
    fn model_spec_defaults_and_builders() {
        let base = spec();
        assert_eq!(base.repo(), "org/repo");
        assert_eq!(base.revision(), SHA);
        assert_eq!(base.file(), "onnx/model.onnx");
        assert_eq!(base.native_dim(), 768);
        assert_eq!(base.additional(), <&[String]>::default());
        assert_eq!(base.output(), None);
        assert_eq!(base.pooling(), None);
        assert_eq!(base.quantization(), Quantization::None);
        assert!(base.pooling_metadata_from().is_none());
        assert!(base.truncate_to().is_none());

        let built = spec()
            .with_additional(["onnx/model.onnx_data"])
            .with_output("sentence_embedding")
            .with_pooling(Pooling::Cls)
            .with_quantization(Quantization::Dynamic)
            .with_pooling_metadata_from("upstream/repo", SHA)
            .with_truncated_dims(TruncatedDims::new(256).expect("256 ok"));
        assert_eq!(built.additional(), ["onnx/model.onnx_data"]);
        assert_eq!(built.output(), Some("sentence_embedding"));
        assert_eq!(built.pooling(), Some(Pooling::Cls));
        assert_eq!(built.quantization(), Quantization::Dynamic);
        assert_eq!(
            built.pooling_metadata_from().map(|m| m.repo.as_str()),
            Some("upstream/repo")
        );
        assert_eq!(
            built.truncate_to().map(TruncatedDims::get),
            Some(256),
            "truncated width is carried through"
        );
    }
}
