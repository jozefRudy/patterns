//! Embedding generation via fastembed (ONNX, in-process).
//!
//! Machinery: model loading, blocking-offload, batching, fake embedder for
//! tests, and query/document prefix handling. The **prefixes are model
//! configuration** — passed at load time (e.g. `"search_query: "` /
//! `"search_document: "` for nomic, `""`/`""` for BGE-M3) — after which
//! `embed_query` and the chunked document methods apply them automatically.
//!
//! A model is described by one [`ModelSpec`]: repo + pinned revision + file, fetched and verified
//! at load, never taken from a fastembed table or a moving branch. Identity is
//! `{repo}/{file}@{revision}#d{effective}[/meta|/spec]`; a failed fetch or verification is an
//! `Err` — never a silent model substitution, never a fallback model. Guard failure modes are
//! documented on [`Embedder::load`].
//!
//! fastembed types never appear here: callers use [`ModelSpec`], [`Pooling`], [`Quantization`] and
//! [`TruncatedDims`], and `patterns::fastembed` is no longer re-exported — `patterns::ort` still
//! is, since that re-export is what keeps a single ONNX Runtime linkage in the graph.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail, ensure};
use fastembed::{InitOptionsUserDefined, OutputKey, TextEmbedding, UserDefinedEmbeddingModel};
use ort::session::builder::GraphOptimizationLevel;

pub use crate::chunk::{ChunkOptions, EmbeddedChunk, TextChunk};
use crate::chunk::{SPECIAL_TOKEN_HEADROOM, TokenSpan, chunk_spans, fake_token_spans};
pub use crate::prefixes::Prefixes;

pub(crate) mod hub;
pub mod spec;

pub use spec::{ModelSpec, Pooling, Quantization, TruncatedDims};

/// Load-time configuration for [`Embedder`].
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Which model to load; see [`ModelSpec`].
    pub spec: ModelSpec,
    /// ONNX intra-op thread count (`None` = all cores). Set below core
    /// count to leave CPU for other tasks.
    pub intra_threads: Option<usize>,
    /// Query/document prefixes (see [`Prefixes`]).
    pub prefixes: Prefixes,
    /// Show model download progress bar on first fetch.
    pub show_download_progress: bool,
}

impl LoadOptions {
    /// Defaults: all cores, no prefixes, download progress shown.
    #[must_use]
    pub const fn new(spec: ModelSpec) -> Self {
        Self {
            spec,
            intra_threads: None,
            prefixes: Prefixes::none(),
            show_download_progress: true,
        }
    }

    /// Cap ONNX intra-op threads.
    #[must_use]
    pub const fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = Some(threads);
        self
    }

    /// Set the model's query/document prefixes.
    #[must_use]
    pub fn with_prefixes(mut self, prefixes: &Prefixes) -> Self {
        self.prefixes = prefixes.clone();
        self
    }

    /// Toggle download progress output.
    #[must_use]
    pub const fn with_show_download_progress(mut self, show: bool) -> Self {
        self.show_download_progress = show;
        self
    }
}

/// Embedder handle. `Embedder::load*` runs real ONNX inference;
/// `Embedder::fake` returns deterministic hash-based vectors for tests.
pub struct Embedder(Inner);

/// Internals stay private so the public API isn't pinned to fastembed/ort
/// types (e.g. the model handle can be swapped without a breaking change).
enum Inner {
    Fake {
        /// Effective width (post-truncation), so offline tests exercise 32/96-byte shapes.
        dim: usize,
        native_dim: usize,
        truncated: Option<TruncatedDims>,
        prefixes: Prefixes,
    },
    FastEmbed {
        model: Arc<Mutex<TextEmbedding>>,
        /// Effective width: `truncated.map_or(native_dim, TruncatedDims::get)`.
        dim: usize,
        /// The model's own width, before MRL truncation (probed at load).
        native_dim: usize,
        truncated: Option<TruncatedDims>,
        /// `{repo}/{file}@{revision}#d{effective}[/meta|/spec]`; part of every row's key.
        model_id: String,
        prefixes: Prefixes,
        /// Truncation-disabled copy for counting: `token_count` reads it
        /// immutably (`Tokenizer::encode` takes `&self`) — no lock, so counting
        /// never contends with inference on the model mutex.
        count_tokenizer: Arc<tokenizers::Tokenizer>,
    },
}

impl Embedder {
    /// Load the model described by `options.spec` into `cache_dir` (created if absent), CPU EP.
    ///
    /// Loads the artifact in memory (`UserDefinedEmbeddingModel`), so its bytes count against RSS
    /// (~S steady, ~2S transient during load).
    ///
    /// # Errors
    /// Every guard is always on and returns `Err` (each failure would otherwise be silent): a
    /// failed fetch or sha256 mismatch; session inputs outside
    /// `{input_ids, attention_mask, token_type_ids}` (fastembed feeds only those, so a graph
    /// wanting `position_ids`/`past_key_values` cannot run — fail here, not at the first embed);
    /// output selection failures (see [`validate_graph`]); a 3-D selected output without an
    /// explicitly declared pooling, or one that contradicts the artifact's
    /// `1_Pooling/config.json`; `native_dim != spec.dim` or a native width not a multiple of 8;
    /// a truncated width that exceeds the native width or is absent from the declared
    /// `matryoshka_dimensions`. Never falls back to another model.
    pub async fn load(options: LoadOptions, cache_dir: &Path) -> Result<Self> {
        let LoadOptions {
            spec,
            intra_threads,
            prefixes,
            show_download_progress,
        } = options;
        let cache_dir = cache_dir.to_path_buf();

        let loaded = tokio::task::spawn_blocking(move || {
            load_blocking(&spec, &cache_dir, intra_threads, show_download_progress)
        })
        .await
        .context("the model load task failed")??;

        Ok(Self(Inner::FastEmbed {
            model: Arc::new(Mutex::new(loaded.embedder)),
            dim: loaded
                .truncated
                .map_or(loaded.native_dim, TruncatedDims::get),
            native_dim: loaded.native_dim,
            truncated: loaded.truncated,
            model_id: loaded.model_id,
            prefixes,
            count_tokenizer: Arc::new(loaded.count_tokenizer),
        }))
    }

    /// Deterministic fake embedder (hash-based) for tests, no prefixes.
    #[must_use]
    pub fn fake(dim: usize) -> Self {
        Self::fake_with_prefixes(dim, &Prefixes::none())
    }

    /// Deterministic fake embedder with the given prefixes for tests.
    #[must_use]
    pub fn fake_with_prefixes(dim: usize, prefixes: &Prefixes) -> Self {
        Self(Inner::Fake {
            dim,
            native_dim: dim,
            truncated: None,
            prefixes: prefixes.clone(),
        })
    }

    /// Fake embedder whose vectors are truncated to `truncated`, for offline width tests.
    #[cfg(test)]
    #[must_use]
    pub const fn fake_truncated(native_dim: usize, truncated: TruncatedDims) -> Self {
        Self(Inner::Fake {
            dim: truncated.get(),
            native_dim,
            truncated: Some(truncated),
            prefixes: Prefixes::none(),
        })
    }

    /// Effective width written to the index: the native width, or the MRL width when truncated.
    #[must_use]
    pub const fn dim(&self) -> usize {
        match &self.0 {
            Inner::Fake { dim, .. } | Inner::FastEmbed { dim, .. } => *dim,
        }
    }

    /// The model's own width, before any MRL truncation.
    #[must_use]
    pub const fn native_dim(&self) -> usize {
        match &self.0 {
            Inner::Fake { native_dim, .. } | Inner::FastEmbed { native_dim, .. } => *native_dim,
        }
    }

    /// Identity of the loaded model, written into every embedding row:
    /// `{repo}/{file}@{revision}#d{effective}`, with `/meta` when the truncated width came from
    /// the artifact's `matryoshka_dimensions` and `/spec` when the request itself was the claim.
    #[must_use]
    pub fn model_id(&self) -> &str {
        match &self.0 {
            Inner::Fake { .. } => FAKE_MODEL_ID,
            Inner::FastEmbed { model_id, .. } => model_id,
        }
    }

    /// Embed a batch of raw texts (no prefix applied).
    ///
    /// # Panics
    /// Panics if the embedder mutex is poisoned (another task panicked
    /// while holding it) — invariant: single well-behaved owner.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let rows = match &self.0 {
            Inner::Fake { dim, .. } => {
                let () = std::future::ready(()).await;
                Ok(texts.iter().map(|t| embedding_for(t, *dim)).collect())
            }
            Inner::FastEmbed { model, .. } => {
                let model = Arc::clone(model);
                let texts: Vec<String> = texts.iter().map(|s| (*s).to_string()).collect();
                tokio::task::spawn_blocking(move || {
                    let mut model = model.lock().expect("embedder mutex poisoned");
                    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
                    model
                        .embed(&refs, None)
                        .map_err(|e| anyhow::anyhow!("embed failed: {e}"))
                })
                .await?
            }
        }?;

        // single exit point: MRL truncation happens after fastembed's L2 normalization and is not
        // renormalized (sign binarization ignores magnitude; matches `embed_api` semantics)
        Ok(match self.truncated() {
            None => rows,
            Some(dims) => rows
                .into_iter()
                .map(|mut row| {
                    row.truncate(dims.get());
                    row
                })
                .collect(),
        })
    }

    /// Embed one already-prefixed text.
    async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut batch = self.embed_batch(&[text]).await?;
        batch
            .pop()
            .ok_or_else(|| anyhow::anyhow!("expected one embedding, got none"))
    }

    /// Embed a query with the model's query prefix.
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_one(&self.prefixed_query(text)).await
    }

    /// Context window of the loaded model, in tokens.
    ///
    /// # Panics
    /// Panics if the embedder mutex is poisoned (invariant: single well-behaved
    /// owner), or if the tokenizer has no truncation configured — fastembed
    /// always sets truncation from the model config at load, so its absence
    /// means an unexpected fastembed change (fail loudly, don't guess a window).
    #[must_use]
    pub fn model_max_tokens(&self) -> usize {
        match &self.0 {
            Inner::Fake { .. } => FAKE_MODEL_MAX_TOKENS,
            Inner::FastEmbed { model, .. } => {
                let model = model.lock().expect("embedder mutex poisoned");
                model
                    .tokenizer
                    .get_truncation()
                    .map(|truncation| truncation.max_length)
                    .expect("fastembed configures tokenizer truncation at load")
            }
        }
    }

    /// Chunk options sized to this model's context window, minus the special
    /// tokens the tokenizer adds to every input ([CLS]/[SEP]). A static default
    /// would silently over-truncate small-context models and waste large ones;
    /// tweak the returned value per call site if needed.
    #[must_use]
    pub fn default_chunk_options(&self) -> ChunkOptions {
        let max = self
            .model_max_tokens()
            .saturating_sub(SPECIAL_TOKEN_HEADROOM)
            .max(1);
        ChunkOptions::new(max)
    }

    /// Full-text token count (truncation disabled).
    ///
    /// Lock-free: reads the dedicated counting tokenizer, never the model
    /// mutex — safe to call at high frequency alongside inference.
    ///
    /// Fake embedders approximate (whitespace words) — don't assert exact
    /// counts against them.
    pub fn token_count(&self, text: &str) -> Result<usize> {
        match &self.0 {
            Inner::Fake { .. } => Ok(fake_token_spans(text).len()),
            Inner::FastEmbed {
                count_tokenizer, ..
            } => {
                let encoding = count_tokenizer
                    .encode(text, false)
                    .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
                Ok(encoding.get_ids().len())
            }
        }
    }

    /// Embed documents, chunked, returning a flat row batch. Pass a
    /// single-element slice for one document (`doc_ix` is then 0). Accepts any
    /// `AsRef<str>` elements (`&[&str]`, `&[String]`, …) so callers don't have to
    /// own the text.
    ///
    /// Each [`EmbeddedChunk`] carries its `doc_ix` (index into `texts`) and
    /// `chunk_ix`, so callers can write rows or regroup without relying on
    /// position; rows are ordered by `(doc_ix, chunk_ix)`. Documents below
    /// `opts.min_tokens` contribute no chunks — the caller derives its skip set
    /// from the `doc_ix` values that appear.
    pub async fn embed_batch_document_chunks<S: AsRef<str> + Sync>(
        &self,
        texts: &[S],
        opts: &ChunkOptions,
    ) -> Result<Vec<EmbeddedChunk>> {
        // chunk per document, then ONE batched model call across all of them
        let mut pending: Vec<(usize, usize, TextChunk)> = Vec::new();
        for (doc_ix, text) in texts.iter().enumerate() {
            let text = text.as_ref();
            if self.token_count(text)? < opts.min_tokens {
                continue;
            }
            pending.extend(
                self.chunk(text, opts)?
                    .into_iter()
                    .enumerate()
                    .map(|(chunk_ix, chunk)| (doc_ix, chunk_ix, chunk)),
            );
        }
        if pending.is_empty() {
            return Ok(Vec::new());
        }

        let prefixed: Vec<String> = pending
            .iter()
            .map(|(_, _, chunk)| self.prefixed_document(&chunk.text))
            .collect();
        let refs: Vec<&str> = prefixed.iter().map(String::as_str).collect();
        let embeddings = self.embed_batch(&refs).await?;

        Ok(pending
            .into_iter()
            .zip(embeddings)
            .map(|((doc_ix, chunk_ix, chunk), embedding)| EmbeddedChunk {
                doc_ix,
                chunk_ix,
                chunk,
                embedding,
            })
            .collect())
    }

    /// Split `text` into token-aligned chunks covering all content. Empty
    /// input (or input with no tokens) yields no chunks; byte spans are
    /// relative to `text` and each chunk's `text` is that span, trimmed.
    fn chunk(&self, text: &str, opts: &ChunkOptions) -> Result<Vec<TextChunk>> {
        let spans = self.token_spans(text)?;
        Ok(chunk_spans(text, &spans, opts))
    }

    /// Byte span of every token in `text`, in order, truncation disabled.
    fn token_spans(&self, text: &str) -> Result<Vec<TokenSpan>> {
        match &self.0 {
            Inner::Fake { .. } => Ok(fake_token_spans(text)),
            Inner::FastEmbed { model, .. } => {
                // clone under the lock, then mutate the clone outside it —
                // `with_truncation(&mut self)` in place would disable truncation
                // for every later embed call
                let mut tokenizer = {
                    let model = model
                        .lock()
                        .map_err(|e| anyhow::anyhow!("embedder mutex poisoned: {e}"))?;
                    model.tokenizer.clone()
                };
                tokenizer
                    .with_truncation(None)
                    .map_err(|e| anyhow::anyhow!("disable truncation: {e}"))?;
                crate::chunk::token_spans(&tokenizer, text)
            }
        }
    }

    fn prefixed_query(&self, text: &str) -> String {
        format!("{}{text}", self.prefixes().query)
    }

    fn prefixed_document(&self, text: &str) -> String {
        format!("{}{text}", self.prefixes().document)
    }

    const fn truncated(&self) -> Option<TruncatedDims> {
        match &self.0 {
            Inner::Fake { truncated, .. } | Inner::FastEmbed { truncated, .. } => *truncated,
        }
    }

    const fn prefixes(&self) -> &Prefixes {
        match &self.0 {
            Inner::Fake { prefixes, .. } | Inner::FastEmbed { prefixes, .. } => prefixes,
        }
    }
}

/// Result of a successful blocking load.
struct Loaded {
    embedder: TextEmbedding,
    native_dim: usize,
    truncated: Option<TruncatedDims>,
    model_id: String,
    count_tokenizer: tokenizers::Tokenizer,
}

/// Fetch → verify → build → guard. Blocking: hf-hub's API and ONNX Runtime are sync.
fn load_blocking(
    spec: &ModelSpec,
    cache_dir: &Path,
    intra_threads: Option<usize>,
    show_download_progress: bool,
) -> Result<Loaded> {
    std::fs::create_dir_all(cache_dir).context("create the model cache dir")?;
    let client = hub::HubFiles::new(cache_dir.to_path_buf(), show_download_progress)?;
    let artifact = hub::fetch(&client, spec)?;

    let desc = graph_desc(&artifact.onnx, &artifact.external)?;
    let selected = validate_graph(
        &desc,
        spec,
        artifact.pooling_meta.as_ref(),
        artifact.matryoshka_dims.as_deref(),
    )?;

    let mut model = UserDefinedEmbeddingModel::new(artifact.onnx, artifact.tokenizer_files)
        .with_quantization(spec.quantization().into());
    for (file_name, buffer) in artifact.external {
        model = model.with_external_initializer(file_name, buffer);
    }
    if let Some(pooling) = selected.pooling {
        model = model.with_pooling(pooling.into());
    }
    // Always name the output explicitly, so fastembed's precedence list (which ranks
    // `last_hidden_state` above `sentence_embedding`) is never consulted. `OutputKey::ByName`
    // takes `&'static str`; a graph-derived name is leaked once per load — a few bytes.
    model.output_key = Some(OutputKey::ByName(match spec.output() {
        Some(name) => name,
        None => Box::leak(selected.name.into_boxed_str()),
    }));

    let mut options = InitOptionsUserDefined::new().with_max_length(artifact.max_length);
    if let Some(threads) = intra_threads {
        options = options.with_intra_threads(threads);
    }
    let mut embedder = TextEmbedding::try_new_from_user_defined(model, options)
        .context("build the embedding session")?;

    let probe = embedder.embed(["probe"], None).context("probe embed")?;
    let native_dim = probe
        .first()
        .ok_or_else(|| anyhow!("the model returned no probe embedding"))?
        .len();
    ensure!(
        native_dim == spec.native_dim(),
        "native width {native_dim} does not match the declared dim {}",
        spec.native_dim()
    );
    ensure!(
        native_dim % 8 == 0,
        "native width {native_dim} is not a multiple of 8 (8 sign bits per byte)"
    );
    if let Some(truncated) = spec.truncate_to() {
        ensure!(
            truncated.get() <= native_dim,
            "truncated width {} exceeds the native width {native_dim}",
            truncated.get()
        );
    }

    // counting copy: truncation disabled once, then frozen behind an Arc — `encode()` is `&self`,
    // so counts stay full-length and lock-free forever
    let mut count_tokenizer = embedder.tokenizer.clone();
    count_tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow!("disable truncation: {e}"))?;

    let model_id = model_id(spec, selected.mrl_from_metadata);
    Ok(Loaded {
        embedder,
        native_dim,
        truncated: spec.truncate_to(),
        model_id,
        count_tokenizer,
    })
}

/// Read a graph's inputs and outputs (name + rank) without keeping a session alive.
///
/// A throwaway session is the only way to see this: fastembed owns the real one and does not
/// expose it. Graph optimisation is disabled here, so this is cheap relative to the real load.
fn graph_desc(onnx: &[u8], external: &[(String, Vec<u8>)]) -> Result<GraphDesc> {
    let mut builder = ort::session::Session::builder()
        .map_err(|err| anyhow!("create an ONNX Runtime session builder: {err}"))?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|err| anyhow!("disable graph optimisation for the inspection session: {err}"))?;
    for (file_name, buffer) in external {
        builder = builder
            .with_external_initializer_file_in_memory(file_name.clone(), buffer.clone().into())
            .map_err(|err| anyhow!("attach an external initializer for inspection: {err}"))?;
    }
    let session = builder
        .commit_from_memory(onnx)
        .map_err(|err| anyhow!("load the graph for inspection: {err}"))?;

    let inputs = session
        .inputs()
        .iter()
        .map(|input| input.name().to_string())
        .collect();
    let outputs = session
        .outputs()
        .iter()
        .map(|output| (output.name().to_string(), tensor_rank(output.dtype())))
        .collect();
    Ok(GraphDesc { inputs, outputs })
}

/// Tensor rank of a session outlet; non-tensors (sequences/maps) report rank 0 and are rejected by
/// the output-selection guard.
fn tensor_rank(value_type: &ort::value::ValueType) -> usize {
    match value_type {
        ort::value::ValueType::Tensor { shape, .. } => shape.len(),
        _ => 0,
    }
}

/// Identity written into every embedding row:
/// `{repo}/{file}@{revision}#d{effective}`, plus `/meta` or `/spec` when truncated.
fn model_id(spec: &ModelSpec, mrl_from_metadata: bool) -> String {
    let effective = spec
        .truncate_to()
        .map_or_else(|| spec.native_dim(), TruncatedDims::get);
    let base = format!(
        "{}/{}@{}#d{effective}",
        spec.repo(),
        spec.file(),
        spec.revision()
    );
    match (spec.truncate_to().is_some(), mrl_from_metadata) {
        (false, _) => base,
        (true, true) => format!("{base}/meta"),
        (true, false) => format!("{base}/spec"),
    }
}

/// Plain description of a loaded graph, so the load-time guards are testable without ONNX.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraphDesc {
    pub(crate) inputs: Vec<String>,
    /// Output name → tensor rank (2 = already pooled, 3 = token-level).
    pub(crate) outputs: Vec<(String, usize)>,
}

/// What the guards decided: which output to read, how to reduce it, and how the truncated width
/// was authorised.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SelectedOutput {
    pub(crate) name: String,
    /// `Some` when the selected output is 3-D and we must pool it.
    pub(crate) pooling: Option<Pooling>,
    /// `true` when the truncated width was authorised by the artifact's `matryoshka_dimensions`
    /// (identity suffix `/meta`), `false` when the request itself was the claim (`/spec`).
    pub(crate) mrl_from_metadata: bool,
}

/// Inputs fastembed feeds; a graph needing anything else cannot run in-process.
const FEEDABLE_INPUTS: [&str; 3] = ["input_ids", "attention_mask", "token_type_ids"];

/// Validate a loaded graph against the spec: input set, output selection, pooling, truncation.
///
/// # Errors
/// Inputs outside [`FEEDABLE_INPUTS`]; `spec.output` unset on a multi-output graph; a named output
/// that does not exist; a 3-D output without an explicitly declared pooling, or one that
/// contradicts `pooling_meta`; a truncated width absent from the declared MRL widths.
pub(crate) fn validate_graph(
    desc: &GraphDesc,
    spec: &ModelSpec,
    pooling_meta: Option<&spec::PoolingMeta>,
    matryoshka_dims: Option<&[usize]>,
) -> Result<SelectedOutput> {
    let unsupported: Vec<&str> = desc
        .inputs
        .iter()
        .map(String::as_str)
        .filter(|input| !FEEDABLE_INPUTS.contains(input))
        .collect();
    ensure!(
        unsupported.is_empty(),
        "graph needs inputs fastembed cannot feed: {}",
        unsupported.join(", ")
    );

    let (name, rank) = if let Some(declared) = spec.output() {
        let found = desc
            .outputs
            .iter()
            .find(|(name, _)| name == declared)
            .ok_or_else(|| {
                anyhow!(
                    "declared output `{declared}` is not in the graph (has: {})",
                    output_names(desc)
                )
            })?;
        (found.0.clone(), found.1)
    } else {
        let [(name, rank)] = desc.outputs.as_slice() else {
            bail!(
                "graph has {} outputs ({}); declare one with `with_output(name)`",
                desc.outputs.len(),
                output_names(desc)
            );
        };
        (name.clone(), *rank)
    };

    let pooling = if rank == 3 {
        let declared = spec.pooling().ok_or_else(|| {
            anyhow!("output `{name}` is 3-D; declare `with_pooling(Pooling::Cls|Mean)`")
        })?;
        if let Some(meta) = pooling_meta {
            check_pooling_meta(declared, meta, &name)?;
        }
        Some(declared)
    } else {
        None
    };

    let mrl_from_metadata = match (spec.truncate_to(), matryoshka_dims) {
        (Some(dims), Some(declared)) => {
            ensure!(
                declared.contains(&dims.get()),
                "truncated width {} is not among the declared matryoshka_dimensions {declared:?}",
                dims.get()
            );
            true
        }
        // No declaration: the request itself is the MRL claim (identity suffix `/spec`).
        _ => false,
    };

    Ok(SelectedOutput {
        name,
        pooling,
        mrl_from_metadata,
    })
}

fn output_names(desc: &GraphDesc) -> String {
    desc.outputs
        .iter()
        .map(|(name, rank)| format!("{name}:{rank}D"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Cross-check a declared pooling against the artifact's own `1_Pooling/config.json`.
fn check_pooling_meta(declared: Pooling, meta: &spec::PoolingMeta, output: &str) -> Result<()> {
    ensure!(
        !meta.lasttoken,
        "1_Pooling declares last-token pooling, unsupported in-process (output `{output}`)"
    );
    ensure!(
        !meta.max && !meta.weightedmean && !meta.mean_sqrt_len,
        "1_Pooling declares a pooling mode fastembed cannot apply (output `{output}`)"
    );
    ensure!(
        meta.include_prompt != Some(false),
        "1_Pooling sets include_prompt: false; fastembed's mean would average the instruction in"
    );
    let agrees = match declared {
        Pooling::Cls => meta.cls,
        Pooling::Mean => meta.mean,
    };
    ensure!(
        agrees,
        "declared {declared:?} contradicts 1_Pooling (cls: {}, mean: {})",
        meta.cls,
        meta.mean
    );
    Ok(())
}

/// Token window reported by the `Fake` embedder (no model, no tokenizer).
const FAKE_MODEL_MAX_TOKENS: usize = 512;

/// Identity reported by the `Fake` embedder (tests only).
const FAKE_MODEL_ID: &str = "fake";

/// Number of bytes needed to pack `dims` sign bits (8 bits per byte).
#[must_use]
pub const fn packed_len(dims: usize) -> usize {
    dims.div_ceil(8)
}

/// Binary quantization: one sign bit per dimension, packed 8 bits per byte.
///
/// Model outputs are f32 (also for int8-weight models — the graph dequantizes),
/// L2-normalized, so the sign bit is a stable binarization criterion.
///
/// Bit for dimension `i` lives in byte `i / 8` at position `i % 8` (LSB first);
/// a bit is set when `values[i] > 0.0` (zeros and negatives → 0). The packing
/// order is part of the storage contract — index build and query must use it
/// identically (hamming distance is invariant to a consistent permutation).
#[must_use]
pub fn binarize(values: &[f32]) -> Vec<u8> {
    let mut out = vec![0_u8; packed_len(values.len())];
    for (ix, value) in values.iter().enumerate() {
        if *value > 0.0 {
            let byte = ix / 8;
            let bit = u8::try_from(ix % 8).unwrap_or_default();
            if let Some(slot) = out.get_mut(byte) {
                *slot |= 1 << bit;
            }
        }
    }
    out
}

/// Scalar quantization to one byte per dimension (8 bits/dim).
///
/// Fastembed returns L2-normalized vectors, so components sit well inside
/// `[-1, 1]` (mxbai-embed-large-v1 int8 measured: |x| ≤ 0.25) and the fixed
/// mapping is safe: `q = round((clamp(x, -1, 1) + 1) * 127.5)`, -1 → 0,
/// 0 → 128, 1 → 255. Range is used loosely (components cluster mid-byte) —
/// score quantized vectors directly (integer dot product), don't convert back.
#[must_use]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "value is clamped to [-1,1] then scaled to [0,255] — the cast is exact"
)]
pub fn quantize_u8(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|value| {
            let scaled = ((value.clamp(-1.0, 1.0) + 1.0) * 127.5).round();
            scaled as u8
        })
        .collect()
}

/// FNV-1a hash (fake-vector seeding).
///
/// # Panics
/// Never in practice — `i` stays below `bytes.len()` by loop condition.
#[expect(clippy::as_conversions, reason = "u8 to u64 widening cast is lossless")]
#[expect(
    clippy::indexing_slicing,
    reason = "i < bytes.len() enforced by while condition"
)]
const fn fnv1a(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x0100_0193);
        i += 1;
    }
    hash
}

const fn splitmix64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Deterministic fake vector: hash-seeded splitmix64 stream in [-1, 1).
#[expect(
    clippy::as_conversions,
    reason = "fake test vectors — precision loss is irrelevant"
)]
#[expect(
    clippy::cast_precision_loss,
    reason = "fake test vectors — precision loss is irrelevant"
)]
fn embedding_for(text: &str, dim: usize) -> Vec<f32> {
    let mut seed = fnv1a(text);
    (0..dim)
        .map(|_| {
            let v = splitmix64(&mut seed);
            (v as f32 / u64::MAX as f32).mul_add(2.0, -1.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::indexing_slicing,
        clippy::string_slice,
        reason = "test assertions on vectors/offsets with known length"
    )]

    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

    fn spec() -> ModelSpec {
        ModelSpec::new("org/repo", SHA, "onnx/model.onnx", 768).expect("valid spec")
    }

    fn desc(inputs: &[&str], outputs: &[(&str, usize)]) -> GraphDesc {
        GraphDesc {
            inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
            outputs: outputs
                .iter()
                .map(|(name, rank)| ((*name).to_string(), *rank))
                .collect(),
        }
    }

    fn meta_cls() -> spec::PoolingMeta {
        spec::PoolingMeta {
            cls: true,
            include_prompt: Some(true),
            ..spec::PoolingMeta::default()
        }
    }

    // ---- validate_graph ----------------------------------------------------------------

    #[test]
    fn validate_graph_rejects_inputs_fastembed_cannot_feed() {
        let desc = desc(
            &["input_ids", "attention_mask", "position_ids"],
            &[("last_hidden_state", 3)],
        );
        let error = validate_graph(&desc, &spec().with_pooling(Pooling::Cls), None, None)
            .expect_err("position_ids cannot be fed");
        assert!(error.to_string().contains("position_ids"), "{error}");
    }

    #[test]
    fn validate_graph_requires_an_explicit_output_on_a_multi_output_graph() {
        let desc = desc(
            &["input_ids", "attention_mask"],
            &[("last_hidden_state", 3), ("sentence_embedding", 2)],
        );
        let error = validate_graph(&desc, &spec(), None, None)
            .expect_err("two outputs must be disambiguated");
        assert!(error.to_string().contains("with_output"), "{error}");
    }

    #[test]
    fn validate_graph_rejects_an_unknown_output_name() {
        let desc = desc(&["input_ids"], &[("sentence_embedding", 2)]);
        let error = validate_graph(&desc, &spec().with_output("text_embeds"), None, None)
            .expect_err("declared name must exist");
        assert!(error.to_string().contains("text_embeds"), "{error}");
    }

    #[test]
    fn validate_graph_requires_pooling_for_a_token_level_output() {
        let desc = desc(&["input_ids"], &[("last_hidden_state", 3)]);
        let error = validate_graph(&desc, &spec(), None, None)
            .expect_err("3-D output needs a declared pooling");
        assert!(error.to_string().contains("with_pooling"), "{error}");
    }

    #[test]
    fn validate_graph_accepts_a_single_token_level_output_with_pooling() {
        let desc = desc(
            &["input_ids", "attention_mask"],
            &[("last_hidden_state", 3)],
        );
        let selected = validate_graph(&desc, &spec().with_pooling(Pooling::Cls), None, None)
            .expect("3-D + declared pooling is fine");
        assert_eq!(selected.name, "last_hidden_state");
        assert_eq!(selected.pooling, Some(Pooling::Cls));
        assert!(!selected.mrl_from_metadata);
    }

    #[test]
    fn validate_graph_accepts_a_pre_pooled_output_without_pooling() {
        let desc = desc(
            &["input_ids", "attention_mask"],
            &[("last_hidden_state", 3), ("sentence_embedding", 2)],
        );
        let selected = validate_graph(&desc, &spec().with_output("sentence_embedding"), None, None)
            .expect("2-D pooled output needs no pooling");
        assert_eq!(selected.pooling, None);
    }

    #[test]
    fn validate_graph_allows_lasttoken_metadata_for_a_pre_pooled_output() {
        // The graph already pooled; `lasttoken` describes how the export did it, so it is not an
        // obstacle when we are not the ones pooling.
        let desc = desc(&["input_ids"], &[("sentence_embedding", 2)]);
        let meta = spec::PoolingMeta {
            lasttoken: true,
            ..spec::PoolingMeta::default()
        };
        validate_graph(&desc, &spec(), Some(&meta), None)
            .expect("pre-pooled output is unaffected by pooling metadata");
    }

    #[test]
    fn validate_graph_cross_checks_declared_pooling_against_metadata() {
        let desc = desc(&["input_ids"], &[("last_hidden_state", 3)]);
        validate_graph(
            &desc,
            &spec().with_pooling(Pooling::Cls),
            Some(&meta_cls()),
            None,
        )
        .expect("declared Cls matches 1_Pooling");

        let error = validate_graph(
            &desc,
            &spec().with_pooling(Pooling::Mean),
            Some(&meta_cls()),
            None,
        )
        .expect_err("Mean contradicts 1_Pooling");
        assert!(error.to_string().contains("contradicts"), "{error}");
    }

    #[test]
    fn validate_graph_rejects_unsupported_or_prompt_masking_metadata() {
        let desc = desc(&["input_ids"], &[("last_hidden_state", 3)]);
        let pooling = spec().with_pooling(Pooling::Cls);

        for meta in [
            spec::PoolingMeta {
                cls: true,
                lasttoken: true,
                ..spec::PoolingMeta::default()
            },
            spec::PoolingMeta {
                cls: true,
                max: true,
                ..spec::PoolingMeta::default()
            },
            spec::PoolingMeta {
                cls: true,
                include_prompt: Some(false),
                ..spec::PoolingMeta::default()
            },
        ] {
            assert!(
                validate_graph(&desc, &pooling, Some(&meta), None).is_err(),
                "unsupported metadata must fail the load: {meta:?}"
            );
        }
    }

    #[test]
    fn validate_graph_guards_truncation_against_declared_mrl_widths() {
        let desc = desc(&["input_ids"], &[("last_hidden_state", 3)]);
        let pooling = spec().with_pooling(Pooling::Cls);
        let truncated = pooling.with_truncated_dims(TruncatedDims::new(256).expect("256 ok"));

        let selected = validate_graph(&desc, &truncated, None, Some(&[256])).expect("declared MRL");
        assert!(
            selected.mrl_from_metadata,
            "width came from matryoshka_dimensions"
        );

        let selected = validate_graph(&desc, &truncated, None, None).expect("no declaration");
        assert!(
            !selected.mrl_from_metadata,
            "the request itself is the claim"
        );

        assert!(
            validate_graph(&desc, &truncated, None, Some(&[512])).is_err(),
            "256 is not a declared MRL width"
        );
    }

    // ---- identity ---------------------------------------------------------------------

    #[test]
    fn model_id_carries_revision_width_and_mrl_source() {
        let base = spec();
        assert_eq!(
            model_id(&base, false),
            format!("org/repo/onnx/model.onnx@{SHA}#d768")
        );

        let truncated = base.with_truncated_dims(TruncatedDims::new(256).expect("256 ok"));
        assert_eq!(
            model_id(&truncated, true),
            format!("org/repo/onnx/model.onnx@{SHA}#d256/meta")
        );
        assert_eq!(
            model_id(&truncated, false),
            format!("org/repo/onnx/model.onnx@{SHA}#d256/spec")
        );
    }

    #[test]
    fn fake_embedder_reports_effective_and_native_widths() {
        let plain = Embedder::fake(64);
        assert_eq!(plain.dim(), 64);
        assert_eq!(plain.native_dim(), 64);
        assert_eq!(plain.model_id(), FAKE_MODEL_ID);

        let truncated = Embedder::fake_truncated(64, TruncatedDims::new(32).expect("32 ok"));
        assert_eq!(truncated.dim(), 32, "effective width");
        assert_eq!(truncated.native_dim(), 64, "native width is unchanged");
    }

    // ---- truncation -------------------------------------------------------------------

    /// Bitwise-tolerant float comparison (`float_cmp` is denied).
    fn close(left: &[f32], right: &[f32]) -> bool {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(a, b)| (a - b).abs() <= f32::EPSILON)
    }

    #[tokio::test]
    async fn truncation_slices_after_normalization_without_renormalizing() {
        let full = Embedder::fake(64);
        let truncated = Embedder::fake_truncated(64, TruncatedDims::new(32).expect("32 ok"));

        let full_vector = full.embed_query("some text").await.expect("embed");
        let cut_vector = truncated.embed_query("some text").await.expect("embed");

        assert_eq!(cut_vector.len(), 32);
        assert!(
            close(&cut_vector, &full_vector[..32]),
            "truncation is a plain prefix of the pooled vector"
        );
    }

    // ---- real model (#[ignore]: downloads + verifies at a pinned revision) -------------

    #[tokio::test]
    #[ignore = "downloads the artifact from the hub"]
    async fn arctic_v1_5_loads_and_truncates_to_its_declared_mrl_width() {
        const REPO: &str = "Snowflake/snowflake-arctic-embed-m-v1.5";
        const REV: &str = "e58a8f756156a1293d763f17e3aae643474e9b8a";

        let dir = std::env::temp_dir().join("patterns_embed_hub_tests");
        let base = ModelSpec::new(REPO, REV, "onnx/model_quantized.onnx", 768)
            .expect("valid spec")
            .with_output("sentence_embedding")
            .with_quantization(Quantization::Dynamic);

        let full = Embedder::load(LoadOptions::new(base.clone()), &dir)
            .await
            .expect("loads and passes every guard");
        assert_eq!(full.native_dim(), 768);
        assert_eq!(full.dim(), 768);
        assert_eq!(
            full.model_id(),
            format!("{REPO}/onnx/model_quantized.onnx@{REV}#d768")
        );

        let truncated = Embedder::load(
            LoadOptions::new(base.with_truncated_dims(TruncatedDims::new(256).expect("256 ok"))),
            &dir,
        )
        .await
        .expect("256 is declared in matryoshka_dimensions");
        assert_eq!(truncated.dim(), 256);
        assert!(
            truncated.model_id().ends_with("#d256/meta"),
            "width authorised by the artifact's own matryoshka_dimensions"
        );

        let full_vector = full
            .embed_query("pricing a b2b newsletter")
            .await
            .expect("embed");
        let cut_vector = truncated
            .embed_query("pricing a b2b newsletter")
            .await
            .expect("embed");
        assert_eq!(cut_vector.len(), 256);
        assert!(
            close(&cut_vector, &full_vector[..256]),
            "truncation is a plain prefix"
        );
    }

    /// Chunking via the fake tokenizer (whitespace spans).
    fn fake_chunks(text: &str, opts: &ChunkOptions) -> Vec<TextChunk> {
        chunk_spans(text, &fake_token_spans(text), opts)
    }

    static EMBEDDER: tokio::sync::OnceCell<Embedder> = tokio::sync::OnceCell::const_new();

    async fn test_embedder() -> &'static Embedder {
        EMBEDDER
            .get_or_init(|| async {
                let dir = std::env::temp_dir().join("patterns_embed_tests");
                std::fs::create_dir_all(&dir).expect("test cache dir");
                let spec = ModelSpec::new(
                    "Xenova/all-MiniLM-L6-v2",
                    "751bff37182d3f1213fa05d7196b954e230abad9",
                    "onnx/model_quantized.onnx",
                    384,
                )
                .expect("valid spec")
                .with_pooling(Pooling::Mean)
                .with_quantization(Quantization::Dynamic);
                Embedder::load(LoadOptions::new(spec), &dir)
                    .await
                    .expect("model loads")
            })
            .await
    }

    #[test]
    fn default_chunk_options_fit_the_model_window() {
        let e = Embedder::fake(8);
        let opts = e.default_chunk_options();
        assert_eq!(
            opts.max_tokens,
            e.model_max_tokens() - SPECIAL_TOKEN_HEADROOM
        );
        assert_eq!(opts.overlap_tokens, 64);
        assert_eq!(opts.min_tokens, 5);
    }

    #[test]
    fn fake_token_count_is_approximate_and_monotonic() {
        let e = Embedder::fake(8);
        assert_eq!(e.token_count("").expect("count"), 0);
        assert_eq!(e.token_count("   ").expect("count"), 0);
        assert_eq!(e.token_count("one two three").expect("count"), 3);
        assert!(e.token_count("a b c d e").expect("count") > e.token_count("a b").expect("count"));
        assert_eq!(e.model_max_tokens(), 512);
    }

    #[test]
    fn chunk_short_text_is_single_span_equal_to_input() {
        let opts = ChunkOptions::new(500);
        let chunks = fake_chunks("  hello world  ", &opts);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
        assert_eq!(chunks[0].tokens, 2);
        assert_eq!(chunks[0].byte_start, 2);
        assert_eq!(chunks[0].byte_end, 13);
    }

    #[test]
    fn chunk_splits_within_max_tokens_and_covers_all_content() {
        let text = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9";
        let opts = ChunkOptions {
            max_tokens: 3,
            overlap_tokens: 1,
            min_tokens: 1,
            max_chunks: None,
        };
        let chunks = fake_chunks(text, &opts);
        assert!(chunks.len() > 1, "long text must split");
        for c in &chunks {
            assert!(c.tokens <= opts.max_tokens, "chunk exceeds max_tokens");
            assert_eq!(&text[c.byte_start..c.byte_end], c.text);
        }
        for word in fake_token_spans(text) {
            let covered = chunks
                .iter()
                .any(|c| c.byte_start <= word.start && word.end <= c.byte_end);
            assert!(covered, "word at {}..{} not covered", word.start, word.end);
        }
        assert_eq!(chunks.last().expect("chunks").byte_end, text.len());
    }

    #[test]
    fn chunk_prefers_paragraph_break() {
        let text = "aa bb\n\ncc dd";
        let opts = ChunkOptions {
            max_tokens: 3,
            overlap_tokens: 0,
            min_tokens: 1,
            max_chunks: None,
        };
        let chunks = fake_chunks(text, &opts);
        assert_eq!(
            chunks[0].text, "aa bb",
            "breaks at the paragraph, not mid-way"
        );
        assert_eq!(chunks[1].text, "cc dd");
    }

    #[test]
    fn chunk_makes_progress_with_large_overlap() {
        let text = "a b c d e";
        let opts = ChunkOptions {
            max_tokens: 1,
            overlap_tokens: 64,
            min_tokens: 1,
            max_chunks: None,
        };
        let chunks = fake_chunks(text, &opts);
        assert_eq!(chunks.len(), 5, "one chunk per token, no stalls");
        assert!(chunks.windows(2).all(|w| w[0].byte_start < w[1].byte_start));
    }

    #[test]
    fn max_chunks_caps_emitted_chunks() {
        let text = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9";
        let base = ChunkOptions {
            max_tokens: 3,
            overlap_tokens: 0,
            min_tokens: 1,
            max_chunks: None,
        };
        let uncapped = fake_chunks(text, &base);
        assert!(uncapped.len() > 1, "long text must split when uncapped");

        let capped = fake_chunks(
            text,
            &ChunkOptions {
                max_chunks: std::num::NonZeroUsize::new(1),
                ..base
            },
        );
        assert_eq!(capped.len(), 1, "cap keeps only the first window");
        assert_eq!(capped[0].text, uncapped[0].text);
    }

    #[tokio::test]
    async fn batch_chunks_are_positional_and_gate_short_texts() {
        let e = Embedder::fake(8);
        let opts = ChunkOptions::new(500); // min_tokens = 5
        let texts = vec![
            "one two".to_string(),               // 0: below min -> skipped
            "a b c d e f".to_string(),           // 1: one chunk
            "x ".repeat(600).trim().to_string(), // 2: many chunks
            "hi".to_string(),                    // 3: trailing skip (leaves no trace)
        ];
        let out = e
            .embed_batch_document_chunks(&texts, &opts)
            .await
            .expect("batch");
        // doc 0 is below min_tokens -> contributes nothing; docs 1 and 2 do
        assert!(!out.is_empty());
        assert!(
            out.iter().all(|c| c.doc_ix != 0),
            "short doc must be absent"
        );
        let doc1: Vec<_> = out.iter().filter(|c| c.doc_ix == 1).collect();
        let doc2: Vec<_> = out.iter().filter(|c| c.doc_ix == 2).collect();
        assert_eq!(doc1.len(), 1, "short-ish doc yields one chunk");
        assert!(doc2.len() > 1, "long doc yields multiple chunks");
        assert!(doc1[0].chunk_ix == 0 && doc2.iter().enumerate().all(|(i, c)| c.chunk_ix == i));
        // the caller's skip-set derivation: universe (texts.len) minus seen
        // (doc_ix in rows) must yield exactly the gated documents — including
        // the trailing one, which is invisible in the row sequence itself
        let mut seen: Vec<usize> = out.iter().map(|c| c.doc_ix).collect();
        seen.dedup();
        assert_eq!(
            seen,
            vec![1, 2],
            "only docs with enough tokens produce rows"
        );
        // rows come out ordered by (doc_ix, chunk_ix) — groupable without sorting
        let order: Vec<(usize, usize)> = out.iter().map(|c| (c.doc_ix, c.chunk_ix)).collect();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(order, sorted, "row order is (doc_ix, chunk_ix)");
        for c in &out {
            assert_eq!(
                c.embedding,
                embedding_for(&c.chunk.text, 8),
                "unprefixed fake vector"
            );
        }
    }

    #[tokio::test]
    async fn batch_respects_max_chunks_per_document() {
        let e = Embedder::fake(8);
        let mut opts = ChunkOptions::new(3);
        opts.min_tokens = 1;
        opts.overlap_tokens = 0;
        opts.max_chunks = std::num::NonZeroUsize::new(1);
        let texts = vec![
            "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9".to_string(),
            "x0 x1 x2 x3 x4 x5".to_string(),
        ];
        let out = e
            .embed_batch_document_chunks(&texts, &opts)
            .await
            .expect("batch");
        // exactly one row per doc, nothing beyond the first window
        assert_eq!(out.len(), 2, "one chunk per document");
        assert!(out.iter().all(|c| c.chunk_ix == 0));
        assert_eq!(out.iter().map(|c| c.doc_ix).collect::<Vec<_>>(), vec![0, 1]);
    }

    #[tokio::test]
    async fn document_prefix_is_applied_to_every_chunk() {
        let prefixes = Prefixes::new("search_query: ", "search_document: ");
        let e = Embedder::fake_with_prefixes(8, &prefixes);
        let opts = ChunkOptions {
            max_tokens: 3,
            overlap_tokens: 0,
            min_tokens: 1,
            max_chunks: None,
        };
        let chunks = e
            .embed_batch_document_chunks(&["aa bb cc dd ee"], &opts)
            .await
            .expect("chunks");
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert_eq!(
                c.embedding,
                embedding_for(&format!("search_document: {}", c.chunk.text), 8)
            );
        }
    }

    #[test]
    fn packed_len_rounds_up_to_bytes() {
        assert_eq!(packed_len(0), 0);
        assert_eq!(packed_len(1), 1);
        assert_eq!(packed_len(8), 1);
        assert_eq!(packed_len(9), 2);
        assert_eq!(packed_len(1024), 128);
    }

    #[test]
    fn binarize_packs_signs_lsb_first() {
        // dim 0 positive -> bit 0 set; dim 1 negative -> unset
        assert_eq!(binarize(&[1.0, -1.0]), vec![0b0000_0001]);
        // zeros are not "positive"
        assert_eq!(binarize(&[0.0, 0.0]), vec![0b0000_0000]);
        // nine dims -> two bytes, dim 8 in the second byte's bit 0
        let bits = binarize(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
        assert_eq!(bits, vec![0b0101_0101, 0b0000_0001]);
    }

    #[test]
    fn binarize_of_1024_dims_is_128_bytes() {
        let v = vec![1.0_f32; 1024];
        assert_eq!(binarize(&v).len(), 128);
    }

    #[test]
    fn quantize_u8_maps_range_endpoints() {
        assert_eq!(quantize_u8(&[-1.0, 0.0, 1.0]), vec![0, 128, 255]);
        // out-of-range components clamp
        assert_eq!(quantize_u8(&[-5.0, 5.0]), vec![0, 255]);
    }

    #[test]
    fn quantize_u8_error_is_half_a_step() {
        // lossy by at most half a quantization step (0.5 / 127.5)
        for i in 0..64_u8 {
            let x = (f32::from(i) / 64.0).mul_add(2.0, -1.0);
            let q = quantize_u8(&[x])[0];
            let restored = f32::from(q) / 127.5 - 1.0;
            assert!((x - restored).abs() <= 0.004, "error too large: {x}");
        }
    }

    #[tokio::test]
    async fn fake_embed_is_deterministic_and_correct_dim() {
        const TEST_DIM: usize = 768;
        let e = Embedder::fake(TEST_DIM);
        let a = e
            .embed_batch(&["rust backend role"])
            .await
            .expect("fake embed")
            .remove(0);
        let b = e
            .embed_batch(&["rust backend role"])
            .await
            .expect("fake embed")
            .remove(0);
        assert_eq!(a.len(), TEST_DIM);
        assert_eq!(a, b);

        let c = e
            .embed_batch(&["different text"])
            .await
            .expect("fake embed")
            .remove(0);
        assert_ne!(a, c);
    }

    #[tokio::test]
    async fn embed_batch_matches_individual() {
        let e = Embedder::fake(16);
        let batch = e.embed_batch(&["a", "b", "c"]).await.expect("fake batch");
        let a = e.embed_batch(&["a"]).await.expect("fake embed").remove(0);
        let b = e.embed_batch(&["b"]).await.expect("fake embed").remove(0);
        let c = e.embed_batch(&["c"]).await.expect("fake embed").remove(0);
        assert_eq!(batch, vec![a, b, c]);
    }

    #[test]
    fn const_helpers_are_deterministic() {
        assert_eq!(fnv1a("x"), fnv1a("x"));
        let mut s1 = 42_u64;
        let mut s2 = 42_u64;
        assert_eq!(splitmix64(&mut s1), splitmix64(&mut s2));
        let v = embedding_for("seed", 4);
        assert_eq!(v, embedding_for("seed", 4));
        assert!(v.iter().all(|x| (-1.0..1.0).contains(x)));
    }

    #[tokio::test]
    #[ignore = "downloads model"]
    async fn load_returns_expected_dim() {
        let e = test_embedder().await;
        assert_eq!(e.dim(), 384);
    }

    #[tokio::test]
    #[ignore = "downloads model"]
    async fn token_count_counts_every_word_and_empty_input() {
        let e = test_embedder().await;
        assert_eq!(
            e.token_count("").expect("count"),
            0,
            "empty text has no tokens"
        );
        let text = "hello world foo bar baz qux";
        let n = e.token_count(text).expect("count");
        let words = text.split_whitespace().count();
        assert!(
            n >= words,
            "every English word is at least one token (count {n} < {words} words)"
        );
        assert_eq!(
            e.token_count(text).expect("count"),
            n,
            "counting is deterministic"
        );
    }

    #[tokio::test]
    #[ignore = "downloads model"]
    async fn real_token_count_is_not_truncated() {
        let e = test_embedder().await;
        let long = "word ".repeat(2_000);
        let count = e.token_count(&long).expect("count");
        assert!(
            count > 1000,
            "count must reflect the whole text, not the ctx window ({count})"
        );
        assert!(e.model_max_tokens() > 0);
    }

    #[tokio::test]
    #[ignore = "downloads model"]
    async fn real_token_count_does_not_break_embedding() {
        let e = test_embedder().await;
        let long = "word ".repeat(3_000);
        // counting tokenizes without truncation (mutating a clone, not the model)
        assert!(e.token_count(&long).expect("count") > 2_000);
        // embedding the same text must still work (model tokenizer still truncates)
        let chunks = e
            .embed_batch_document_chunks(std::slice::from_ref(&long), &e.default_chunk_options())
            .await
            .expect("embed long text");
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].embedding.len(), e.dim());
    }

    #[tokio::test]
    #[ignore = "downloads model"]
    async fn real_chunking_covers_the_tail() {
        let e = test_embedder().await;
        let text = (0..1_200).fold(String::new(), |mut acc, i| {
            acc.push_str("token");
            acc.push_str(&i.to_string());
            acc.push(' ');
            acc
        });
        let text = text.trim();
        let opts = ChunkOptions {
            max_tokens: 200,
            overlap_tokens: 32,
            min_tokens: 1,
            max_chunks: None,
        };
        let chunks = e.chunk(text, &opts).expect("tokenizer-exact chunking");
        assert!(chunks.len() >= 5, "1200 tokens / 200 = 6 chunks");
        for c in &chunks {
            assert!(
                c.tokens <= opts.max_tokens,
                "chunk tokens {} > max",
                c.tokens
            );
            assert_eq!(&text[c.byte_start..c.byte_end], c.text);
        }
        assert_eq!(chunks.last().expect("chunks").byte_end, text.len());
        assert!(
            chunks.last().expect("chunks").text.contains("token1199"),
            "tail content must survive chunking"
        );
    }
}
