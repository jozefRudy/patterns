//! Embedding generation via fastembed (ONNX, in-process).
//!
//! Machinery: model loading, blocking-offload, batching, fake embedder for
//! tests, and query/document prefix handling. The **prefixes are model
//! configuration** — passed at load time (e.g. `"search_query: "` /
//! `"search_document: "` for nomic, `""`/`""` for BGE-M3) — after which
//! `embed_query` and the chunked document methods apply them automatically.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use fastembed::{EmbeddingModel, ExecutionProviderDispatch, TextEmbedding, TextInitOptions};
use ort::ep::cpu::CPU;

/// Model's query/document prefixes (empty strings when the model uses none).
#[derive(Clone, Debug)]
pub struct Prefixes {
    /// Prepended to queries, e.g. `"search_query: "`.
    pub query: String,
    /// Prepended to documents, e.g. `"search_document: "`.
    pub document: String,
}

impl Prefixes {
    /// No prefixes (symmetric models, e.g. BGE-M3).
    #[must_use]
    pub const fn none() -> Self {
        Self {
            query: String::new(),
            document: String::new(),
        }
    }
}

/// Chunking policy for document embedding (model-agnostic limits).
#[derive(Clone, Copy, Debug)]
pub struct ChunkOptions {
    /// Content tokens per chunk, excluding special tokens ([CLS]/[SEP]).
    pub max_tokens: usize,
    /// Token overlap between consecutive chunks.
    pub overlap_tokens: usize,
    /// Texts below this token count are not worth embedding.
    pub min_tokens: usize,
}

impl ChunkOptions {
    /// Options at `max_tokens` with the default overlap/minimum.
    #[must_use]
    pub const fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            overlap_tokens: 64,
            min_tokens: 5,
        }
    }
}

/// A slice of a source text, aligned to token boundaries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextChunk {
    pub text: String,
    pub tokens: usize,
    /// `source[byte_start..byte_end]`, outer whitespace trimmed.
    pub byte_start: usize,
    pub byte_end: usize,
}

/// One embedded chunk — a storage row: `(doc_ix → content id, chunk_ix, vector)`.
#[derive(Clone, Debug)]
pub struct EmbeddedChunk {
    /// Index of the source text in the batch's input slice (0 for single-doc calls).
    pub doc_ix: usize,
    /// Position of this chunk within its document.
    pub chunk_ix: usize,
    pub chunk: TextChunk,
    pub embedding: Vec<f32>,
}

/// Load-time configuration for [`Embedder`].
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Which fastembed model to run.
    pub model: EmbeddingModel,
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
    pub const fn new(model: EmbeddingModel) -> Self {
        Self {
            model,
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
        dim: usize,
        prefixes: Prefixes,
    },
    FastEmbed {
        model: Arc<Mutex<TextEmbedding>>,
        dim: usize,
        prefixes: Prefixes,
        /// Truncation-disabled copy for counting: `token_count` reads it
        /// immutably (`Tokenizer::encode` takes `&self`) — no lock, so counting
        /// never contends with inference on the model mutex.
        count_tokenizer: Arc<tokenizers::Tokenizer>,
    },
}

impl Embedder {
    /// Load a fastembed model into `cache_dir` (created if absent) per
    /// `options` (model, threads, prefixes, download progress), CPU EP.
    pub async fn load(options: LoadOptions, cache_dir: &Path) -> Result<Self> {
        let LoadOptions {
            model,
            intra_threads,
            prefixes,
            show_download_progress,
        } = options;
        let cache_dir = cache_dir.to_path_buf();
        tokio::fs::create_dir_all(&cache_dir).await?;
        let dim = TextEmbedding::get_model_info(&model)?.dim;
        let embedder = tokio::task::spawn_blocking(move || {
            let mut options = TextInitOptions::new(model)
                .with_cache_dir(cache_dir)
                .with_show_download_progress(show_download_progress)
                .with_execution_providers(vec![ExecutionProviderDispatch::from(CPU::default())]);
            if let Some(threads) = intra_threads {
                options = options.with_intra_threads(threads);
            }
            TextEmbedding::try_new(options)
        })
        .await??;

        // counting copy: truncation disabled once, then frozen behind an Arc —
        // encode() is &self, so counts are full-length and lock-free forever
        let mut count_tokenizer = embedder.tokenizer.clone();
        count_tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("disable truncation: {e}"))?;

        Ok(Self(Inner::FastEmbed {
            model: Arc::new(Mutex::new(embedder)),
            dim,
            prefixes,
            count_tokenizer: Arc::new(count_tokenizer),
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
            prefixes: prefixes.clone(),
        })
    }

    #[must_use]
    pub const fn dim(&self) -> usize {
        match &self.0 {
            Inner::Fake { dim, .. } | Inner::FastEmbed { dim, .. } => *dim,
        }
    }

    /// Embed a batch of raw texts (no prefix applied).
    ///
    /// # Panics
    /// Panics if the embedder mutex is poisoned (another task panicked
    /// while holding it) — invariant: single well-behaved owner.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        match &self.0 {
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
        }
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
                let encoding = tokenizer
                    .encode(text, false)
                    .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
                Ok(encoding
                    .get_offsets()
                    .iter()
                    .filter(|(start, end)| start != end)
                    .map(|(start, end)| *start..*end)
                    .collect())
            }
        }
    }

    fn prefixed_query(&self, text: &str) -> String {
        format!("{}{text}", self.prefixes().query)
    }

    fn prefixed_document(&self, text: &str) -> String {
        format!("{}{text}", self.prefixes().document)
    }

    const fn prefixes(&self) -> &Prefixes {
        match &self.0 {
            Inner::Fake { prefixes, .. } | Inner::FastEmbed { prefixes, .. } => prefixes,
        }
    }
}

/// Byte span of a single token within its source text.
type TokenSpan = std::ops::Range<usize>;

/// Tokens the tokenizer adds around every input ([CLS]/[SEP]) — excluded from
/// chunk content so the model never truncates a chunk we built.
const SPECIAL_TOKEN_HEADROOM: usize = 2;

/// Token window reported by the `Fake` embedder (no model, no tokenizer).
const FAKE_MODEL_MAX_TOKENS: usize = 512;

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

/// Whitespace-word spans — the Fake embedder's documented token approximation.
fn fake_token_spans(text: &str) -> Vec<TokenSpan> {
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    for (ix, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(s) = start.take() {
                spans.push(s..ix);
            }
        } else if start.is_none() {
            start = Some(ix);
        }
    }
    if let Some(s) = start {
        spans.push(s..text.len());
    }
    spans
}

/// Trim ASCII/Unicode whitespace from a byte span.
fn trim_span(text: &str, start: usize, end: usize) -> (usize, usize) {
    let slice = text.get(start..end).unwrap_or_default();
    let trimmed = slice.trim();
    if trimmed.is_empty() {
        return (start, start);
    }
    let lead = slice.len() - slice.trim_start().len();
    let new_start = start + lead;
    (new_start, new_start + trimmed.len())
}

/// Split pre-tokenized `spans` into chunks of at most `opts.max_tokens`,
/// preferring paragraph, then sentence, then whitespace boundaries.
///
/// Invariant: every non-whitespace byte of `text` is inside exactly one chunk
/// span's coverage (no tail loss); consecutive chunks overlap by at most
/// `opts.overlap_tokens`.
fn chunk_spans(text: &str, spans: &[TokenSpan], opts: &ChunkOptions) -> Vec<TextChunk> {
    let max = opts.max_tokens.max(1);
    let mut out = Vec::new();
    if spans.is_empty() {
        return out;
    }
    let mut start = 0_usize;
    while start < spans.len() {
        let remaining = spans.len() - start;
        if remaining <= max {
            if let Some(chunk) = make_chunk(text, spans, start, spans.len()) {
                out.push(chunk);
            }
            break;
        }
        let end = start + max;
        let break_tok = find_break(text, spans, start, end);
        let break_tok = if break_tok > start { break_tok } else { end };
        if let Some(chunk) = make_chunk(text, spans, start, break_tok) {
            out.push(chunk);
        }
        let next = break_tok.saturating_sub(opts.overlap_tokens).max(start + 1);
        start = next;
    }
    out
}

/// Last natural break inside `spans[start..end]`, as a token index.
fn find_break(text: &str, spans: &[TokenSpan], start: usize, end: usize) -> usize {
    let (Some(first), Some(last)) = (spans.get(start), spans.get(end.saturating_sub(1))) else {
        return end;
    };
    let Some(window) = text.get(first.start..last.end) else {
        return end;
    };
    let break_pos = ["\n\n", ". ", "! ", "? "]
        .iter()
        .filter_map(|sep| window.rfind(sep).map(|ix| ix + sep.len()))
        .max()
        .or_else(|| window.rfind(char::is_whitespace));
    let Some(pos) = break_pos else {
        return end;
    };
    let abs = first.start + pos;
    let ix = spans.get(start..end).map_or(0, |window| {
        window.iter().take_while(|span| span.end <= abs).count()
    });
    start + ix
}

fn make_chunk(text: &str, spans: &[TokenSpan], start: usize, end: usize) -> Option<TextChunk> {
    let raw_start = spans.get(start)?.start;
    let raw_end = spans.get(end.checked_sub(1)?)?.end;
    let (byte_start, byte_end) = trim_span(text, raw_start, raw_end);
    if byte_start >= byte_end {
        return None;
    }
    Some(TextChunk {
        text: text.get(byte_start..byte_end)?.to_string(),
        tokens: end - start,
        byte_start,
        byte_end,
    })
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
                Embedder::load(LoadOptions::new(EmbeddingModel::AllMiniLML6V2), &dir)
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
        };
        let chunks = fake_chunks(text, &opts);
        assert_eq!(chunks.len(), 5, "one chunk per token, no stalls");
        assert!(chunks.windows(2).all(|w| w[0].byte_start < w[1].byte_start));
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
    async fn document_prefix_is_applied_to_every_chunk() {
        let prefixes = Prefixes {
            query: "search_query: ".to_string(),
            document: "search_document: ".to_string(),
        };
        let e = Embedder::fake_with_prefixes(8, &prefixes);
        let opts = ChunkOptions {
            max_tokens: 3,
            overlap_tokens: 0,
            min_tokens: 1,
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
