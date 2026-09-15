//! Embedding generation via fastembed (ONNX, in-process).
//!
//! Machinery: model loading, blocking-offload, batching, fake embedder for
//! tests, and query/document prefix handling. The **prefixes are model
//! configuration** — passed at load time (e.g. `"search_query: "` /
//! `"search_document: "` for nomic, `""`/`""` for BGE-M3) — after which
//! `embed_query`/`embed_document` apply them automatically.

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
pub enum Embedder {
    Fake {
        dim: usize,
        prefixes: Prefixes,
    },
    FastEmbed {
        model: Arc<Mutex<TextEmbedding>>,
        dim: usize,
        prefixes: Prefixes,
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

        Ok(Self::FastEmbed {
            model: Arc::new(Mutex::new(embedder)),
            dim,
            prefixes,
        })
    }

    /// Deterministic fake embedder (hash-based) for tests, no prefixes.
    #[must_use]
    pub fn fake(dim: usize) -> Self {
        Self::fake_with_prefixes(dim, &Prefixes::none())
    }

    /// Deterministic fake embedder with the given prefixes for tests.
    #[must_use]
    pub fn fake_with_prefixes(dim: usize, prefixes: &Prefixes) -> Self {
        Self::Fake {
            dim,
            prefixes: prefixes.clone(),
        }
    }

    #[must_use]
    pub const fn dim(&self) -> usize {
        match self {
            Self::Fake { dim, .. } | Self::FastEmbed { dim, .. } => *dim,
        }
    }

    /// Embed a batch of raw texts (no prefix applied).
    ///
    /// # Panics
    /// Panics if the embedder mutex is poisoned (another task panicked
    /// while holding it) — invariant: single well-behaved owner.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Fake { dim, .. } => {
                let () = std::future::ready(()).await;
                Ok(texts.iter().map(|t| embedding_for(t, *dim)).collect())
            }
            Self::FastEmbed { model, .. } => {
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

    /// Embed a document with the model's document prefix.
    pub async fn embed_document(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_one(&self.prefixed_document(text)).await
    }

    /// Embed a batch of documents with the model's document prefix.
    pub async fn embed_batch_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = texts.iter().map(|t| self.prefixed_document(t)).collect();
        let refs: Vec<&str> = prefixed.iter().map(String::as_str).collect();
        self.embed_batch(&refs).await
    }

    fn prefixed_query(&self, text: &str) -> String {
        format!("{}{text}", self.prefixes().query)
    }

    fn prefixed_document(&self, text: &str) -> String {
        format!("{}{text}", self.prefixes().document)
    }

    const fn prefixes(&self) -> &Prefixes {
        match self {
            Self::Fake { prefixes, .. } | Self::FastEmbed { prefixes, .. } => prefixes,
        }
    }
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
    use super::*;

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

    #[tokio::test]
    async fn prefixes_are_applied() {
        let prefixes = Prefixes {
            query: "search_query: ".to_string(),
            document: "search_document: ".to_string(),
        };
        let e = Embedder::fake_with_prefixes(8, &prefixes);
        let q = e.embed_query("text").await.expect("fake query");
        let d = e.embed_document("text").await.expect("fake document");
        let raw = e.embed_batch(&["text"]).await.expect("fake raw").remove(0);
        // distinct prefixes -> distinct fake vectors; raw differs from both
        assert_ne!(q, d);
        assert_ne!(q, raw);
        assert_ne!(d, raw);

        let batch = e
            .embed_batch_documents(&["x".to_string()])
            .await
            .expect("fake batch");
        let single = e.embed_document("x").await.expect("fake doc");
        assert_eq!(batch, vec![single]);
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
        let dir = std::env::temp_dir().join("patterns_embed_tests");
        let e = Embedder::load(LoadOptions::new(EmbeddingModel::AllMiniLML6V2), &dir)
            .await
            .expect("model loads");
        assert_eq!(e.dim(), 384);
    }
}
