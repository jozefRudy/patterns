//! Cross-encoder reranker (fastembed `TextRerank`), loaded from a pinned hub
//! artifact through the same `HubSpec` + `hub::fetch` path as the embedder.
//!
//! Stage 2 of hybrid search: scores `(query, doc)` pairs jointly, which neither
//! BM25 nor the bi-encoder embedding can do. The score is a raw model score
//! (often a logit), not a calibrated probability - threshold it relatively, or
//! calibrate before using it as an absolute decision.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use fastembed::{RerankInitOptionsUserDefined, TextRerank, UserDefinedRerankingModel};

use super::LoadOptions;
use super::hub;
use super::spec::HubSpec;

/// One `(query, doc)` relevance score. `index` refers to the input slice order;
/// results are sorted by score, descending.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RerankHit {
    pub index: usize,
    pub score: f32,
}

/// Reranker handle. `Reranker::load` runs real ONNX inference;
/// `Reranker::fake` returns caller-supplied scores for tests.
pub struct Reranker(Inner);

/// Caller-supplied `score(query, doc)` for tests. `Arc` so a blocking task can
/// own a clone, exactly like the loaded model.
type FakeScorer = Arc<dyn Fn(&str, &str) -> f32 + Send + Sync>;

/// Internals stay private so the public API isn't pinned to fastembed/ort types.
enum Inner {
    Fake(FakeScorer),
    FastEmbed(Arc<Mutex<TextRerank>>),
}

impl Reranker {
    /// Load the pinned artifact described by `options` into `cache_dir` (created
    /// if absent), CPU execution provider. Same fetch/verify path as the embedder.
    ///
    /// # Errors
    /// A failed fetch or content-hash mismatch (see [`hub::fetch`]), a graph that
    /// cannot build a session, or a missing tokenizer file. Never falls back to
    /// another model.
    pub async fn load(options: LoadOptions<HubSpec>, cache_dir: &Path) -> Result<Self> {
        let LoadOptions {
            spec,
            intra_threads,
            show_download_progress,
        } = options;
        let cache_dir = cache_dir.to_path_buf();
        let reranker = tokio::task::spawn_blocking(move || -> Result<TextRerank> {
            std::fs::create_dir_all(&cache_dir).context("create the model cache dir")?;
            let client = hub::HubFiles::new(cache_dir, show_download_progress)?;
            let artifact = hub::fetch(&client, &spec)?;
            let model = UserDefinedRerankingModel::new(artifact.onnx, artifact.tokenizer_files);

            let mut init = RerankInitOptionsUserDefined::new()
                .with_execution_providers(vec![super::cpu_execution_provider()])
                .with_max_length(artifact.max_length);
            super::register_cpu_arena()?;
            super::use_env_allocators(&mut init.session_config);
            if let Some(threads) = intra_threads {
                init = init.with_intra_threads(threads);
            }
            TextRerank::try_new_from_user_defined(model, init).context("build the reranker session")
        })
        .await
        .context("the reranker load task failed")??;
        Ok(Self(Inner::FastEmbed(Arc::new(Mutex::new(reranker)))))
    }

    /// Deterministic fake reranker: `score(query, doc)`, for tests.
    #[must_use]
    pub fn fake(score: impl Fn(&str, &str) -> f32 + Send + Sync + 'static) -> Self {
        Self(Inner::Fake(Arc::new(score)))
    }

    /// Score every `doc` against `query`, descending, on the blocking pool — an
    /// async caller never stalls a runtime worker for the duration of the ONNX runs.
    ///
    /// `batch_size` caps how many docs go through one ONNX run (`None` = 256): it
    /// only splits the work into several runs with identical results, trading
    /// throughput for lower peak memory.
    ///
    /// Inputs stay borrowed: the owned copies `spawn_blocking`'s `'static` bound
    /// requires happen here, not at the call site.
    ///
    /// # Errors
    /// On tokenization or inference failure, a poisoned model lock, or a failed
    /// blocking task.
    pub async fn rerank<S: AsRef<str> + Sync>(
        &self,
        query: &str,
        docs: &[S],
        batch_size: Option<usize>,
    ) -> Result<Vec<RerankHit>> {
        match &self.0 {
            // no inference, no lock: nothing worth a blocking thread, and no copy
            Inner::Fake(score) => Ok(rerank_with_fake(score, query, docs)),
            Inner::FastEmbed(model) => {
                let model = Arc::clone(model);
                let query = query.to_owned();
                let docs: Vec<String> = docs.iter().map(|doc| doc.as_ref().to_owned()).collect();
                tokio::task::spawn_blocking(move || {
                    rerank_with_model(&model, &query, &docs, batch_size)
                })
                .await
                .context("the rerank task failed")?
            }
        }
    }
}

/// Score every doc against `query` with a caller-supplied scorer (tests).
fn rerank_with_fake<S: AsRef<str>>(score: &FakeScorer, query: &str, docs: &[S]) -> Vec<RerankHit> {
    let mut hits: Vec<RerankHit> = docs
        .iter()
        .enumerate()
        .map(|(index, doc)| RerankHit {
            index,
            score: score(query, doc.as_ref()),
        })
        .collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score));
    hits
}

/// Score every doc against `query` with the loaded cross-encoder.
fn rerank_with_model<S: AsRef<str>>(
    model: &Mutex<TextRerank>,
    query: &str,
    docs: &[S],
    batch_size: Option<usize>,
) -> Result<Vec<RerankHit>> {
    let docs: Vec<&str> = docs.iter().map(AsRef::as_ref).collect();
    let results = {
        let mut guard = model
            .lock()
            .map_err(|e| anyhow::anyhow!("reranker mutex poisoned: {e}"))?;
        guard.rerank(query, &docs, false, batch_size)?
    };
    Ok(results
        .into_iter()
        .map(|r| RerankHit {
            index: r.index,
            score: r.score,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fake_rerank_orders_by_score() {
        let r = Reranker::fake(|_q, doc| f32::from(u8::try_from(doc.len()).unwrap_or(u8::MAX)));
        let hits = r
            .rerank("q", &["a", "aaaa", "aa"], None)
            .await
            .expect("rerank");
        assert_eq!(hits.first().map(|h| h.index), Some(1), "longest first");
        assert_eq!(hits.last().map(|h| h.index), Some(0), "shortest last");
    }

    #[tokio::test]
    #[ignore = "downloads model (~90 MB first run)"]
    async fn loads_and_reranks() {
        let dir = tempfile::tempdir().expect("temp cache dir");
        // jinaai/jina-reranker-v1-turbo-en @ pinned revision.
        let spec = HubSpec::new(
            "jinaai/jina-reranker-v1-turbo-en",
            "b8c14f4e723d9e0aab4732a7b7b93741eeeb77c2",
            "onnx/model.onnx",
        )
        .expect("pinned spec");
        let r = Reranker::load(LoadOptions::new(spec), dir.path())
            .await
            .expect("reranker loads");
        let hit = r
            .rerank(
                "rust programming language",
                &["errors in rust lifetimes", "banana bread recipe"],
                None,
            )
            .await
            .expect("rerank");
        assert_eq!(hit.first().map(|h| h.index), Some(0), "relevant doc first");
    }
}
