//! OpenAI-compatible embeddings client (`POST {base_url}/embeddings`), sibling
//! to `embed`/`llm_cli`/`systemone`.
//!
//! Backends differ only by `base_url` (`DeepInfra`, `OpenAI`, Together,
//! `SiliconFlow`). The caller supplies base URL, API key, model and dimensions
//! — the client never reads the environment. Only the OpenAI-compatible schema
//! is in scope; native Cohere/Voyage/Jina APIs are not supported.
//!
//! `dims = None` sends no `dimensions` field (the model's native output, as for
//! non-MRL models); `dims = Some(n)` requests an MRL-truncated `n`-dimensional
//! output (`n >= 32`). Truncated vectors are not renormalized.
//!
//! Query/document prefixes (via [`Prefixes`], model config) are applied by
//! [`EmbeddingApi::embed_query`] and [`EmbeddingApi::embed_documents`];
//! documents are always chunked to the declared `max_tokens` window (requires
//! `.with_max_tokens` + `.with_tokenizer`), queries never are.

use crate::chunk::{ChunkOptions, EmbeddedChunk, TextChunk};
use crate::limits::ConcurrencyLimits;
use crate::prefixes::Prefixes;
use crate::retry::{HttpResponse, RetryPolicy};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::{OnceCell, Semaphore};
use tokio::task::JoinSet;

/// Default inputs per request.
const DEFAULT_MAX_BATCH_SIZE: usize = 256;
/// Smallest MRL output dimension accepted by OpenAI-compatible providers.
const MIN_DIMS: usize = 32;

#[derive(Serialize)]
struct RequestBody<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<&'a str>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
    index: usize,
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// HTTP seam; injectable so tests run fully offline.
trait Transport: Send + Sync + 'static {
    fn post<'a>(
        &'a self,
        url: &'a str,
        api_key: &'a str,
        body: &'a [u8],
    ) -> BoxFuture<'a, Result<HttpResponse>>;
}

struct HttpTransport {
    client: reqwest::Client,
}

impl Transport for HttpTransport {
    fn post<'a>(
        &'a self,
        url: &'a str,
        api_key: &'a str,
        body: &'a [u8],
    ) -> BoxFuture<'a, Result<HttpResponse>> {
        Box::pin(async move {
            let response = self
                .client
                .post(url)
                .bearer_auth(api_key)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body.to_vec())
                .send()
                .await
                .context("embeddings request failed")?;
            let status = response.status().as_u16();
            let body = response
                .text()
                .await
                .context("read embeddings response body")?;
            Ok(HttpResponse { status, body })
        })
    }
}

/// Immutable per-handle configuration (everything except transport/permits).
#[derive(Clone)]
struct ApiConfig {
    base_url: String,
    api_key: String,
    model: String,
    dims: Option<usize>,
    service_tier: Option<String>,
    max_batch_size: usize,
    prefixes: Prefixes,
    max_tokens: Option<usize>,
    limits: ConcurrencyLimits,
}

/// Bounded, cloneable embeddings client. Cheap to clone — share one handle so
/// the concurrency cap holds. The API key is redacted in [`Debug`].
#[derive(Clone)]
pub struct EmbeddingApi {
    config: ApiConfig,
    tokenizer_source: Option<String>,
    hf_home: Option<PathBuf>,
    tokenizer: Arc<OnceCell<Tokenizer>>,
    retry: RetryPolicy,
    permits: Arc<Semaphore>,
    transport: Arc<dyn Transport>,
}

impl fmt::Debug for EmbeddingApi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmbeddingApi")
            .field("base_url", &self.config.base_url)
            .field("api_key", &"<redacted>")
            .field("model", &self.config.model)
            .field("dims", &self.config.dims)
            .field("service_tier", &self.config.service_tier)
            .field("max_batch_size", &self.config.max_batch_size)
            .field("prefixes", &self.config.prefixes)
            .field("max_tokens", &self.config.max_tokens)
            .field("limits", &self.config.limits)
            .field("tokenizer_source", &self.tokenizer_source)
            .field("hf_home", &self.hf_home)
            .field("retry", &self.retry)
            .finish_non_exhaustive()
    }
}

impl EmbeddingApi {
    /// Build from a base URL, API key, model and shared limits.
    ///
    /// `base_url` is the OpenAI-compatible root without a trailing slash, e.g.
    /// `"https://api.deepinfra.com/v1/openai"` or `"https://api.openai.com/v1"`.
    /// Vectors use the model's native dimensions unless
    /// [`.with_mrl_truncation`](Self::with_mrl_truncation) is called.
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        limits: ConcurrencyLimits,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .build()
            .context("build reqwest client")?;
        let config = ApiConfig {
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            dims: None,
            service_tier: None,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            prefixes: Prefixes::none(),
            max_tokens: None,
            limits,
        };
        Ok(Self::with_transport(
            Arc::new(HttpTransport { client }),
            config,
        ))
    }

    /// Request Matryoshka (MRL) truncation to `dims` dimensions; default is the
    /// model's native dimensions.
    ///
    /// # Panics
    /// If `dims < 32`.
    #[must_use]
    pub fn with_mrl_truncation(mut self, dims: usize) -> Self {
        assert!(
            dims >= MIN_DIMS,
            "requested dimensions {dims} below minimum {MIN_DIMS}"
        );
        self.config.dims = Some(dims);
        self
    }

    /// Set the model's query/document prefixes (see [`Prefixes`]). Defaults to
    /// none; [`Self::embed_query`] and [`Self::embed_documents`] apply them.
    #[must_use]
    pub fn with_prefixes(mut self, prefixes: &Prefixes) -> Self {
        self.config.prefixes = prefixes.clone();
        self
    }

    /// Fetch query/document prefixes from the model's Hugging Face
    /// `config_sentence_transformers.json` (e.g. Qwen3-Embedding's
    /// `Instruct: ...\nQuery:`), using the configured model as the repo id.
    ///
    /// `query_key`/`document_key` name the `prompts` entries to read — the keys
    /// vary by model (`query`/`document`, `query`/`passage`,
    /// `retrieval.query`/`retrieval.passage`). Requires
    /// [`.with_hf_home`](Self::with_hf_home); fails when the file, its `prompts`
    /// map, or either key is missing. Overrides any earlier [`Self::with_prefixes`].
    pub async fn with_prefixes_from_hf(
        self,
        query_key: impl Into<String>,
        document_key: impl Into<String>,
    ) -> Result<Self> {
        let repo = self.config.model.clone();
        let hf_home = self.hf_home.clone();
        let query_key = query_key.into();
        let document_key = document_key.into();
        let prefixes = tokio::task::spawn_blocking(move || {
            let path = get_repo_file(
                &repo,
                "config_sentence_transformers.json",
                hf_home.as_deref(),
            )?;
            let json = std::fs::read_to_string(&path)
                .with_context(|| format!("read {}", path.display()))?;
            extract_prefixes(&json, &query_key, &document_key)
        })
        .await
        .context("join prefixes fetch")??;
        let mut this = self;
        this.config.prefixes = prefixes;
        Ok(this)
    }

    /// Declare the model's context window in tokens (read from the model card).
    ///
    /// Required before [`Self::default_chunk_options`]; not introspectable from
    /// an OpenAI-compatible endpoint. Set it from the card, not `config.json`
    /// (`max_position_embeddings` / `model_max_length` often exceed the real
    /// window — e.g. Qwen3-Embedding-8B: 40960 / 131072 vs 32768).
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.config.max_tokens = Some(max_tokens);
        self
    }

    /// Set the optional `service_tier` (e.g. `"flex"`); omitted when unset.
    #[must_use]
    pub fn with_service_tier(mut self, tier: impl Into<String>) -> Self {
        self.config.service_tier = Some(tier.into());
        self
    }

    /// Set inputs per request (default 256).
    ///
    /// # Panics
    /// If `n == 0`.
    #[must_use]
    pub fn with_max_batch_size(mut self, n: usize) -> Self {
        assert!(n > 0, "max_batch_size must be positive");
        self.config.max_batch_size = n;
        self
    }

    /// Configure lazy local tokenizer: `source` is a local `tokenizer.json`
    /// path or an HF repo id (e.g. `"Qwen/Qwen3-Embedding-8B"`). Required
    /// before [`Self::token_count`].
    #[must_use]
    pub fn with_tokenizer(mut self, source: impl Into<String>) -> Self {
        self.tokenizer_source = Some(source.into());
        self
    }

    /// Set the Hugging Face home used for the tokenizer cache/auth.
    ///
    /// Required when `tokenizer_source` is an HF repo id: the cache is
    /// `{home}/hub` and the token file is `{home}/token`. Ignored for a local
    /// `tokenizer.json` path. The client never reads `HF_HOME` from the
    /// environment.
    #[must_use]
    pub fn with_hf_home(mut self, home: impl Into<PathBuf>) -> Self {
        self.hf_home = Some(home.into());
        self
    }

    /// Set the transient-failure retry policy (default 3 attempts, 250ms→2s).
    #[must_use]
    pub const fn with_retry_policy(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Test seam: build with an injected transport and explicit config.
    fn with_transport(transport: Arc<dyn Transport>, config: ApiConfig) -> Self {
        let permits = Arc::new(Semaphore::new(config.limits.max_concurrent_calls));
        Self {
            config,
            tokenizer_source: None,
            hf_home: None,
            tokenizer: Arc::new(OnceCell::new()),
            retry: RetryPolicy::default(),
            permits,
            transport,
        }
    }

    /// Embed one query with the model's query prefix.
    ///
    /// Queries are never chunked — one vector per call. Transient failures
    /// (429/5xx/timeouts) are retried with backoff, payload 4xx fail
    /// immediately.
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let prefix = &self.config.prefixes.query;
        let mut vectors = self.embed_owned(vec![format!("{prefix}{text}")]).await?;
        vectors
            .pop()
            .ok_or_else(|| anyhow::anyhow!("expected one embedding, got none"))
    }

    /// Embed documents, chunked to `opts`, returning a flat row batch.
    ///
    /// Each document is split on token boundaries (see [`ChunkOptions`]), the
    /// document prefix is applied per chunk, and all chunks across the batch go
    /// out as one bounded-concurrency request batch. No tail loss; rows are
    /// ordered by `(doc_ix, chunk_ix)`. Documents below `opts.min_tokens`
    /// contribute no rows — derive the skip set from the `doc_ix` values
    /// present. Requires [`.with_tokenizer`](Self::with_tokenizer).
    pub async fn embed_documents(
        &self,
        texts: &[impl AsRef<str> + Sync],
        opts: &ChunkOptions,
    ) -> Result<Vec<EmbeddedChunk>> {
        let tokenizer = self.tokenizer().await?;
        let mut pending: Vec<(usize, usize, TextChunk)> = Vec::new();
        for (doc_ix, text) in texts.iter().enumerate() {
            let text = text.as_ref();
            let spans = crate::chunk::token_spans(tokenizer, text)?;
            if spans.len() < opts.min_tokens {
                continue;
            }
            pending.extend(
                crate::chunk::chunk_spans(text, &spans, opts)
                    .into_iter()
                    .enumerate()
                    .map(|(chunk_ix, chunk)| (doc_ix, chunk_ix, chunk)),
            );
        }
        if pending.is_empty() {
            return Ok(Vec::new());
        }
        let prefix = &self.config.prefixes.document;
        let prefixed: Vec<String> = pending
            .iter()
            .map(|(_, _, chunk)| format!("{prefix}{}", chunk.text))
            .collect();
        let embeddings = self.embed_owned(prefixed).await?;
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

    /// Chunk options sized to the declared context window (see
    /// [`Self::with_max_tokens`]), minus the special tokens the tokenizer adds
    /// to every input ([CLS]/[SEP]). Errors when `max_tokens` is not set.
    pub fn default_chunk_options(&self) -> Result<ChunkOptions> {
        let max = self
            .config
            .max_tokens
            .context("max_tokens not configured (call .with_max_tokens)")?
            .saturating_sub(crate::chunk::SPECIAL_TOKEN_HEADROOM)
            .max(1);
        Ok(ChunkOptions::new(max))
    }

    /// Batched implementation over owned inputs.
    async fn embed_owned(&self, owned: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if owned.is_empty() {
            return Ok(Vec::new());
        }
        let mut set = JoinSet::new();
        for (ordinal, batch) in owned.chunks(self.config.max_batch_size).enumerate() {
            let handle = self.clone();
            let batch = batch.to_vec();
            set.spawn(async move { (ordinal, handle.embed_batch(&batch).await) });
        }
        let mut batches: Vec<(usize, Vec<Vec<f32>>)> = Vec::new();
        while let Some(joined) = set.join_next().await {
            let (ordinal, result) = joined.context("embeddings batch task failed")?;
            batches.push((ordinal, result?));
        }
        batches.sort_by_key(|(ordinal, _)| *ordinal);
        Ok(batches
            .into_iter()
            .flat_map(|(_, vectors)| vectors)
            .collect())
    }

    /// One request for one batch, with retries.
    async fn embed_batch(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let _permit = self
            .permits
            .acquire()
            .await
            .context("embeddings semaphore closed")?;
        let url = format!("{}/embeddings", self.config.base_url);
        let body = serde_json::to_vec(&self.build_body(inputs))
            .context("serialize embeddings request body")?;
        let response = self
            .retry
            .run("embeddings", self.config.limits.call_timeout, {
                move || {
                    let transport = Arc::clone(&self.transport);
                    let url = url.clone();
                    let body = body.clone();
                    async move { transport.post(&url, &self.config.api_key, &body).await }
                }
            })
            .await?;
        parse_response(&response.body, inputs.len())
    }

    /// Assemble the request body for a batch.
    fn build_body<'a>(&'a self, inputs: &'a [String]) -> RequestBody<'a> {
        RequestBody {
            model: &self.config.model,
            input: inputs,
            encoding_format: "float",
            dimensions: self.config.dims,
            service_tier: self.config.service_tier.as_deref(),
        }
    }

    /// Lazy tokenizer handle; errors when [`Self::with_tokenizer`] was not called.
    async fn tokenizer(&self) -> Result<&Tokenizer> {
        let source = self
            .tokenizer_source
            .as_deref()
            .context("tokenizer not configured (call .with_tokenizer)")?;
        self.tokenizer
            .get_or_try_init(|| load_tokenizer(source, self.hf_home.as_deref()))
            .await
    }

    /// Full-text token count (truncation disabled), using the configured
    /// tokenizer. Requires [`Self::with_tokenizer`]; the tokenizer is loaded
    /// once, lazily, then shared across clones.
    pub async fn token_count(&self, text: &str) -> Result<usize> {
        let tokenizer = self.tokenizer().await?;
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(encoding.get_ids().len())
    }
}

/// Load the tokenizer from a local `tokenizer.json` path or an HF repo id.
async fn load_tokenizer(source: &str, hf_home: Option<&Path>) -> Result<Tokenizer> {
    let mut tokenizer = if std::path::Path::new(source).is_file() {
        let path = source.to_owned();
        tokio::task::spawn_blocking(move || Tokenizer::from_file(&path))
            .await
            .context("join tokenizer file load")?
            .map_err(|e| anyhow::anyhow!("load tokenizer from file {source}: {e}"))?
    } else {
        let repo = source.to_owned();
        let home = hf_home.map(Path::to_path_buf);
        tokio::task::spawn_blocking(move || load_hf_tokenizer(&repo, home.as_deref()))
            .await
            .context("join tokenizer download")??
    };
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow::anyhow!("disable tokenizer truncation: {e}"))?;
    Ok(tokenizer)
}

/// Download one file from an HF repo (cached under `{hf_home}/hub`).
fn get_repo_file(repo: &str, filename: &str, hf_home: Option<&Path>) -> Result<PathBuf> {
    let home = hf_home.context("hf_home not configured (call .with_hf_home)")?;
    let api = hf_hub::api::sync::ApiBuilder::from_cache(hf_hub::Cache::new(home.join("hub")))
        .build()
        .context("init hf-hub api")?;
    api.model(repo.to_owned())
        .get(filename)
        .map_err(|e| anyhow::anyhow!("download {filename} for {repo}: {e}"))
}

/// Download `tokenizer.json` from an HF repo (cached under `{hf_home}/hub`).
fn load_hf_tokenizer(repo: &str, hf_home: Option<&Path>) -> Result<Tokenizer> {
    let path = get_repo_file(repo, "tokenizer.json", hf_home)?;
    Tokenizer::from_file(&path)
        .map_err(|e| anyhow::anyhow!("load tokenizer from {}: {e}", path.display()))
}

/// The `prompts` map of a Hugging Face `config_sentence_transformers.json`.
#[derive(Deserialize)]
struct SentenceTransformersConfig {
    #[serde(default)]
    prompts: Option<HashMap<String, String>>,
}

/// Read two named prompt strings out of a `config_sentence_transformers.json`.
fn extract_prefixes(json: &str, query_key: &str, document_key: &str) -> Result<Prefixes> {
    let config: SentenceTransformersConfig =
        serde_json::from_str(json).context("parse config_sentence_transformers.json")?;
    let prompts = config
        .prompts
        .context("config_sentence_transformers.json has no `prompts` map")?;
    let query = prompts
        .get(query_key)
        .with_context(|| format!("prompts has no key `{query_key}`"))?
        .clone();
    let document = prompts
        .get(document_key)
        .with_context(|| format!("prompts has no key `{document_key}`"))?
        .clone();
    Ok(Prefixes { query, document })
}

/// Parse a 2xx body, ordering vectors by `index` and validating the batch shape.
fn parse_response(body: &str, expected: usize) -> Result<Vec<Vec<f32>>> {
    let response: EmbeddingResponse =
        serde_json::from_str(body).with_context(|| format!("parse embeddings response: {body}"))?;
    let mut items = response.data;
    items.sort_by_key(|item| item.index);
    for (position, item) in items.iter().enumerate() {
        anyhow::ensure!(
            item.index == position,
            "embeddings response index {} out of order at position {position}",
            item.index
        );
    }
    anyhow::ensure!(
        items.len() == expected,
        "embeddings response has {} items, expected {expected}",
        items.len()
    );
    Ok(items.into_iter().map(|item| item.embedding).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    #[derive(Clone)]
    struct Recorded {
        url: String,
        api_key: String,
        body: Value,
    }

    struct FakeTransport {
        responses: Mutex<VecDeque<HttpResponse>>,
        requests: Mutex<Vec<Recorded>>,
    }

    impl FakeTransport {
        fn new(responses: Vec<HttpResponse>) -> Arc<Self> {
            Arc::new(Self {
                responses: Mutex::new(responses.into_iter().collect()),
                requests: Mutex::new(Vec::new()),
            })
        }

        fn last_request(&self) -> Recorded {
            self.requests
                .lock()
                .expect("lock requests")
                .last()
                .cloned()
                .expect("a request was recorded")
        }

        fn call_count(&self) -> usize {
            self.requests.lock().expect("lock requests").len()
        }
    }

    impl Transport for FakeTransport {
        fn post<'a>(
            &'a self,
            url: &'a str,
            api_key: &'a str,
            body: &'a [u8],
        ) -> BoxFuture<'a, Result<HttpResponse>> {
            self.requests.lock().expect("lock requests").push(Recorded {
                url: url.to_owned(),
                api_key: api_key.to_owned(),
                body: serde_json::from_slice(body).expect("request body is JSON"),
            });
            let next = self.responses.lock().expect("lock responses").pop_front();
            Box::pin(async move { next.ok_or_else(|| anyhow::anyhow!("no fake response queued")) })
        }
    }

    fn ok(body: String) -> HttpResponse {
        HttpResponse { status: 200, body }
    }

    fn failing(status: u16, body: &str) -> HttpResponse {
        HttpResponse {
            status,
            body: body.to_owned(),
        }
    }

    fn response_body(vectors: &[(usize, Vec<f32>)]) -> String {
        let data: Vec<Value> = vectors
            .iter()
            .map(|(index, embedding)| json!({"embedding": embedding, "index": index}))
            .collect();
        json!({"data": data, "usage": {"prompt_tokens": 1, "total_tokens": 1}}).to_string()
    }

    fn firsts(vectors: &[Vec<f32>]) -> Vec<f32> {
        vectors
            .iter()
            .map(|v| v.first().copied().unwrap_or(f32::NAN))
            .collect()
    }

    fn client(transport: Arc<dyn Transport>) -> EmbeddingApi {
        client_with_dims(transport, None)
    }

    fn client_with_dims(transport: Arc<dyn Transport>, dims: Option<usize>) -> EmbeddingApi {
        let config = ApiConfig {
            base_url: "https://example.test/v1/openai".to_owned(),
            api_key: "secret".to_owned(),
            model: "Qwen/Qwen3-Embedding-8B".to_owned(),
            dims,
            service_tier: None,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            prefixes: Prefixes::none(),
            max_tokens: None,
            limits: ConcurrencyLimits::default(),
        };
        EmbeddingApi::with_transport(transport, config)
    }

    #[tokio::test]
    async fn applies_query_and_document_prefixes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, WORDLEVEL_TOKENIZER).expect("write tokenizer");
        let prefixes = Prefixes::new("Instruct: retrieve\nQuery:", "");
        let transport = FakeTransport::new(vec![
            ok(response_body(&[(0, vec![1.0])])),
            ok(response_body(&[(0, vec![2.0])])),
        ]);
        let api = client(transport.clone())
            .with_prefixes(&prefixes)
            .with_tokenizer(path.to_str().expect("utf8 path"));

        api.embed_query("capital").await.expect("query");
        let query_input = transport
            .last_request()
            .body
            .get("input")
            .cloned()
            .expect("input field");
        assert_eq!(query_input, json!(["Instruct: retrieve\nQuery:capital"]));

        let opts = ChunkOptions {
            max_tokens: 16,
            overlap_tokens: 0,
            min_tokens: 1,
        };
        let rows = api
            .embed_documents(&["hello world"], &opts)
            .await
            .expect("documents");
        let document_input = transport
            .last_request()
            .body
            .get("input")
            .cloned()
            .expect("input field");
        assert_eq!(
            document_input,
            json!(["hello world"]),
            "empty document prefix leaves input unchanged"
        );
        assert_eq!(rows.len(), 1);
    }

    #[tokio::test]
    async fn chunks_documents_and_skips_short() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, WORDLEVEL_TOKENIZER).expect("write tokenizer");
        let transport =
            FakeTransport::new(vec![ok(response_body(&[(0, vec![0.0]), (1, vec![1.0])]))]);
        let api = client(transport.clone()).with_tokenizer(path.to_str().expect("utf8 path"));
        let opts = ChunkOptions {
            max_tokens: 1,
            overlap_tokens: 0,
            min_tokens: 2,
        };

        // doc0 "hello world" (2 tokens) -> two 1-token chunks; doc1 "hello"
        // (1 token) is below min_tokens and contributes nothing.
        let rows = api
            .embed_documents(&["hello world", "hello"], &opts)
            .await
            .expect("documents");

        assert_eq!(rows.len(), 2);
        assert_eq!(transport.call_count(), 1, "one batched request");
        let first = rows.first().expect("first row");
        let second = rows.get(1).expect("second row");
        assert_eq!((first.doc_ix, first.chunk_ix), (0, 0));
        assert_eq!((second.doc_ix, second.chunk_ix), (0, 1));
        assert_eq!(first.chunk.text, "hello");
        assert_eq!(second.chunk.text, "world");
    }

    #[test]
    fn default_chunk_options_require_max_tokens() {
        let configured = client(FakeTransport::new(vec![])).with_max_tokens(512);
        let opts = configured.default_chunk_options().expect("opts");
        assert_eq!(opts.max_tokens, 512 - crate::chunk::SPECIAL_TOKEN_HEADROOM);

        let unset = client(FakeTransport::new(vec![]));
        let error = unset.default_chunk_options().expect_err("must error");
        assert!(
            format!("{error:#}").contains("not configured"),
            "error: {error:#}"
        );
    }

    #[test]
    fn extract_prefixes_reads_named_keys() {
        let json = r#"{"prompts":{"query":"Q:","document":""}}"#;
        let prefixes = extract_prefixes(json, "query", "document").expect("prefixes");
        assert_eq!(prefixes.query, "Q:");
        assert_eq!(prefixes.document, "");

        let missing_key =
            extract_prefixes(json, "query", "passage").expect_err("missing key errors");
        assert!(format!("{missing_key:#}").contains("passage"));

        let no_prompts = extract_prefixes(r#"{"default_prompt_name":null}"#, "query", "document")
            .expect_err("absent prompts errors");
        assert!(format!("{no_prompts:#}").contains("prompts"));
    }

    #[tokio::test]
    async fn sends_expected_request_body() {
        let transport = FakeTransport::new(vec![ok(response_body(&[(0, vec![0.1])]))]);
        let api = client_with_dims(transport.clone(), Some(64))
            .with_service_tier("flex")
            .with_max_batch_size(2);
        api.embed_query("hello").await.expect("embed");

        let request = transport.last_request();
        assert_eq!(request.url, "https://example.test/v1/openai/embeddings");
        assert_eq!(request.api_key, "secret");
        assert_eq!(
            request.body,
            json!({
                "model": "Qwen/Qwen3-Embedding-8B",
                "input": ["hello"],
                "encoding_format": "float",
                "dimensions": 64,
                "service_tier": "flex"
            })
        );
    }

    #[tokio::test]
    async fn omits_dimensions_and_service_tier_when_unset() {
        let transport = FakeTransport::new(vec![ok(response_body(&[(0, vec![0.1])]))]);
        let api = client(transport.clone());
        api.embed_query("hello").await.expect("embed");

        assert_eq!(
            transport.last_request().body,
            json!({
                "model": "Qwen/Qwen3-Embedding-8B",
                "input": ["hello"],
                "encoding_format": "float"
            })
        );
    }

    #[tokio::test]
    async fn restores_index_order() {
        let transport = FakeTransport::new(vec![ok(response_body(&[
            (1, vec![1.0, 1.1]),
            (0, vec![0.0, 0.1]),
        ]))]);
        let out = client(transport)
            .embed_owned(vec!["a".to_owned(), "b".to_owned()])
            .await
            .expect("embed");

        assert_eq!(out.len(), 2);
        let first = out.first().expect("first").first().copied().expect("dim");
        let second = out.get(1).expect("second").first().copied().expect("dim");
        assert!((first - 0.0).abs() < 1e-6);
        assert!((second - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn splits_into_batches_and_preserves_order() {
        let transport = FakeTransport::new(vec![
            ok(response_body(&[(0, vec![0.0]), (1, vec![1.0])])),
            ok(response_body(&[(0, vec![2.0]), (1, vec![3.0])])),
            ok(response_body(&[(0, vec![4.0])])),
        ]);
        let api = client(transport.clone()).with_max_batch_size(2);
        let out = api
            .embed_owned(vec![
                "a".to_owned(),
                "b".to_owned(),
                "c".to_owned(),
                "d".to_owned(),
                "e".to_owned(),
            ])
            .await
            .expect("embed");

        assert_eq!(transport.call_count(), 3, "one request per batch");
        for (got, want) in firsts(&out).into_iter().zip([0.0, 1.0, 2.0, 3.0, 4.0]) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[tokio::test]
    async fn retries_transient_then_succeeds() {
        let transport = FakeTransport::new(vec![
            failing(429, "slow down"),
            ok(response_body(&[(0, vec![1.0])])),
        ]);
        let out = client(transport.clone())
            .embed_query("a")
            .await
            .expect("embed");

        assert_eq!(transport.call_count(), 2);
        assert_eq!(out.len(), 1);
    }

    #[tokio::test]
    async fn fails_fast_on_payload_error() {
        let transport = FakeTransport::new(vec![failing(400, "bad input")]);
        let error = client(transport.clone())
            .embed_query("a")
            .await
            .expect_err("must fail");

        assert_eq!(transport.call_count(), 1, "no retry on 4xx");
        assert!(format!("{error:#}").contains("400"), "error: {error:#}");
    }

    #[tokio::test]
    async fn rejects_short_response() {
        let transport = FakeTransport::new(vec![ok(response_body(&[(0, vec![1.0])]))]);
        let error = client(transport)
            .embed_owned(vec!["a".to_owned(), "b".to_owned()])
            .await
            .expect_err("must fail");
        assert!(
            format!("{error:#}").contains("expected 2"),
            "error: {error:#}"
        );
    }

    #[test]
    #[should_panic(expected = "below minimum")]
    fn rejects_dims_below_minimum() {
        let _api = client(FakeTransport::new(vec![])).with_mrl_truncation(16);
    }

    #[tokio::test]
    async fn empty_input_makes_no_request() {
        let transport = FakeTransport::new(vec![]);
        let out = client(transport.clone())
            .embed_owned(Vec::new())
            .await
            .expect("embed");
        assert_eq!(out.len(), 0);
        assert_eq!(transport.call_count(), 0);
    }

    #[test]
    fn debug_redacts_api_key() {
        let api = client(FakeTransport::new(vec![]));
        let rendered = format!("{api:?}");
        assert!(rendered.contains("<redacted>"), "debug: {rendered}");
        assert!(!rendered.contains("secret"), "debug: {rendered}");
    }

    /// Minimal valid tokenizer.json (Whitespace pre-tokenizer + `WordLevel`).
    const WORDLEVEL_TOKENIZER: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": {"type": "Whitespace"},
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": {"[UNK]": 0, "hello": 1, "world": 2},
        "unk_token": "[UNK]"
      }
    }"#;

    #[tokio::test]
    async fn token_count_uses_configured_tokenizer() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, WORDLEVEL_TOKENIZER).expect("write tokenizer");
        let api =
            client(FakeTransport::new(vec![])).with_tokenizer(path.to_str().expect("utf8 path"));

        assert_eq!(api.token_count("hello world").await.expect("count"), 2);
        assert_eq!(api.token_count("hello").await.expect("count"), 1);
        assert_eq!(api.token_count("unknown").await.expect("count"), 1);
    }

    #[tokio::test]
    async fn token_count_without_tokenizer_errors() {
        let api = client(FakeTransport::new(vec![]));
        let error = api.token_count("hi").await.expect_err("must error");
        assert!(
            format!("{error:#}").contains("not configured"),
            "error: {error:#}"
        );
    }

    #[tokio::test]
    async fn hf_repo_tokenizer_requires_hf_home() {
        let api = client(FakeTransport::new(vec![])).with_tokenizer("Qwen/Qwen3-Embedding-8B");
        let error = api.token_count("hi").await.expect_err("must error");
        assert!(
            format!("{error:#}").contains("hf_home not configured"),
            "error: {error:#}"
        );
    }

    /// Client for the live integration test: only the API key comes from the env.
    fn live_client(dims: Option<usize>) -> EmbeddingApi {
        let api_key = std::env::var("EMBED_API_KEY").expect("EMBED_API_KEY set");
        let api = EmbeddingApi::new(
            "https://api.deepinfra.com/v1/openai",
            api_key,
            "Qwen/Qwen3-Embedding-8B",
            ConcurrencyLimits::default(),
        )
        .expect("build EmbeddingApi");
        match dims {
            Some(n) => api.with_mrl_truncation(n),
            None => api,
        }
    }

    #[tokio::test]
    #[ignore = "live: requires EMBED_API_KEY"]
    async fn live_qwen_round_trip() {
        // MRL: native output exceeds the requested truncation; truncated == 128.
        let native = live_client(None)
            .embed_query("hello world")
            .await
            .expect("native embed");
        let truncated = live_client(Some(128))
            .embed_query("hello world")
            .await
            .expect("mrl embed");
        assert!(
            native.len() > 128,
            "model native dims {} must exceed requested 128",
            native.len()
        );
        assert_eq!(truncated.len(), 128, "truncated dims must equal requested");

        // Tokenizer comes from the HF repo (downloads tokenizer.json once).
        let cache = tempfile::tempdir().expect("tempdir");
        let api = live_client(Some(128))
            .with_tokenizer("Qwen/Qwen3-Embedding-8B")
            .with_hf_home(cache.path())
            .with_max_tokens(32_768);
        assert!(api.token_count("hello world").await.expect("count") > 0);
        assert_eq!(
            api.default_chunk_options().expect("opts").max_tokens,
            32_768 - crate::chunk::SPECIAL_TOKEN_HEADROOM
        );

        // Prefixes fetched from the model's config_sentence_transformers.json.
        let with_prompts = live_client(Some(128))
            .with_hf_home(cache.path())
            .with_prefixes_from_hf("query", "document")
            .await
            .expect("prefixes from HF");
        assert!(
            format!("{with_prompts:?}").contains("Instruct: Given a web search query"),
            "fetched query prompt: {with_prompts:?}"
        );

        // Full chunked-document round trip: a long doc splits into several rows,
        // a doc below `min_tokens` contributes none.
        let long = (0..400)
            .map(|i| format!("Sentence {i} about retrieval and embeddings."))
            .collect::<Vec<_>>()
            .join("\n\n");
        let docs = [long.as_str(), "too short"];
        let opts = ChunkOptions {
            max_tokens: 64,
            overlap_tokens: 8,
            min_tokens: 10,
        };
        let rows = api
            .embed_documents(&docs, &opts)
            .await
            .expect("embed_documents");

        assert!(rows.len() > 1, "long document must chunk into several rows");
        let pairs: Vec<(usize, usize)> = rows.iter().map(|r| (r.doc_ix, r.chunk_ix)).collect();
        let mut sorted = pairs.clone();
        sorted.sort_unstable();
        assert_eq!(pairs, sorted, "rows ordered by (doc_ix, chunk_ix)");
        for r in &rows {
            assert_eq!(
                r.doc_ix, 0,
                "short doc below min_tokens contributes nothing"
            );
            assert_eq!(r.embedding.len(), 128, "MRL dims on document chunks");
            assert_eq!(
                long.get(r.chunk.byte_start..r.chunk.byte_end),
                Some(r.chunk.text.as_str()),
                "chunk text must match its source span"
            );
        }
        assert_eq!(
            rows.last().expect("rows").chunk.byte_end,
            long.len(),
            "chunking must cover the tail"
        );
    }
}
