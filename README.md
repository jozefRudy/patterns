# patterns

Personal pattern library: reusable building blocks shared across my projects
via a pinned git dependency. One consumer-facing crate, module per pattern;
an internal `patterns-macros` proc-macro crate provides the `#[derive(Extractable)]`
and `#[derive(SystemOne)]` derives (consumers still depend on `patterns` alone).

Modules (feature-gated; `default = ["llm_cli", "embed", "embed_api", "language", "systemone"]`):
- `llm_cli` (feature `llm_cli`) — structured extraction from text via a local
  LLM CLI, with one-repair-retry semantics, via `SharedLlm`: a cloneable
  handle with a process-wide concurrency cap. The domain is one annotated
  struct (`#[derive(Extractable)]` + `#[extract(template = "…", healthcheck = "…")]`);
  template files stay in consumers.
- `embed` (feature `embed`) — text embeddings via fastembed (ONNX, CPU,
  in-process). Model, thread count and query/document prefixes configured at
  load; query/document methods apply prefixes automatically.
- `language` (feature `language`) — English text detection via lingua.
  Process-wide singleton detector (`OnceLock<Arc<...>>` — memory optimization:
  preloaded language models load exactly once, shared by all instances; no
  locks since the detector is immutable after build), `detect(&str)` off the
  blocking pool.
- `systemone` (feature `systemone`) — bounded HTTP client for the TypeSafe
  SystemOne (Jev) protocol (`POST {base_url}/v1/systemone`), via
  `SharedSystemOne`: a cloneable handle with a process-wide concurrency cap, a
  per-call timeout and a configurable `RetryPolicy` (default 3 attempts,
  250ms→2s) for transient failures. Base URL/API key/model come from the caller
  (never read from env); backends differ only by base URL. Questions are
  independent; `#[derive(SystemOne)]` generates the typed question set and,
  with `healthcheck = "…"`, the batch-gate `Evaluatable` impl.
- `embed_api` (feature `embed_api`) — bounded HTTP client for any
  OpenAI-compatible `/embeddings` endpoint (DeepInfra, OpenAI, Together,
  SiliconFlow), via `EmbeddingApi`: a cloneable handle with a process-wide
  concurrency cap, batching and retry. Base URL/API key/model/dims come from
  the caller (never read from env).
- `lance_store` — reserved.

Usage (consumers pin exactly what they use — don't rely on defaults):

```toml
patterns = { git = "https://github.com/jozefRudy/patterns", rev = "<sha>", default-features = false, features = ["embed"] }
```

Deps with versions that matter are pinned exactly in this crate
(`fastembed =6.1.0`, `ort =2.0.0-rc.13` — which transitively pins a
sha256-verified ONNX Runtime binary) and re-exported (`patterns::fastembed`,
`patterns::ort`, `patterns::askama`); consumers never declare them separately.
`patterns-macros` is an internal workspace member (path dependency) re-exported
as `patterns::Extractable` / `patterns::SystemOne`.

## `embed` usage

```rust
use patterns::embed::{Embedder, LoadOptions, Prefixes};
use patterns::fastembed::EmbeddingModel;

// model + optional query/document prefixes (empty for symmetric models)
let embedder = Embedder::load(
    LoadOptions::new(EmbeddingModel::MxbaiEmbedLargeV1Q)
        .with_prefixes(&Prefixes::new("search_query: ", "search_document: "))
        .with_intra_threads(4),      // leave cores for other tasks
    &cache_dir,
).await?;

// queries: one vector, never chunked
let q = embedder.embed_query("rust jobs").await?;

// documents: always chunked — the only document API, one call per batch
let opts = embedder.default_chunk_options();   // ctx − special tokens, 64 overlap, 5 min
let rows = embedder.embed_batch_document_chunks(&texts, &opts).await?;
// single document: embed_batch_document_chunks(&[text], &opts)  (doc_ix == 0)
for r in &rows {
    // r.doc_ix -> ids[doc_ix]; r.chunk_ix; r.chunk.byte_start..byte_end; r.embedding
}
let seen: std::collections::HashSet<usize> = rows.iter().map(|r| r.doc_ix).collect();
let skipped = (0..texts.len()).filter(|ix| !seen.contains(ix));  // below min_tokens
```

Design:
- **prefixes are model config**, applied by `embed_query` and the chunked
  document methods — no prefix logic at call sites
- **chunking is forced for documents**: tokenizer-exact, boundary-aware
  (paragraph → sentence → whitespace), with overlap and a no-tail-loss
  guarantee. Long-tail text is chunked, never silently truncated
- **`opts` is explicit** (transparent, per-call-site overridable);
  `default_chunk_options()` derives from the model's context window,
  `ChunkOptions::new(max_tokens)` is the manual escape hatch. Below
  `min_tokens` a document contributes no rows — the caller owns whether that
  becomes a status row, a query filter, or nothing
- **row batch**: `EmbeddedChunk { doc_ix, chunk_ix, chunk, embedding }`, ordered
  by `(doc_ix, chunk_ix)`; `doc_ix` is the index into the slice you passed, so
  ids stay positional and skipped documents are `(0..texts.len()) − seen`
- **inspection before embedding**: `token_count(text)` (full count, truncation
  off) and `model_max_tokens()` for callers that gate or size things themselves
- **storage helpers**: `packed_len`, `binarize` (1 bit/dim, LSB-first,
  hamming-ready) and `quantize_u8` (8 bits/dim). One packing contract shared by
  index build and query — see
  https://emschwartz.me/binary-vector-embeddings-are-so-cool/ (binary vectors:
  32× smaller, ~25× faster lookups, ~96% of float retrieval quality)
- inference runs on the blocking pool (`spawn_blocking`); the mutex is never
  held across `.await`. One handle serializes its callers — for
  latency-sensitive serving alongside bulk embedding, load **two instances**
  (query + bulk). That's capacity policy, so it stays a consumer decision
- `Embedder::fake(dim)` returns deterministic hash vectors for tests of
  embedding-adjacent logic (stores, ranking) without a model download


## `embed_api` usage

Call any OpenAI-compatible `/embeddings` endpoint. The caller owns config
(base URL, key, model, dims) — nothing is read from the environment.

```rust
use patterns::embed_api::EmbeddingApi;
use patterns::limits::ConcurrencyLimits;
use patterns::prefixes::Prefixes;

let prefixes = Prefixes::new(
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
    "", // documents are embedded raw
);

let api = EmbeddingApi::new(
    "https://api.deepinfra.com/v1/openai",
    api_key,
    "Qwen/Qwen3-Embedding-8B",
    ConcurrencyLimits::default(), // process-wide concurrency cap + per-call timeout
)?
.with_mrl_truncation(512)           // optional; default = model native dims
.with_prefixes(&prefixes)           // optional; model config, shared with `embed`
.with_max_tokens(32_768)            // model ctx window (card): https://huggingface.co/Qwen/Qwen3-Embedding-8B
                                    // — read "Context Length" from the card; for chunking
.with_tokenizer("Qwen/Qwen3-Embedding-8B")  // for chunking + `token_count`
.with_hf_home(cache_dir)            // required when `with_tokenizer` is an HF repo id
.with_service_tier("flex")          // optional; omitted when unset
.with_max_batch_size(512);         // optional; default 256

// query: one vector, never chunked
let q = api.embed_query("first query").await?;

// documents: always chunked to the declared window, one batched call
let opts = api.default_chunk_options()?;   // ctx − special tokens, 64 overlap, 5 min
let rows = api.embed_documents(&["first text", "second text"], &opts).await?;
// rows: EmbeddedChunk { doc_ix, chunk_ix, chunk, embedding }, ordered by (doc_ix, chunk_ix)
```

Notes:
- `.with_mrl_truncation(n)` requests MRL truncation to `n` dims (`n >= 32`);
  without it the vector uses the model's native output dimensions. The
  `dimensions` field is omitted when unset. Truncated vectors are **not**
  renormalized — normalize downstream if you need unit length.
- Inputs are split into batches and issued with bounded concurrency; results
  come back in input order. Retries cover 429/5xx/timeouts; other 4xx fail fast.
- `embed_query` embeds **one** query (never chunked); `embed_documents` chunks
  each document to `opts`, applies the document prefix per chunk, and embeds all
  chunks across the batch in one call. Same document-always-chunked rule as the
  in-process `embed` backend.
- `.with_max_tokens(n)` declares the model's context window (read it from the
  model card: https://huggingface.co/Qwen/Qwen3-Embedding-8B → "Context Length:
  32k" — `config.json`/`tokenizer_config.json` fields like
  `max_position_embeddings`/`model_max_length` can exceed the real window).
  `.default_chunk_options()` derives `ChunkOptions` from it and errors until set.
- `.with_tokenizer(source)` enables `token_count(text).await` and is required by
  `embed_documents`. `source` is an HF repo id (only
  `tokenizer.json` is fetched; requires `.with_hf_home(path)` — cache
  `{path}/hub`, token file `{path}/token`) or a local `tokenizer.json` path
  (no `hf_home` needed). The client never reads `HF_HOME` from the environment.
  Lazy-loaded once and shared across clones; without it `token_count` errors.
- `.with_prefixes(&Prefixes)` sets the model's fixed query/document prefixes;
  `embed_query` prepends `query`, `embed_documents` prepends `document` to each
  chunk. Defaults to none (symmetric models). `Prefixes`
  lives in `patterns::prefixes` (re-exported as `patterns::embed::Prefixes`).
  For asymmetric open models this replaces a provider's native `input_type` —
  e.g. Qwen3-Embedding's `Instruct: <task>\nQuery:` query prefix.
- `.with_prefixes_from_hf(query_key, document_key).await?` fetches the two
  prompts from the model's `config_sentence_transformers.json` instead (requires
  `.with_hf_home`). Keys vary by model (`query`/`document`,
  `query`/`passage`, `retrieval.query`/`retrieval.passage`); it fails when the
  file, `prompts` map, or a key is missing — not every repo ships this file.
- Only the OpenAI-compatible schema is supported (DeepInfra, OpenAI, Together,
  SiliconFlow). Native Cohere/Voyage/Jina APIs are out of scope.

## `language` usage

```rust
use patterns::language::LanguageService;

let svc = LanguageService::new();   // all instances share one singleton detector
let is_english: anyhow::Result<bool> = svc.detect("Senior Rust developer, remote").await;
```

English with confidence > 0.5 counts as English. Examples that detect as
English: "the", "programming", "I love programming", long advert sentences
("Machine learning is a subset of artificial intelligence..."), and typos
don't derail it ("I love programing", "artificail inteligence"). Candidate languages are
narrowed to five (en, fr, de, es, pl) — a small, distinctive set keeps
detection confident on almost any input while avoiding the false-English
results of full 75-language mode; models preloaded.

Minimum text: single meaningful words already work ("the", "programming" →
English; "Bonjour", "Cześć" → not), typos don't derail detection, but a
sentence or more is the reliable zone — below ~2-3 words treat the result as
weak.

Languages verified in tests (correctly NOT detected as English): Polish,
Spanish, French, German, Italian, Dutch, Swedish, Russian, Japanese, Korean,
Hindi, Thai, Arabic, Hebrew, Turkish, Zulu.

## `llm_cli` usage

Shell out to any local LLM CLI that takes the rendered prompt as its **last
argument** and prints JSON to stdout (markdown `json` code fences stripped). Entry point is
`SharedLlm`: a cloneable handle holding the command string plus a process-wide
concurrency cap (`Arc<Semaphore>` — `Clone` + `Send` + `Sync`, no locks). The
cap only works if all callers share one handle. Limits are app policy, passed
at construction (env var reading stays in the consumer):

```rust
use patterns::limits::ConcurrencyLimits;

let llm = SharedLlm::new(
    "pi",
    [
        "--print", "--no-session", "--no-tools", "--no-extensions",
        "--mode", "text", "--thinking", "off",
        "--model", "deepseek/deepseek-v4-flash",
    ],
    ConcurrencyLimits::default(),   // max_concurrent_calls + call_timeout
)
.with_max_text_len(4000);           // llm-local, defaults to 4000
```

The consumer defines the domain in one annotated struct: `#[derive(JsonSchema)]`
+ `#[schemars(description)]` for the output shape and per-field guidance, plus
`#[extract(template = "...", healthcheck = "...")]` for the prompt template and
healthcheck fixture.

```rust
use patterns::Extractable;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema, Extractable)]
#[extract(
    template = "prompts/job_ad.md",
    healthcheck = "Senior Rust dev, fully remote, EUR 80k-100k",
)]
struct JobAd {
    #[schemars(description = "job title or role; if multiple listed, join them with ' + '")]
    title: String,
    #[schemars(description = "true if fully remote, false if location-restricted ('US only', 'onsite'); null if not mentioned")]
    remote: Option<bool>,
    #[schemars(description = "raw compensation snippet, e.g. '$150k-$175k' or 'EUR 80k-100k'")]
    salary: Option<String>,
    #[schemars(description = "tech/stack keywords")]
    tags: Vec<String>,
}
```

- `#[schemars(description = ...)]` per field — lands in the JSON schema rendered
  into the prompt and steers the LLM. Describe meaning, nullability rules,
  format examples. `Option<T>` fields render as nullable and serde accepts
  `null` *or* omission.
- `#[extract(template = "...")]` — the strongly typed askama template; exactly
  `{{ schema }}`, `{{ text }}`, `{{ prompt_context }}` are available (anything
  else is a compile error). `render_prompt` is generated from it.
- `#[extract(healthcheck = "...")]` — the fixture; generates
  `HEALTHCHECK_TEXT`.

`templates/prompts/job_ad.md`:

```md
You extract structured data from job postings.
Return ONLY valid JSON with no markdown and no explanation.

JSON schema:
{{ schema }}

Additional context:
{{ prompt_context }}

Post:
{{ text }}
```

The only hand-written piece is the semantic `verify` (picked up by name):

```rust
impl JobAd {
    // semantic smoke test on the known healthcheck text — proves the model
    // understands the task, not just that it emits schema-valid JSON
    fn verify(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.title.to_lowercase().contains("rust"), "bad title");
        anyhow::ensure!(self.remote == Some(true), "remote not detected");
        Ok(())
    }
}
```

Then:

```rust
let job: JobAd = llm
    .extract(&posting_text, "prefer EU-based roles".into())
    .await?;
// JobAd { title: "Senior Rust Developer", remote: Some(true),
//         salary: Some("EUR 80k-100k".into()), tags: ["rust", "backend"] }
```

Extraction is strongly typed: the LLM's raw JSON is deserialized straight
into `T` via serde — unknown/missing/wrong-typed fields fail parsing and
trigger the repair retry below. You never touch untyped JSON yourself.

## Healthcheck as a batch gate

`SharedLlm::verify::<T>()` runs the **whole pipeline** (prompt → subprocess →
parse) on `T::HEALTHCHECK_TEXT` — a fixture with a known-correct answer —
and asserts `T::verify()` on the result. It validates the *system*, not the
data: prompt wording regressions, model swaps/fallbacks, JSON-mode breakage,
parse drift, auth degradation.

Call it **once per batch, before the pass** — never per item (it costs one
full LLM round-trip and proves nothing new per row). Standard consumer
pattern:

```rust
// at batch start; on failure skip the batch and retry next tick
llm.verify::<JobAd>().await?;
for item in items {
    let out: JobAd = llm.extract(&item.text, ctx).await?;
    ...
}
```

A failing healthcheck is a loud tripwire *before* a batch burns — a broken
pipeline otherwise surfaces as silent parse failures (or worse, silently
wrong rows) across every item in the run.

## `systemone` usage

One bounded HTTP client for the TypeSafe SystemOne (Jev) protocol
(`POST {base_url}/v1/systemone`). Backends differ only by base URL — direct
(`https://api.typesafe.ai`) or OpenRouter (`https://openrouter.ai/api`); the
path is fixed internally. Entry point is `SharedSystemOne`: a cloneable handle
holding the base URL, API key, model and `ConcurrencyLimits`, with a
process-wide concurrency cap and a per-call timeout. Transient failures
(429/5xx, timeouts, transport errors) are retried per a configurable
`RetryPolicy` (default 3 attempts, 250ms exponential backoff capped at 2s).
The caller supplies base URL/API key/model — the client
never reads the environment.

Same shape as `llm_cli`: the consumer owns the domain, the questions and the
validation. The difference is the transport — instead of a prompt template and
a CLI, SystemOne takes a `state` plus a typed question set and returns one
answer per question id.

`#[derive(SystemOne)]` makes the annotated struct the source of truth — the
same role `#[derive(JsonSchema)]` + `#[schemars(description)]` play for
`llm_cli`. Field attributes declare each question; `#[systemone(template =
"...")]` binds the input template (askama, `{{ text }}` + `{{ prompt_context }}`
only); the derive emits `questions()` + `render_state`, so the questions and
the template cannot drift apart. `#[systemone(healthcheck = "...")]`
additionally emits the batch-gate `Evaluatable` impl.

```rust
use patterns::SystemOne;
use patterns::limits::ConcurrencyLimits;
use patterns::systemone::{Choice, Noul, Questions, Score, SharedSystemOne};

let client = SharedSystemOne::new(
    "https://api.typesafe.ai".into(),
    api_key,
    "jev-latest".into(),
    ConcurrencyLimits::default(),
);

// 1. domain: one annotated struct -> `Questions` impl + `render_state`.
#[derive(SystemOne, Debug, serde::Deserialize)]
#[systemone(
    template = "job_input.md",
    healthcheck = "Senior Rust dev, fully remote, EUR 80k-100k",
)]
struct JobAssessment {
    #[noul("Is the role fully remote, with no onsite or region restriction? Judge only from the job posting in the input.")]
    is_remote: Noul,

    #[choice("Which seniority level does the posting target?",
        junior  = "0-2 years, mentored work",
        mid     = "3-5 years, works independently",
        senior  = "6+ years, leads work and reviews others",
        staff   = "org-wide technical leadership",
        unknown = "not stated or genuinely ambiguous")]
    seniority: Choice,

    #[score("How strong is the match for a senior Rust/backend engineer?",
        "Poor", "Weak", "Fair", "Strong", "Excellent")]
    match_score: Score,

    #[noul("Does the posting contain a red flag (unpaid trial, vague comp, 'rockstar/ninja' culture)?")]
    red_flags: Noul,
}

// 2. evaluate: `evaluate_text` renders the state from the template declared
//    above (`{{ text }}` + `{{ prompt_context }}`) and sends it. Use
//    `evaluate` directly to pass an already-built state.
let assessment = client.evaluate_text::<JobAssessment>(&posting_text, &context).await?;
// assessment.is_remote.noul       -> P(remote), e.g. 0.94
// assessment.seniority.choice     -> "senior", .confidence
// assessment.match_score.expected() -> 3.4; .argmax_level() -> 4
// assessment.red_flags.noul       -> P(red flag)
```

`job_input.md` (resolved against your own `askama.toml` dirs):

```md
Job posting:
{{ text }}

Additional context:
{{ prompt_context }}
```

Questions are **independent and evaluated in parallel** — none sees another's
answer, so don't chain them. Each question carries free-form `instructions`
(any `serde_json::Value`: string, object or array) plus structured `criteria`
(options/levels) — criteria go over the wire as data, not baked into the
instructions text.

Answers map straight onto the response `answers` object, keyed by id:
- `noul` → `Noul { noul }` (probability of yes)
- `choice` → `Choice { choice, confidence, probabilities }` (argmax +
  confidence — **not** `P(option)`)
- `score` → `Score { confidence, probabilities }`; `expected()` gives the
  probability-weighted position, `argmax_level()` the highest-probability
  level index. The wire `score`/`legend` are derived and not stored.

Unknown response fields (`model`, `usage`, `legend`, and `score`'s derived
`score` value) are ignored; on parse failure the raw `answers` JSON is
included in the error context.

### Healthcheck as a batch gate

Same pattern as `SharedLlm::verify::<T>()`. The `healthcheck = "..."`
container attribute generates the `Evaluatable` impl; you write only the
semantic `verify` as an inherent method (picked up by name). The gate renders
the fixture through the **real** template and reuses the **real** questions —
call it **once per batch, before the pass** (never per item):

```rust
impl JobAssessment {
    // semantic smoke test on the known fixture — proves the model understands
    // the task, not just that it emits parseable JSON
    fn verify(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.is_remote.noul > 0.5, "remote not detected");
        anyhow::ensure!(self.seniority.choice == "senior", "wrong seniority");
        Ok(())
    }
}

client.verify::<JobAssessment>().await?;   // skip the batch on failure
```

License: MIT.
