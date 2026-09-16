# patterns

Personal pattern library: reusable building blocks shared across my projects
via a pinned git dependency. One crate, module per pattern.

Modules (feature-gated; `default = ["llm_cli", "embed"]`):
- `llm_cli` (feature `llm_cli`) — structured extraction from text via a local
  LLM CLI, with one-repair-retry semantics, via `SharedLlm`: a cloneable
  handle with a process-wide concurrency cap. Prompt templating stays in
  consumers.
- `embed` (feature `embed`) — text embeddings via fastembed (ONNX, CPU,
  in-process). Model, thread count and query/document prefixes configured at
  load; query/document methods apply prefixes automatically.
- `lance_store` — reserved.

Usage (consumers pin exactly what they use — don't rely on defaults):

```toml
patterns = { git = "https://github.com/jozefRudy/patterns", rev = "<sha>", default-features = false, features = ["embed"] }
```

Deps with versions that matter are pinned exactly in this crate
(`fastembed =6.1.0`, `ort =2.0.0-rc.13` — which transitively pins a
sha256-verified ONNX Runtime binary) and re-exported (`patterns::fastembed`,
`patterns::ort`); consumers never declare them separately.

## `embed` usage

```rust
use patterns::embed::{Embedder, LoadOptions, Prefixes};
use patterns::fastembed::EmbeddingModel;

// model + optional query/document prefixes (empty for symmetric models)
let embedder = Embedder::load(
    LoadOptions::new(EmbeddingModel::MxbaiEmbedLargeV1Q)
        .with_prefixes(&Prefixes { query: "search_query: ".into(), document: "search_document: ".into() })
        .with_intra_threads(4),      // leave cores for other tasks
    &cache_dir,
).await?;

// queries: one vector, never chunked
let q = embedder.embed_query("rust jobs").await?;

// documents: always chunked — the only document API
let opts = embedder.default_chunk_options();   // ctx − special tokens, 64 overlap, 5 min
for c in embedder.embed_document_chunks(&long_post, &opts).await? {
    // c.chunk.text / c.chunk.tokens / c.chunk.byte_start..byte_end / c.embedding
}

// batches: one model call across every document's chunks, flat row batch
let rows = embedder.embed_batch_document_chunks(&texts, &opts).await?;
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

## `llm_cli` usage

Shell out to any local LLM CLI that takes the rendered prompt as its **last
argument** and prints JSON to stdout (markdown `json` code fences stripped). Entry point is
`SharedLlm`: a cloneable handle holding the command string plus a process-wide
concurrency cap (`Arc<Semaphore>` — `Clone` + `Send` + `Sync`, no locks). The
cap only works if all callers share one handle. Limits are app policy, passed
at construction (env var reading stays in the consumer):

```rust
let llm = SharedLlm::new(
    "pi --print --no-session --no-tools --no-extensions --mode text --thinking off --model deepseek/deepseek-v4-flash".into(),
    SharedLimits {
        max_concurrent_calls: 2,
        max_text_len: 4000,
        call_timeout: Duration::from_secs(30),
    },
);
```

Consumer defines three things:

**1. Output struct** — `#[schemars(description = ...)]` per field; these land
in the JSON schema rendered into the prompt and steer the LLM. Describe
meaning, nullability rules, format examples. `Option<T>` fields render as
nullable in the schema and serde accepts `null` *or* omission.

```rust
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
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

**2. Prompt template** — `render_prompt` is free-form; consumers either hand
roll it (simple `format!` compositions) or use strongly typed askama
templates registered via `define_prompts!`. Templates are strongly typed:
exactly `{{ schema }}`, `{{ text }}`, `{{ prompt_context }}` available,
compile error otherwise.

```rust
patterns::define_prompts!((JobAdExtract, "prompts/job_ad.md"));
```

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

**3. `Extractable` impl**

```rust
impl patterns::llm_cli::Extractable for JobAd {
    const HEALTHCHECK_TEXT: &'static str = "Senior Rust dev, fully remote, EUR 80k-100k";

    fn render_prompt(schema: &str, text: &str, prompt_context: &str) -> anyhow::Result<String> {
        PromptKind::JobAdExtract.render_prompt(schema, text, prompt_context)
    }

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

License: MIT.
