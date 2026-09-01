# Shared `patterns` library — plan

Public git-dep pattern workspace. Goal: projects keep only domain code
(schemas, prompts, command enums, SQL); machinery lives in one maintained
place with fixes propagating via rev bumps.

## Repo layout

```
patterns/                    (public repo, single crate to start, MIT/Apache)
  src/
    llm_cli.rs               # LLM extraction via CLI subprocess
    lance_store.rs           # writer actor + query helpers
    search.rs                # hybrid FTS+ANN+RRF helpers (after v2 lands)
    browser_cdp.rs           # later
  LICENSE  README.md
```

Start with one crate, one module per pattern. Split into `crates/*`
workspace later if: lance deps become a burden for llm-only consumers
(heavy arrow/C++ compile), versioning needs diverge, or a module grows
independent consumers. Keep module boundaries strict from day one (no
cross-imports) so the split is a file move, not a refactor.

Consumers:

```toml
patterns = { git = "https://github.com/jozefRudy/patterns", rev = "<tag>" }
```

Optional lance deps behind a feature flag (`features = ["lance"]`) so
llm-only consumers skip the heavy compile.

Pin revs deliberately; tag repo on breaking bumps (esp. lance version bumps).
During dev use `[patch]` to a local checkout.

CI (patterns repo): `cargo build && cargo clippy --all-targets && cargo test && cargo fmt`
on push (plus `--all-features` build once lance lands).

## Module 1: `llm_cli` — extract NOW

Two consumers exist: job_search (implemented) + reddit_v2 (about to adopt).

### Source

- `job_search/src/extractors/llm.rs` (incl. new retry logic)
- `job_search/src/extractors/prompts/repair.md`

### Move into crate (context-free machinery)

- `Extractable` trait: `PROMPT`, `HEALTHCHECK_TEXT`, `verify()`
- `LlmExtractor<T>`: `from_bin`, `with_prompt_context`, `extract`, `verify`,
  `run`, `run_and_parse` + one-retry with serde error
- `ParseFailure`, `build_repair_prompt` (uses `REPAIR_PROMPT` const —
  keep as plain template string with `{original_prompt}`/`{previous_output}`/`{error}`
  placeholders)
- `truncate`, `strip_json_fences`, `MAX_TEXT_LEN`, `DEFAULT_TIMEOUT`
- All unit tests

### Stay in projects (domain)

- `PromptKind` + `define_prompts!` macro + askama templates
  (`hackernews_fields.md`, `reddit_rust_fields.md`) — project-local
- `[llm] bin` config (jobsearch.toml / reddit_v2 config)
- concrete `Extractable` impls

### Migration steps

1. Create repo + crate skeleton; move machinery + tests into `src/llm_cli.rs`.
2. job_search: replace local `llm.rs` with git dep; keep macro/templates;
   `cargo build && cargo clippy --all-targets && cargo test && cargo fmt`.
3. reddit_v2: adopt via git dep instead of copying; define own
   `PromptKind` templates as needed.

## Module 2: `lance_store` — extract NOW (two consumers)

Two real implementations to diff:

- `job_search/src/embeddings_store.rs`, `job_search/src/vector_db/reddit_store.rs`
- `reddit_v2/src/store/{mod.rs,writer.rs,queries.rs,entities.rs}`

### Move into crate (machinery)

- `Schema` trait: `const DATASET`, arrow schema, startup schema assert
  (invariant panic per AGENTS.md rules)
- `Command` trait: `apply(&mut Dataset) -> Result<Out>`
- writer actor: message-passing loop, oneshot replies, backpressure
  (contract: no `Arc<Mutex>`, no locks across `.await` — document in README)
- DataFusion read-side: `SessionContext` registration helper for lance datasets
- Re-exports: `pub use arrow; pub use lance;` so consumers never declare
  lance separately (single version, enforced)

### Stay in projects (domain)

- entities, concrete command enums, SQL strings, config paths, retention

### Process

1. Read both implementations; diff machinery vs domain.
2. Design minimal traits from the actual diff (if generic version ends up
   longer than both concrete versions combined — don't extract).
3. Implement crate; migrate both projects; delete local copies; run
   validation in each.

## Queued module: `search` — after reddit_v2 implements binary pipeline

Both projects share the hybrid-search skeleton but with different legs:

- job_search `embeddings_store.rs`: float cosine ANN leg + FTS `MatchQuery`
  leg + `rrf_merge` (pure, k=60, tested)
- reddit_v2 (docs/ranking.md): hamming ANN on binary embedding + oversample
  (10–20x) + weighted RRF + float-column exact cosine rerank + bm25 combine

Shared sliver (small): `rrf_merge`, scanner/collect plumbing, leg builders
parameterized by (column, metric, oversample). Do NOT extract from
job_search alone — v2's binary+rerank variant defines the real parameter
surface; extracting early overfits to float-cosine.

Trigger: reddit_v2 ranking implemented and verified (regression test:
"rust" ranks programming above Rust Belt). Then diff against job_search
and extract shapes like:

```rust
fn rrf_merge(legs: &[&[i64]], weights: &[f32], k: f32) -> Vec<(i64, f32)>;
async fn fts_leg(ds: &Dataset, query: &str, limit: usize) -> Result<Vec<i64>>;
async fn ann_leg(ds: &Dataset, col: &str, metric: DistanceType, limit: usize) -> Result<Vec<i64>>;
async fn cosine_rerank(ds: &Dataset, ids: &[i64], float_col: &str, query: &[f32])
    -> Result<Vec<(i64, f32)>>;
```

(final signatures from the diff). Stays in reddit_v2: tier weights,
score combination, freshness/diversity, pre-filters — domain.
Optionally backport binary ANN to job_search once proven.

## Deferred crate: `browser-cdp` — one consumer so far

Only job_search drives a local browser from Rust (`src/browser.rs`,
chromiumoxide/CDP on :9222). reddit_v2 browser tests use pi's JS-evaluate
tooling, not Rust — so no second consumer yet.

Future extract (trait-shaped already, easy when warranted):

- `BrowserExt` trait: tab management, `get_page_targets`,
  `close_pages_except`, `set_cookie`, connect/retry to CDP
- stays in job_search: `DEFAULT_INIT_URLS`, `REQUIRED_HOSTS` (domain)

Extract when a second Rust project drives a local browser; diff then.

## Extraction rules (for future patterns)

1. Context-free patterns: one consumer suffices if API is stable.
2. Context-coupled patterns: wait for second real consumer; extract from diff.
3. One crate, one module per pattern; split into `crates/*` only on
   evidence (compile weight, diverging versions); never a catch-all "utils" module.
4. Public repo + git deps with rev pins; crates.io publish only if APIs
   stabilize and external use matters.
5. Constraints/rules travel with the crate README (invariant panics,
   message-passing ownership, no locks across await).
