# patterns

Personal pattern library: reusable building blocks shared across my projects
via a pinned git dependency. One crate, module per pattern.

Modules:
- `llm_cli` — structured extraction from text via a local LLM CLI, with
  one-repair-retry semantics. Prompt templating stays in consumers.
- `lance_store` — reserved.

Usage:
`patterns = { git = "https://github.com/jozefRudy/patterns", rev = "<sha>" }`

## `llm_cli` usage

Shell out to any local LLM CLI that takes the rendered prompt as its **last
argument** and prints JSON to stdout (```json fences stripped). Configure with
a command string, e.g. from a `[llm] bin` config entry:

```rust
let extractor = LlmExtractor::<JobAd>::from_bin(
    "pi --print --no-session --no-tools --no-extensions --mode text --thinking off --model deepseek/deepseek-v4-flash",
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

**2. Prompt template** — hand-written askama file in the consumer's
`templates/` dir, registered via `define_prompts!`. Templates are strongly
typed: exactly `{{ schema }}`, `{{ text }}`, `{{ prompt_context }}` available,
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
let job: JobAd = extractor
    .with_prompt_context("prefer EU-based roles".into())
    .extract(&posting_text)
    .await?;
// JobAd { title: "Senior Rust Developer", remote: Some(true),
//         salary: Some("EUR 80k-100k".into()), tags: ["rust", "backend"] }
```

Extraction is strongly typed: the LLM's raw JSON is deserialized straight
into `T` via serde — unknown/missing/wrong-typed fields fail parsing and
trigger the repair retry below. You never touch untyped JSON yourself.

License: MIT.
