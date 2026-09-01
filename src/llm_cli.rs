//! Generic LLM extraction via a local CLI binary.
//!
//! The CLI receives the fully rendered prompt as its last argument and must
//! print JSON to stdout. Extraction targets implement [`Extractable`]; prompt
//! templating (askama, prompt kinds) stays in the consuming project.
//!
//! Behavior: exactly one repair retry per [`LlmExtractor::extract`]; the
//! timeout applies per LLM call; empty or `"NONE"` responses bail.

use anyhow::{Context, Result};
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use std::marker::PhantomData;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

/// Per-call timeout for the LLM CLI.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum byte length of text embedded into prompts.
pub const MAX_TEXT_LEN: usize = 4000;

/// A type that can be extracted from LLM output.
///
/// Implementations render their own prompt (e.g. via an askama template and a
/// project-local prompt-kind enum) so this crate stays template-agnostic.
pub trait Extractable: JsonSchema + for<'de> Deserialize<'de> {
    /// Sample text used to verify the LLM produces valid structured output.
    const HEALTHCHECK_TEXT: &'static str;

    /// Render the extraction prompt from the JSON schema, the (already
    /// truncated) input text, and optional dynamic prompt context.
    fn render_prompt(schema: &str, text: &str, prompt_context: &str) -> Result<String>;

    /// Validate that a healthcheck extraction succeeded.
    fn verify(&self) -> Result<()>;
}

/// Generic LLM extractor that calls a local CLI.
///
/// Configure with a command string via [`LlmExtractor::from_bin`], e.g. from a
/// project-local `[llm] bin` config entry.
#[derive(Debug, Clone)]
pub struct LlmExtractor<T: Extractable> {
    bin: String,
    args: Vec<String>,
    prompt_context: String,
    // `fn() -> T`: covariant marker, keeps LlmExtractor Send + Sync regardless of T
    _phantom: PhantomData<fn() -> T>,
}

impl<T: Extractable> LlmExtractor<T> {
    /// Extract structured data from `text`.
    pub async fn extract(&self, text: &str) -> Result<T> {
        let schema = serde_json::to_string_pretty(&schema_for!(T))?;
        let truncated = truncate(text);
        let rendered = T::render_prompt(&schema, &truncated, &self.prompt_context)?;
        self.run_and_parse::<T>(&rendered).await
    }

    /// Attach dynamic context text rendered into the prompt template.
    #[must_use]
    pub fn with_prompt_context(mut self, context: String) -> Self {
        self.prompt_context = context;
        self
    }

    /// Run the healthcheck: extract from [`Extractable::HEALTHCHECK_TEXT`] and
    /// validate via [`Extractable::verify`].
    pub async fn verify(&self) -> Result<()> {
        self.extract(T::HEALTHCHECK_TEXT).await?.verify()
    }

    /// Configure with a command string, e.g. `"llm -m claude-4-sonnet"`.
    #[must_use]
    pub fn from_bin(llm_bin: &str) -> Self {
        let tokens = shell_words::split(llm_bin).unwrap_or_default();
        let (bin, args) = tokens
            .split_first()
            .map(|(h, t)| (h.clone(), t.to_vec()))
            .unwrap_or_default();
        Self {
            bin,
            args,
            prompt_context: String::new(),
            _phantom: PhantomData,
        }
    }

    /// Run the LLM with the rendered prompt and deserialize the response into `R`.
    /// On parse failure, retry once with a repair prompt containing the invalid
    /// output and the serde error.
    async fn run_and_parse<R>(&self, prompt: &str) -> Result<R>
    where
        R: for<'de> Deserialize<'de>,
    {
        match self.try_parse(prompt).await {
            Ok(parsed) => Ok(parsed),
            Err(failure) => {
                let repair = build_repair_prompt(prompt, &failure.output, &failure.error);
                self.try_parse(&repair).await.map_err(|retry| {
                    retry.error.context(format!(
                        "llm parse failed after one retry; first error: {}; first output: {}",
                        failure.error, failure.output
                    ))
                })
            }
        }
    }

    async fn try_parse<R>(&self, prompt: &str) -> Result<R, ParseFailure>
    where
        R: for<'de> Deserialize<'de>,
    {
        let out = self.run(prompt).await.map_err(|e| ParseFailure {
            error: e,
            output: String::new(),
        })?;
        let out = out.unwrap_or_default();
        if out.is_empty() || out.eq_ignore_ascii_case("none") {
            return Err(ParseFailure {
                error: anyhow::anyhow!("llm returned empty or NONE response"),
                output: out,
            });
        }
        let stripped = strip_json_fences(&out);
        serde_json::from_str(&stripped).map_err(|e| ParseFailure {
            error: anyhow::Error::new(e).context(format!("failed to parse LLM JSON: {stripped}")),
            output: out,
        })
    }

    async fn run(&self, prompt: &str) -> Result<Option<String>> {
        let mut cmd = Command::new(&self.bin);
        cmd.args(&self.args);
        cmd.arg(prompt);

        let output = timeout(DEFAULT_TIMEOUT, cmd.output())
            .await
            .context("llm extractor timed out")?
            .context("failed to run llm extractor")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("llm extractor failed: {stderr}");
        }

        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if text.is_empty() || text.eq_ignore_ascii_case("none") {
            Ok(None)
        } else {
            Ok(Some(text))
        }
    }
}

struct ParseFailure {
    error: anyhow::Error,
    output: String,
}

fn truncate(text: &str) -> String {
    let mut s = text.to_string();
    while s.len() > MAX_TEXT_LEN {
        s.pop();
    }
    s
}

const REPAIR_PROMPT: &str = include_str!("prompts/repair.md");

const PLACEHOLDER_ORIGINAL: &str = "{original_prompt}";
const PLACEHOLDER_PREVIOUS: &str = "{previous_output}";
const PLACEHOLDER_ERROR: &str = "{error}";

fn build_repair_prompt(
    original_prompt: &str,
    previous_output: &str,
    error: &anyhow::Error,
) -> String {
    REPAIR_PROMPT
        .replace(PLACEHOLDER_ORIGINAL, original_prompt)
        .replace(PLACEHOLDER_PREVIOUS, &truncate(previous_output))
        .replace(PLACEHOLDER_ERROR, &format!("{error:#}"))
}

fn strip_json_fences(text: &str) -> String {
    let trimmed = text.trim();
    trimmed
        .strip_prefix("```json")
        .and_then(|s| s.trim_end().strip_suffix("```"))
        .map_or_else(|| trimmed.to_string(), |body| body.trim().to_string())
}

#[cfg(test)]
#[derive(Debug, Deserialize, JsonSchema)]
struct Dummy {
    value: String,
}

#[cfg(test)]
impl Extractable for Dummy {
    const HEALTHCHECK_TEXT: &'static str = "hello";

    fn render_prompt(_schema: &str, _text: &str, _ctx: &str) -> Result<String> {
        Ok("prompt".to_string())
    }

    fn verify(&self) -> Result<()> {
        if self.value.is_empty() {
            anyhow::bail!("empty value");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_respects_char_boundaries() {
        let s = "αβγδ".repeat(1000);
        let t = truncate(&s);
        assert!(t.len() <= MAX_TEXT_LEN);
        assert!(t.is_char_boundary(t.len()));
    }

    #[test]
    fn test_truncate_leaves_short_text() {
        let s = "short";
        assert_eq!(truncate(s), s);
    }

    #[test]
    fn test_build_repair_prompt_contains_context() {
        let err = anyhow::anyhow!("expected value at line 1 column 2");
        let prompt = build_repair_prompt("ORIGINAL PROMPT", "not json at all", &err);
        assert!(prompt.contains("ORIGINAL PROMPT"));
        assert!(prompt.contains("not json at all"));
        assert!(prompt.contains("expected value at line 1 column 2"));
        assert!(prompt.contains("Return only the corrected JSON"));
    }

    #[test]
    fn test_build_repair_prompt_truncates_long_output() {
        let err = anyhow::anyhow!("boom");
        let long = "x".repeat(MAX_TEXT_LEN * 2);
        let prompt = build_repair_prompt("p", &long, &err);
        assert!(!prompt.contains(&"x".repeat(MAX_TEXT_LEN + 1)));
    }

    #[test]
    fn test_strip_json_fences_removes_fences() {
        let raw = "```json\n{\"is_job_ad\": true}\n```";
        assert_eq!(strip_json_fences(raw), "{\"is_job_ad\": true}");
    }

    #[test]
    fn test_strip_json_fences_leaves_plain_json() {
        let raw = "{\"is_job_ad\": true}";
        assert_eq!(strip_json_fences(raw), "{\"is_job_ad\": true}");
    }

    #[test]
    fn test_from_bin_parses_command_string() {
        let e = LlmExtractor::<Dummy>::from_bin("llm -m sonnet --flag 'quoted arg'");
        assert_eq!(e.bin, "llm");
        assert_eq!(e.args, vec!["-m", "sonnet", "--flag", "quoted arg"]);
    }

    #[test]
    fn test_from_bin_empty_string() {
        let e = LlmExtractor::<Dummy>::from_bin("");
        assert_eq!(e.bin, "");
        assert!(e.args.is_empty(), "args: {:?}", e.args);
    }
}

/// End-to-end tests against a fake CLI implemented as a shell script.
/// The prompt is passed as the last argument; scripts ignore it.
#[cfg(test)]
mod cli_tests {
    use super::*;

    fn fake_cli(dir: &tempfile::TempDir, body: &str) -> String {
        let path = dir.path().join("fake.sh");
        std::fs::write(&path, body).expect("write fake cli script");
        format!("sh {}", path.display())
    }

    /// Script that emits invalid JSON on first invocation, valid JSON after,
    /// counting invocations in a sibling file.
    const FLAKY_SCRIPT: &str = r#"
cfile="$(dirname "$0")/count"
c=$(cat "$cfile" 2>/dev/null || echo 0)
echo $((c + 1)) > "$cfile"
if [ "$c" -eq 0 ]; then echo 'not json'; else echo '{"value":"fixed"}'; fi
"#;

    fn call_count(dir: &tempfile::TempDir) -> usize {
        std::fs::read_to_string(dir.path().join("count"))
            .expect("read count")
            .trim()
            .parse()
            .expect("count is a number")
    }

    #[tokio::test]
    async fn test_extract_success() {
        let dir = tempfile::tempdir().expect("tempdir");
        let e = LlmExtractor::<Dummy>::from_bin(&fake_cli(&dir, "echo '{\"value\":\"ok\"}'"));
        let d = e.extract("text").await.expect("extract");
        assert_eq!(d.value, "ok");
    }

    #[tokio::test]
    async fn test_extract_strips_fences() {
        let dir = tempfile::tempdir().expect("tempdir");
        let e = LlmExtractor::<Dummy>::from_bin(&fake_cli(
            &dir,
            "printf '```json\n{\"value\":\"fenced\"}\n```\n'",
        ));
        let d = e.extract("text").await.expect("extract");
        assert_eq!(d.value, "fenced");
    }

    #[tokio::test]
    async fn test_extract_repairs_once_and_succeeds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let e = LlmExtractor::<Dummy>::from_bin(&fake_cli(&dir, FLAKY_SCRIPT));
        let d = e.extract("text").await.expect("extract after repair");
        assert_eq!(d.value, "fixed");
        assert_eq!(call_count(&dir), 2, "exactly one retry");
    }

    #[tokio::test]
    async fn test_extract_fails_after_one_retry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let e = LlmExtractor::<Dummy>::from_bin(&fake_cli(&dir, "echo 'not json'"));
        let err = e.extract("text").await.expect_err("must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains("after one retry"), "msg: {msg}");
        assert!(msg.contains("not json"), "msg: {msg}");
    }

    #[tokio::test]
    async fn test_extract_none_response_bails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let e = LlmExtractor::<Dummy>::from_bin(&fake_cli(&dir, "echo NONE"));
        let err = e.extract("text").await.expect_err("must fail");
        assert!(format!("{err:#}").contains("empty or NONE"), "err: {err:#}");
    }

    #[tokio::test]
    async fn test_extract_cli_failure_bails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let e = LlmExtractor::<Dummy>::from_bin(&fake_cli(&dir, "echo boom >&2; exit 1"));
        let err = e.extract("text").await.expect_err("must fail");
        assert!(format!("{err:#}").contains("boom"), "err: {err:#}");
    }
}
