//! Generic LLM extraction via a local CLI binary.
//!
//! The CLI receives the fully rendered prompt as its last argument and must
//! print JSON to stdout. Extraction targets implement [`Extractable`]; prompt
//! templating (askama, prompt kinds) stays in the consuming project.
//!
//! Behavior: exactly one repair retry per extraction; the
//! timeout applies per LLM call; empty or `"NONE"` responses bail.

use anyhow::{Context, Result};
use schemars::{JsonSchema, schema_for};
use serde::Deserialize;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;
use tokio::process::Command;
use tokio::sync::Semaphore;
use tokio::time::timeout;

/// Default per-call timeout for the LLM CLI.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default maximum byte length of text embedded into prompts.
pub const DEFAULT_MAX_TEXT_LEN: usize = 4000;

/// Default cap on concurrent LLM CLI calls across a process.
pub const DEFAULT_MAX_CONCURRENT_CALLS: usize = 2;

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
/// Crate-internal engine behind [`SharedLlm`]; not part of the public API.
/// Configure with a command string via [`LlmExtractor::from_bin`].
#[derive(Debug, Clone)]
pub(crate) struct LlmExtractor<T: Extractable> {
    bin: String,
    args: Vec<String>,
    prompt_context: String,
    max_text_len: usize,
    timeout: Duration,
    // `fn() -> T`: covariant marker, keeps LlmExtractor Send + Sync regardless of T
    _phantom: PhantomData<fn() -> T>,
}

impl<T: Extractable> LlmExtractor<T> {
    /// Extract structured data from `text`.
    pub async fn extract(&self, text: &str) -> Result<T> {
        let schema = serde_json::to_string_pretty(&schema_for!(T))?;
        let truncated = truncate(text, self.max_text_len);
        let rendered = T::render_prompt(&schema, truncated, &self.prompt_context)?;
        self.run_and_parse::<T>(&rendered).await
    }

    /// Attach dynamic context text rendered into the prompt template.
    #[must_use]
    pub fn with_prompt_context(mut self, context: String) -> Self {
        self.prompt_context = context;
        self
    }

    /// Override the maximum byte length of text embedded into prompts
    /// (default: [`DEFAULT_MAX_TEXT_LEN`]).
    #[must_use]
    pub const fn with_max_text_len(mut self, max_text_len: usize) -> Self {
        self.max_text_len = max_text_len;
        self
    }

    /// Override the per-call timeout for the LLM CLI (default: [`DEFAULT_TIMEOUT`]).
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
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
            max_text_len: DEFAULT_MAX_TEXT_LEN,
            timeout: DEFAULT_TIMEOUT,
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
                let repair =
                    build_repair_prompt(prompt, &failure.output, &failure.error, self.max_text_len);
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

        let output = timeout(self.timeout, cmd.output())
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

/// Limits for a [`SharedLlm`] gateway.
///
/// App-specific policy (e.g. pi cold start, long wiki pages) is passed in by
/// the consumer; see `DEFAULT_*` constants for the generic defaults.
#[derive(Debug, Clone)]
pub struct SharedLimits {
    /// Cap on concurrent LLM CLI calls across all holders of the handle.
    pub max_concurrent_calls: usize,
    /// Max bytes of task text embedded into a prompt.
    pub max_text_len: usize,
    /// Per-call subprocess timeout.
    pub call_timeout: Duration,
}

impl Default for SharedLimits {
    fn default() -> Self {
        Self {
            max_concurrent_calls: DEFAULT_MAX_CONCURRENT_CALLS,
            max_text_len: DEFAULT_MAX_TEXT_LEN,
            call_timeout: DEFAULT_TIMEOUT,
        }
    }
}

/// Shared, cloneable LLM handle with bounded concurrency.
///
/// Single access point for all LLM-backed tasks in a process: owns the CLI
/// command and a process-wide semaphore cap. Per-task specifics (extraction
/// type, prompt) stay with the caller. Cheap to clone — share one handle
/// across tasks; the cap only works if all callers share it.
/// `Arc<Semaphore>` + immutable fields: `Clone` + `Send` + `Sync`, no locks.
#[derive(Debug, Clone)]
pub struct SharedLlm {
    bin: String,
    limits: SharedLimits,
    permits: Arc<Semaphore>,
}

impl SharedLlm {
    /// Build from a command string (e.g. `"pi --print"`) and limits.
    #[must_use]
    pub fn new(bin: String, limits: SharedLimits) -> Self {
        Self {
            bin,
            permits: Arc::new(Semaphore::new(limits.max_concurrent_calls)),
            limits,
        }
    }

    /// One bounded extraction of `T` from `text`.
    ///
    /// Waits for a semaphore permit (process-wide concurrency cap), then runs
    /// the subprocess via the internal extractor (one repair retry, limits from
    /// [`SharedLimits`]). `context` is rendered into the caller's prompt
    /// template.
    ///
    /// # Errors
    /// On LLM failure (timeout / CLI crash / invalid JSON after repair retry).
    pub async fn extract<T: Extractable + Send>(&self, text: &str, context: String) -> Result<T> {
        let _permit = self
            .permits
            .acquire()
            .await
            .context("LLM semaphore closed")?;
        LlmExtractor::<T>::from_bin(&self.bin)
            .with_prompt_context(context)
            .with_max_text_len(self.limits.max_text_len)
            .with_timeout(self.limits.call_timeout)
            .extract(text)
            .await
    }

    /// Healthcheck: extract from `T::HEALTHCHECK_TEXT` and validate via
    /// `T::verify`. Cheap enough to run once per batch (not per item);
    /// catches broken bin/auth/model drift before a whole pass burns.
    ///
    /// # Errors
    /// On any extraction or validation failure.
    pub async fn verify<T: Extractable + Send>(&self) -> Result<()> {
        self.extract::<T>(T::HEALTHCHECK_TEXT, "healthcheck".to_owned())
            .await?
            .verify()
    }
}

struct ParseFailure {
    error: anyhow::Error,
    output: String,
}

fn truncate(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.get(..end).expect("end is char boundary")
}

const REPAIR_PROMPT: &str = include_str!("prompts/repair.md");

const PLACEHOLDER_ORIGINAL: &str = "{original_prompt}";
const PLACEHOLDER_PREVIOUS: &str = "{previous_output}";
const PLACEHOLDER_ERROR: &str = "{error}";

fn build_repair_prompt(
    original_prompt: &str,
    previous_output: &str,
    error: &anyhow::Error,
    max_text_len: usize,
) -> String {
    REPAIR_PROMPT
        .replace(PLACEHOLDER_ORIGINAL, original_prompt)
        .replace(
            PLACEHOLDER_PREVIOUS,
            truncate(previous_output, max_text_len),
        )
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
        let t = truncate(&s, DEFAULT_MAX_TEXT_LEN);
        assert!(t.len() <= DEFAULT_MAX_TEXT_LEN);
        assert!(t.is_char_boundary(t.len()));
    }

    #[test]
    fn test_truncate_leaves_short_text() {
        let s = "short";
        assert_eq!(truncate(s, DEFAULT_MAX_TEXT_LEN), s);
    }

    #[test]
    fn test_truncate_respects_custom_limit() {
        let s = "abcdef";
        assert_eq!(truncate(s, 3), "abc");
        assert_eq!(truncate(s, 100), s);
    }

    #[test]
    fn test_build_repair_prompt_contains_context() {
        let err = anyhow::anyhow!("expected value at line 1 column 2");
        let prompt = build_repair_prompt(
            "ORIGINAL PROMPT",
            "not json at all",
            &err,
            DEFAULT_MAX_TEXT_LEN,
        );
        assert!(prompt.contains("ORIGINAL PROMPT"));
        assert!(prompt.contains("not json at all"));
        assert!(prompt.contains("expected value at line 1 column 2"));
        assert!(prompt.contains("Return only the corrected JSON"));
    }

    #[test]
    fn test_build_repair_prompt_truncates_long_output() {
        let err = anyhow::anyhow!("boom");
        let long = "x".repeat(DEFAULT_MAX_TEXT_LEN * 2);
        let prompt = build_repair_prompt("p", &long, &err, 100);
        assert_eq!(prompt.matches('x').count(), 100);
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

    #[test]
    fn test_builder_defaults_and_overrides() {
        let e = LlmExtractor::<Dummy>::from_bin("llm");
        assert_eq!(e.max_text_len, DEFAULT_MAX_TEXT_LEN);
        assert_eq!(e.timeout, DEFAULT_TIMEOUT);
        let e = e
            .with_max_text_len(50_000)
            .with_timeout(Duration::from_secs(90));
        assert_eq!(e.max_text_len, 50_000);
        assert_eq!(e.timeout, Duration::from_secs(90));
    }
}

/// End-to-end tests against a fake CLI implemented as a shell script.
/// The prompt is passed as the last argument; scripts ignore it.
#[cfg(test)]
mod cli_tests {
    use super::*;
    use serde::Serialize;

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

    /// Extractor whose prompt embeds the input text verbatim (as JSON), so a
    /// CLI echoing its argument reveals exactly what text reached the prompt.
    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct Echo {
        text: String,
    }

    impl Extractable for Echo {
        const HEALTHCHECK_TEXT: &'static str = "hi";

        fn render_prompt(_schema: &str, text: &str, _ctx: &str) -> Result<String> {
            Ok(serde_json::to_string(&Self {
                text: text.to_string(),
            })?)
        }

        fn verify(&self) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_default_max_text_len_applied() {
        let dir = tempfile::tempdir().expect("tempdir");
        let e = LlmExtractor::<Echo>::from_bin(&fake_cli(&dir, "echo \"$1\""));
        let long = "x".repeat(DEFAULT_MAX_TEXT_LEN * 2);
        let d = e.extract(&long).await.expect("extract");
        assert_eq!(d.text.len(), DEFAULT_MAX_TEXT_LEN);
    }

    #[tokio::test]
    async fn test_max_text_len_override_applied() {
        let dir = tempfile::tempdir().expect("tempdir");
        let e =
            LlmExtractor::<Echo>::from_bin(&fake_cli(&dir, "echo \"$1\"")).with_max_text_len(100);
        let long = "x".repeat(250);
        let d = e.extract(&long).await.expect("extract");
        assert_eq!(d.text, "x".repeat(100));
    }

    #[tokio::test]
    async fn test_timeout_override_applied() {
        let dir = tempfile::tempdir().expect("tempdir");
        let e =
            LlmExtractor::<Dummy>::from_bin(&fake_cli(&dir, "sleep 2; echo '{\"value\":\"ok\"}'"))
                .with_timeout(Duration::from_millis(100));
        let err = e.extract("text").await.expect_err("must time out");
        assert!(format!("{err:#}").contains("timed out"), "err: {err:#}");
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

    #[tokio::test]
    async fn test_shared_llm_extract_success() {
        let dir = tempfile::tempdir().expect("tempdir");
        let llm = SharedLlm::new(
            fake_cli(&dir, "echo '{\"value\":\"ok\"}'"),
            SharedLimits::default(),
        );
        let d: Dummy = llm
            .extract("text", "ctx".to_owned())
            .await
            .expect("extract");
        assert_eq!(d.value, "ok");
    }

    #[tokio::test]
    async fn test_shared_llm_verify_healthcheck() {
        let dir = tempfile::tempdir().expect("tempdir");
        let llm = SharedLlm::new(
            fake_cli(&dir, "echo '{\"value\":\"ok\"}'"),
            SharedLimits::default(),
        );
        llm.verify::<Dummy>().await.expect("verify");
    }

    #[tokio::test]
    async fn test_shared_llm_caps_concurrency() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Each call increments `cur` under a mkdir lock, records the max into
        // `max`, sleeps, then decrements — portable across GNU/BSD userlands.
        let script = r#"
d="$(dirname "$0")"
lock="$d/.lock"
while ! mkdir "$lock" 2>/dev/null; do :; done
c=$(cat "$d/cur" 2>/dev/null || echo 0)
c=$((c + 1))
echo "$c" > "$d/cur"
m=$(cat "$d/max" 2>/dev/null || echo 0)
[ "$c" -gt "$m" ] && echo "$c" > "$d/max"
rm -rf "$lock"
sleep 0.2
while ! mkdir "$lock" 2>/dev/null; do :; done
c=$(cat "$d/cur")
echo $((c - 1)) > "$d/cur"
rm -rf "$lock"
echo '{"value":"ok"}'
"#;
        let llm = SharedLlm::new(
            fake_cli(&dir, script),
            SharedLimits {
                max_concurrent_calls: 2,
                ..SharedLimits::default()
            },
        );
        futures(llm.clone(), 8).await;
        let max: usize = std::fs::read_to_string(dir.path().join("max"))
            .expect("read max")
            .trim()
            .parse()
            .expect("max is a number");
        assert_eq!(max, 2, "never more than 2 concurrent calls");
    }

    async fn futures(llm: SharedLlm, n: usize) {
        let mut handles = Vec::new();
        for _ in 0..n {
            let llm = llm.clone();
            handles.push(tokio::spawn(async move {
                let _: Dummy = llm
                    .extract("text", "ctx".to_owned())
                    .await
                    .expect("extract");
            }));
        }
        for h in handles {
            h.await.expect("join");
        }
    }
}
