//! Rust-native chat client for OpenAI-compatible endpoints.
//!
//! Extraction contract (prompt template + healthcheck) over HTTP instead of a CLI:
//! (prompt template, healthcheck, one repair retry), but the request goes over
//! HTTP to `/chat/completions` instead of spawning a CLI process.
//!
//! Bounded like the other gateways: one semaphore cap per handle
//! ([`ConcurrencyLimits`]) and a [`RetryPolicy`] for transient failures.
//!
//! ```no_run
//! # use patterns::limits::ConcurrencyLimits;
//! # async fn run<T: patterns::llm::Extractable + Send>() -> anyhow::Result<()> {
//! let llm = patterns::llm::ChatLlm::new(
//!     "https://api.deepseek.com",
//!     "key",
//!     "deepseek-flash",
//!     ConcurrencyLimits::default(),
//! );
//! let value: T = llm.extract("text to read", "company context".to_owned()).await?;
//! # let _ = value;
//! # Ok(())
//! # }
//! ```

use crate::limits::ConcurrencyLimits;
use crate::retry::{HttpResponse, RetryPolicy};
use anyhow::{Context, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

pub use crate::limits::{
    ConcurrencyLimits as Limits, DEFAULT_MAX_CONCURRENT_CALLS, DEFAULT_TIMEOUT,
};

/// Default maximum byte length of text embedded into prompts.
pub const DEFAULT_MAX_TEXT_LEN: usize = 4000;

/// A type that can be extracted from LLM output.
///
/// Implementations render their own prompt (e.g. via an askama template and a
/// project-local prompt-kind enum) so this crate stays template-agnostic.
pub trait Extractable: JsonSchema + for<'de> Deserialize<'de> {
    /// Fixture text used to verify the LLM produces valid structured output.
    fn healthcheck_text() -> Result<String>;

    /// Context the healthcheck fixture is rendered with; mirrors the real call's
    /// `prompt_context` (e.g. a company/CV blurb).
    fn healthcheck_context() -> String;

    /// Render the extraction prompt from the JSON schema, the (already
    /// truncated) input text, and optional dynamic prompt context.
    fn render_prompt(schema: &str, text: &str, prompt_context: &str) -> Result<String>;

    /// Validate that a healthcheck extraction succeeded.
    fn verify(&self) -> Result<()>;
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

/// Longest prefix of `text` with at most `max` bytes, on a char boundary.
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

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Semaphore;

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
                .context("chat request failed")?;
            let status = response.status().as_u16();
            let body = response.text().await.context("read chat response body")?;
            Ok(HttpResponse { status, body })
        })
    }
}

/// How the model is asked to shape its answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseFormat {
    /// Prompt only; the JSON contract lives in the caller's template.
    PromptOnly,
    /// Also send `response_format: {"type":"json_object"}`. Not every provider
    /// accepts it, so the default stays [`Self::PromptOnly`].
    JsonObject,
}

impl ResponseFormat {
    /// The `response_format` value sent on the wire, if any.
    const fn wire(self) -> Option<WireResponseFormat> {
        match self {
            Self::PromptOnly => None,
            Self::JsonObject => Some(WireResponseFormat {
                kind: "json_object",
            }),
        }
    }
}

/// Shared, cloneable chat handle with bounded concurrency.
///
/// Single access point for LLM-backed tasks in a process: owns the endpoint,
/// credentials and a semaphore cap for every clone of this handle. Per-task specifics (extraction
/// type, prompt) stay with the caller. Cheap to clone — share one handle across
/// tasks; the cap only works if all callers share it.
#[derive(Clone)]
pub struct ChatLlm {
    base_url: String,
    api_key: String,
    model: String,
    max_text_len: usize,
    max_tokens: Option<u32>,
    response_format: ResponseFormat,
    limits: ConcurrencyLimits,
    retry: RetryPolicy,
    permits: Arc<Semaphore>,
    transport: Arc<dyn Transport>,
}

impl fmt::Debug for ChatLlm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChatLlm")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .field("max_text_len", &self.max_text_len)
            .field("max_tokens", &self.max_tokens)
            .field("response_format", &self.response_format)
            .finish_non_exhaustive()
    }
}

impl ChatLlm {
    /// Build from a base URL (behaviour-specific part included, e.g.
    /// `https://api.deepseek.com/v1`), an API key, a model id and shared limits.
    ///
    /// # Panics
    /// If the default `reqwest` client cannot be built (invalid system TLS config).
    #[must_use]
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        limits: ConcurrencyLimits,
    ) -> Self {
        let client = reqwest::Client::builder()
            .build()
            .expect("default reqwest client builds");
        Self::with_transport(
            Arc::new(HttpTransport { client }),
            base_url.into(),
            api_key.into(),
            model.into(),
            limits,
        )
    }

    /// Test seam: build with an injected transport.
    fn with_transport(
        transport: Arc<dyn Transport>,
        base_url: String,
        api_key: String,
        model: String,
        limits: ConcurrencyLimits,
    ) -> Self {
        let permits = Arc::new(Semaphore::new(limits.max_concurrent_calls));
        Self {
            base_url,
            api_key,
            model,
            max_text_len: DEFAULT_MAX_TEXT_LEN,
            max_tokens: None,
            response_format: ResponseFormat::PromptOnly,
            limits,
            retry: RetryPolicy::default(),
            permits,
            transport,
        }
    }

    /// Override the maximum byte length of text embedded into prompts
    /// (default: [`DEFAULT_MAX_TEXT_LEN`]).
    #[must_use]
    pub const fn with_max_text_len(mut self, max_text_len: usize) -> Self {
        self.max_text_len = max_text_len;
        self
    }

    /// Cap the answer length, when the provider accepts `max_tokens`.
    #[must_use]
    pub const fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Choose how the model is asked to shape its answer.
    #[must_use]
    pub const fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = response_format;
        self
    }

    /// Override the transient-failure retry policy (default 3 attempts, 250ms→2s).
    #[must_use]
    pub const fn with_retry_policy(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// One bounded extraction of `T` from `text`.
    ///
    /// Waits for a semaphore permit (this handle's concurrency cap), then runs one
    /// request through [`RetryPolicy`] (transient failures only) and parses the
    /// answer, retrying once with a repair prompt when parsing fails. `context` is
    /// rendered into the caller's prompt template.
    ///
    /// # Errors
    /// On transport failure, an empty answer, or invalid JSON after the repair retry.
    pub async fn extract<T: Extractable + Send>(&self, text: &str, context: String) -> Result<T> {
        let _permit = self
            .permits
            .acquire()
            .await
            .context("chat semaphore closed")?;
        let prompt = self.render::<T>(text, &context)?;
        match self.parse::<T>(&prompt).await {
            Ok(parsed) => Ok(parsed),
            Err(failure) => {
                let repair = build_repair_prompt(
                    &prompt,
                    &failure.output,
                    &failure.error,
                    self.max_text_len,
                );
                match self.parse::<T>(&repair).await {
                    Ok(parsed) => Ok(parsed),
                    Err(retry) => Err(retry.error.context(format!(
                        "chat parse failed after one retry; first error: {}; first output: {}",
                        failure.error, failure.output
                    ))),
                }
            }
        }
    }

    /// Healthcheck: extract from `T::healthcheck_text` and validate via `T::verify`.
    ///
    /// Cheap enough to run once per batch (not per item); catches broken endpoint,
    /// auth or model id before a whole pass burns.
    ///
    /// # Errors
    /// On any extraction or validation failure.
    pub async fn verify<T: Extractable + Send>(&self) -> Result<()> {
        let text = T::healthcheck_text()?;
        let value = self.extract::<T>(&text, T::healthcheck_context()).await?;
        value.verify()
    }

    /// Render the prompt from the extracted type's template.
    fn render<T: Extractable>(&self, text: &str, prompt_context: &str) -> Result<String> {
        let schema = serde_json::to_string_pretty(&schemars::schema_for!(T))
            .context("serialize extraction schema")?;
        let truncated = truncate(text, self.max_text_len);
        T::render_prompt(&schema, truncated, prompt_context)
    }

    /// One request + parse; a failure keeps the raw output for the repair prompt.
    async fn parse<T: for<'de> Deserialize<'de>>(&self, prompt: &str) -> Result<T, ParseFailure> {
        let output = self.complete(prompt).await.map_err(|error| ParseFailure {
            error,
            output: String::new(),
        })?;
        if output.is_empty() || output.eq_ignore_ascii_case("none") {
            return Err(ParseFailure {
                error: anyhow::anyhow!("chat returned empty or NONE response"),
                output,
            });
        }
        let stripped = strip_json_fences(&output);
        serde_json::from_str(&stripped).map_err(|error| ParseFailure {
            error: anyhow::Error::new(error)
                .context(format!("failed to parse chat JSON: {stripped}")),
            output,
        })
    }

    /// One chat request, retried on transient failures; returns the answer text.
    async fn complete(&self, prompt: &str) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let body = serde_json::to_vec(&RequestBody {
            model: &self.model,
            messages: vec![WireMessage {
                role: "user",
                content: prompt,
            }],
            max_tokens: self.max_tokens,
            response_format: self.response_format.wire(),
        })
        .context("serialize chat request")?;

        let response = self
            .retry
            .run("chat", self.limits.call_timeout, || {
                let transport = Arc::clone(&self.transport);
                let url = url.clone();
                let body = body.clone();
                let api_key = self.api_key.clone();
                async move { transport.post(&url, &api_key, &body).await }
            })
            .await?;

        let parsed: ChatResponse = serde_json::from_str(&response.body)
            .with_context(|| format!("parse chat response envelope: {}", response.body))?;
        parsed
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .map(|content| content.trim().to_owned())
            .filter(|content| !content.is_empty())
            .context("chat response has no message content")
    }
}

/// A parse failure keeps the raw output so the repair prompt can quote it.
struct ParseFailure {
    error: anyhow::Error,
    output: String,
}

#[derive(Serialize)]
struct RequestBody<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<WireResponseFormat>,
}

#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct WireResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<WireChoice>,
}

#[derive(Deserialize)]
struct WireChoice {
    message: WireMessageOut,
}

#[derive(Deserialize)]
struct WireMessageOut {
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retry::HttpResponse;
    use schemars::JsonSchema;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[derive(Debug, Deserialize, JsonSchema)]
    struct Dummy {
        value: String,
    }

    impl Extractable for Dummy {
        fn healthcheck_text() -> Result<String> {
            Ok("hello".to_owned())
        }

        fn healthcheck_context() -> String {
            "ctx".to_owned()
        }

        fn render_prompt(schema: &str, text: &str, _context: &str) -> Result<String> {
            Ok(format!(
                "Reply with JSON matching this schema and nothing else:\n{schema}\n\nInput:\n{text}"
            ))
        }

        fn verify(&self) -> Result<()> {
            anyhow::ensure!(!self.value.is_empty(), "empty value");
            Ok(())
        }
    }

    fn answer(content: &str) -> String {
        serde_json::json!({ "choices": [{ "message": { "content": content } }] }).to_string()
    }

    struct FakeTransport {
        responses: Arc<std::sync::Mutex<Vec<HttpResponse>>>,
        calls: Arc<AtomicUsize>,
    }

    impl FakeTransport {
        fn new(responses: Vec<HttpResponse>) -> Arc<Self> {
            Arc::new(Self {
                responses: Arc::new(std::sync::Mutex::new(responses)),
                calls: Arc::new(AtomicUsize::new(0)),
            })
        }
    }

    impl Transport for FakeTransport {
        fn post<'a>(
            &'a self,
            _url: &'a str,
            _api_key: &'a str,
            _body: &'a [u8],
        ) -> BoxFuture<'a, Result<HttpResponse>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let next = self.responses.lock().expect("fake transport lock").pop();
            let next = next.unwrap_or_else(|| HttpResponse {
                status: 500,
                body: "no response left".to_owned(),
            });
            Box::pin(async move { Ok(next) })
        }
    }

    /// Client built from the live-test env vars; only called by `#[ignore]`d tests.
    fn live_client() -> ChatLlm {
        let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY set");
        let base_url =
            std::env::var("LLM_BASE_URL").unwrap_or_else(|_| "https://api.deepseek.com".to_owned());
        let model = std::env::var("LLM_MODEL").unwrap_or_else(|_| "deepseek-flash".to_owned());
        ChatLlm::new(base_url, api_key, model, ConcurrencyLimits::default())
            .with_retry_policy(RetryPolicy::none())
    }

    fn instant(attempts: usize) -> RetryPolicy {
        RetryPolicy {
            max_attempts: attempts,
            base_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
        }
    }

    fn client(transport: Arc<dyn Transport>) -> ChatLlm {
        ChatLlm::with_transport(
            transport,
            "http://test/v1".to_owned(),
            "k".to_owned(),
            "m".to_owned(),
            ConcurrencyLimits::default(),
        )
        .with_retry_policy(instant(2))
    }

    #[tokio::test]
    async fn extracts_json_and_strips_fences() {
        let transport = FakeTransport::new(vec![HttpResponse {
            status: 200,
            body: answer("```json\n{\"value\":\"ok\"}\n```"),
        }]);
        let value: Dummy = client(transport)
            .extract("t", "c".to_owned())
            .await
            .expect("extract");
        assert_eq!(value.value, "ok");
    }

    #[tokio::test]
    async fn repairs_once_on_invalid_json() {
        // LIFO: the repair answer is consumed first.
        let transport = FakeTransport::new(vec![
            HttpResponse {
                status: 200,
                body: answer("{\"value\":\"repaired\"}"),
            },
            HttpResponse {
                status: 200,
                body: answer("not json"),
            },
        ]);
        let calls = Arc::clone(&transport.calls);
        let value: Dummy = client(transport)
            .extract("t", "c".to_owned())
            .await
            .expect("repair");
        assert_eq!(value.value, "repaired");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "one repair retry");
    }

    #[tokio::test]
    async fn empty_answer_is_an_error() {
        let transport = FakeTransport::new(vec![
            HttpResponse {
                status: 200,
                body: answer(""),
            },
            HttpResponse {
                status: 200,
                body: answer(""),
            },
        ]);
        client(transport)
            .extract::<Dummy>("t", "c".to_owned())
            .await
            .expect_err("empty answer");
    }

    #[tokio::test]
    async fn transient_status_is_retried() {
        let transport = FakeTransport::new(vec![
            HttpResponse {
                status: 200,
                body: answer("{\"value\":\"ok\"}"),
            },
            HttpResponse {
                status: 503,
                body: "busy".to_owned(),
            },
        ]);
        let calls = Arc::clone(&transport.calls);
        let value: Dummy = client(transport)
            .extract("t", "c".to_owned())
            .await
            .expect("retry");
        assert_eq!(value.value, "ok");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn client_error_is_not_retried() {
        // One for the attempt, one for the repair retry: a 4xx is never retried.
        let transport = FakeTransport::new(vec![
            HttpResponse {
                status: 400,
                body: "bad request".to_owned(),
            },
            HttpResponse {
                status: 400,
                body: "bad request".to_owned(),
            },
        ]);
        let calls = Arc::clone(&transport.calls);
        client(transport)
            .extract::<Dummy>("t", "c".to_owned())
            .await
            .expect_err("client error");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "4xx is not transient");
    }

    #[tokio::test]
    async fn verify_runs_the_healthcheck() {
        let transport = FakeTransport::new(vec![HttpResponse {
            status: 200,
            body: answer("{\"value\":\"ok\"}"),
        }]);
        client(transport)
            .verify::<Dummy>()
            .await
            .expect("healthcheck");
    }

    #[tokio::test]
    #[ignore = "live: requires DEEPSEEK_API_KEY"]
    async fn live_extract_round_trip() {
        let value: Dummy = live_client()
            .extract("The answer is 42.", "test".to_owned())
            .await
            .expect("live extract");
        assert!(!value.value.is_empty(), "empty value: {value:?}");
    }

    #[test]
    fn response_format_maps_to_the_wire() {
        assert!(ResponseFormat::PromptOnly.wire().is_none());
        assert_eq!(
            ResponseFormat::JsonObject.wire().map(|format| format.kind),
            Some("json_object")
        );
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let transport = FakeTransport::new(Vec::new());
        let llm = ChatLlm::with_transport(
            transport,
            "http://test/v1".to_owned(),
            "SECRET-KEY-VALUE".to_owned(),
            "m".to_owned(),
            ConcurrencyLimits::default(),
        );
        let text = format!("{llm:?}");
        assert!(text.contains("<redacted>"), "{text}");
        assert!(!text.contains("SECRET-KEY-VALUE"), "{text}");
    }
}
