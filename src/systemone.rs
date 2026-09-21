//! Generic `TypeSafe` `SystemOne` (Jev) client.
//!
//! One bounded HTTP client for the `SystemOne` protocol (`POST {base_url}/v1/systemone`),
//! sibling to `llm_cli`/`embed`. Backends differ only by `base_url` (direct
//! `https://api.typesafe.ai` vs `OpenRouter` `https://openrouter.ai/api`); the path is
//! fixed internally. The caller supplies base URL, API key and model — the client
//! never reads the environment.
//!
//! Questions are independent and evaluated in parallel: they cannot see each other's
//! answers, so do not chain them. Every answer exposes full probabilities; `score`
//! additionally derives an expected position from them.

use crate::limits::ConcurrencyLimits;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::timeout;

/// One typed question evaluated against the request state.
///
/// `instructions` may be a string, object or array. Serializes with a `type` tag
/// (`noul`/`choice`/`score`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Yes/no; answered with the probability of "yes".
    Noul {
        /// The question, referring to the state.
        instructions: Value,
        /// What "yes" and "no" mean, when spelled out.
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<Value>,
    },
    /// One option from a set; answered with the argmax and its confidence.
    Choice {
        /// The question, referring to the state.
        instructions: Value,
        /// Option → description (text, object, array or null).
        criteria: BTreeMap<String, Option<Value>>,
    },
    /// A position along an ordered rubric; answered with probabilities per level.
    Score {
        /// The question, referring to the state.
        instructions: Value,
        /// Level descriptions (text, object or array), lowest first.
        criteria: Vec<Value>,
    },
}

/// Named questions; answers come back under the same ids.
pub type QuestionMap = BTreeMap<String, Question>;

/// A typed question set: a domain struct whose fields are both the question ids
/// and the deserialized answers.
///
/// Implement by hand, or generate with [`crate::define_questions!`].
pub trait Questions {
    /// The struct the response `answers` object deserializes into.
    type Answers: for<'de> Deserialize<'de>;

    /// Build the request question map; ids must match `Answers` field names.
    fn questions() -> QuestionMap;

    /// Render the shared state from input `text` and dynamic `prompt_context`.
    ///
    /// Defaults to plain concatenation; [`crate::define_questions!`] overrides
    /// it with the askama template declared alongside the questions.
    fn render_state(text: &str, prompt_context: &str) -> Result<String> {
        Ok(format!("{text}\n\n{prompt_context}"))
    }
}

/// Define a [`Questions`] set.
///
/// One `struct` whose fields are both the question ids and the typed answers,
/// plus the `Questions` impl building the map and an inherent `render_state`
/// rendering the input from an askama template.
///
/// The template path resolves against the *consumer* crate's template dirs
/// (its `askama.toml` / `templates/`); it receives `{{ text }}` and
/// `{{ prompt_context }}`.
///
/// ```ignore
/// define_questions! {
///     JobFit: "job_fit_input.md" {
///         relevant:  noul("Is this role a good fit?"),
///         seniority: score("Seniority match?", ["junior", "senior"]),
///         role:      choice("Primary role?", ["trading", ("other", "none of these")]),
///     }
/// }
/// ```
#[macro_export]
macro_rules! define_questions {
    ($Name:ident : $path:literal { $($field:ident : $kind:ident ( $($args:tt)* )),* $(,)? }) => {
        #[derive($crate::serde::Deserialize, ::std::fmt::Debug)]
        pub struct $Name {
            $( pub $field: $crate::__systemone_answer_type!($kind), )*
        }

        $crate::pastey::paste! {
            #[derive($crate::askama::Template)]
            #[template(path = $path, ext = "md", askama = $crate::askama)]
            struct [<$Name Input>]<'a> {
                text: &'a str,
                prompt_context: &'a str,
            }
        }

        impl $crate::systemone::Questions for $Name {
            type Answers = $Name;

            /// Render the `SystemOne` `state` from input `text` and dynamic
            /// `prompt_context` using the template declared with the questions.
            fn render_state(text: &str, prompt_context: &str) -> ::anyhow::Result<String> {
                use $crate::askama::Template;
                $crate::pastey::paste! {
                    [<$Name Input>] { text, prompt_context }
                }
                .render()
                .map_err(::std::convert::Into::into)
            }

            fn questions() -> $crate::systemone::QuestionMap {
                ::std::iter::IntoIterator::into_iter([
                    $(
                        (
                            ::std::string::String::from(::std::stringify!($field)),
                            $crate::__systemone_question!($kind ( $($args)* )),
                        )
                    ),*
                ])
                .collect()
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __systemone_answer_type {
    (noul) => {
        $crate::systemone::Noul
    };
    (score) => {
        $crate::systemone::Score
    };
    (choice) => {
        $crate::systemone::Choice
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __systemone_question {
    (noul($instructions:expr)) => {
        $crate::systemone::noul($instructions)
    };
    (score($instructions:expr, [ $($level:expr),* $(,)? ])) => {
        $crate::systemone::score($instructions, [ $($level),* ])
    };
    (choice($instructions:expr, [ $($entry:tt),* $(,)? ])) => {
        $crate::systemone::choice(
            $instructions,
            [ $( $crate::__systemone_choice_entry!($entry) ),* ],
        )
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __systemone_choice_entry {
    (($label:expr, $description:expr)) => {
        ($label, ::std::option::Option::Some($description))
    };
    ($label:expr) => {
        ($label, ::std::option::Option::<&str>::None)
    };
}

/// A yes/no question without criteria descriptions.
#[must_use]
pub fn noul(instructions: impl Into<Value>) -> Question {
    Question::Noul {
        instructions: instructions.into(),
        criteria: None,
    }
}

/// Pick one of `options`. Each option is `(name, optional description)` where a
/// description is text, an object, an array, or `None` (JSON null).
#[must_use]
pub fn choice<K, V>(
    instructions: impl Into<Value>,
    options: impl IntoIterator<Item = (K, Option<V>)>,
) -> Question
where
    K: Into<String>,
    V: Into<Value>,
{
    Question::Choice {
        instructions: instructions.into(),
        criteria: options
            .into_iter()
            .map(|(name, description)| (name.into(), description.map(Into::into)))
            .collect(),
    }
}

/// Rate along `levels` (text, object or array), ordered from lowest to highest.
#[must_use]
pub fn score<L: Into<Value>>(
    instructions: impl Into<Value>,
    levels: impl IntoIterator<Item = L>,
) -> Question {
    Question::Score {
        instructions: instructions.into(),
        criteria: levels.into_iter().map(Into::into).collect(),
    }
}

/// Answer to a [`Question::Noul`]: probability of "yes", 0..=1.
///
/// Field name matches the JSON key; the `type` tag is ignored.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Noul {
    /// Probability of "yes".
    pub noul: f32,
}

/// Answer to a [`Question::Score`].
///
/// The wire also returns `score` and `legend`; both are derived from
/// `probabilities` and the criteria you sent, so they are not stored.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Score {
    /// Certainty derived from the shape of `probabilities`, 0..=1.
    pub confidence: f32,
    /// Level index → probability; sums to 1.
    pub probabilities: BTreeMap<u8, f32>,
}

impl Score {
    /// Probability-weighted position across the levels; may fall between levels.
    #[must_use]
    pub fn expected(&self) -> f32 {
        self.probabilities
            .iter()
            .map(|(&level, &p)| f32::from(level) * p)
            .sum()
    }

    /// Highest-probability level index, or `None` when empty.
    #[must_use]
    pub fn argmax_level(&self) -> Option<u8> {
        self.probabilities
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(&level, _)| level)
    }
}

/// Answer to a [`Question::Choice`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Choice {
    /// The highest-probability option.
    pub choice: String,
    /// Certainty derived from the shape of the probabilities, 0..=1.
    pub confidence: f32,
    /// Option → probability; sums to 1.
    pub probabilities: BTreeMap<String, f32>,
}

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// HTTP seam; injectable so tests run fully offline.
trait Transport: Send + Sync + 'static {
    fn post<'a>(
        &'a self,
        url: &'a str,
        api_key: &'a str,
        body: &'a [u8],
    ) -> BoxFuture<'a, Result<String>>;
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
    ) -> BoxFuture<'a, Result<String>> {
        Box::pin(async move {
            let response = self
                .client
                .post(url)
                .bearer_auth(api_key)
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(body.to_vec())
                .send()
                .await
                .context("systemone request failed")?;
            let status = response.status();
            let text = response
                .text()
                .await
                .context("read systemone response body")?;
            if !status.is_success() {
                anyhow::bail!("systemone returned HTTP {status}: {text}");
            }
            Ok(text)
        })
    }
}

#[derive(Serialize)]
struct RequestBody<'a> {
    model: &'a str,
    state: &'a Value,
    questions: &'a QuestionMap,
}

#[derive(Deserialize)]
struct RawResponse {
    answers: Value,
}

/// A domain whose healthcheck the client can verify end-to-end.
///
/// Mirrors [`crate::llm_cli::Extractable`]: the domain owns the probe fixture
/// and the validation of the parsed answers. Define one impl per evaluation
/// shape and call [`SharedSystemOne::verify`] before a batch.
pub trait Evaluatable: Questions + for<'de> Deserialize<'de> {
    /// Fixture text rendered through the real template for the batch gate.
    const HEALTHCHECK_TEXT: &'static str;
    /// Validate the parsed healthcheck answers.
    fn verify(&self) -> Result<()>;
}

/// Shared, cloneable `SystemOne` handle with bounded concurrency.
///
/// Cheap to clone — share one handle across tasks so the concurrency cap holds.
/// The API key is redacted in [`Debug`].
#[derive(Clone)]
pub struct SharedSystemOne {
    base_url: String,
    api_key: String,
    model: String,
    limits: ConcurrencyLimits,
    permits: Arc<Semaphore>,
    transport: Arc<dyn Transport>,
}

impl fmt::Debug for SharedSystemOne {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedSystemOne")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl SharedSystemOne {
    /// Build from a base URL, API key, model and shared limits.
    ///
    /// # Panics
    /// If the default `reqwest` client cannot be built (invalid system TLS config).
    #[must_use]
    pub fn new(
        base_url: String,
        api_key: String,
        model: String,
        limits: ConcurrencyLimits,
    ) -> Self {
        let client = reqwest::Client::builder()
            .build()
            .expect("default reqwest client builds");
        Self::with_transport(
            Arc::new(HttpTransport { client }),
            base_url,
            api_key,
            model,
            limits,
        )
    }

    /// Test seam: build with an injected transport.
    #[must_use]
    fn with_transport(
        transport: Arc<dyn Transport>,
        base_url: String,
        api_key: String,
        model: String,
        limits: ConcurrencyLimits,
    ) -> Self {
        Self {
            base_url,
            api_key,
            model,
            permits: Arc::new(Semaphore::new(limits.max_concurrent_calls)),
            limits,
            transport,
        }
    }

    /// Evaluate a typed question set against `state`, returning its `Answers`.
    ///
    /// `state` is anything serializable: a rendered input string, a typed
    /// struct, or a `json!({...})` object. One bounded request (no retries)
    /// under the process-wide concurrency cap. On parse failure the raw
    /// `answers` JSON is included in the error context.
    pub async fn evaluate<Q: Questions>(&self, state: impl Serialize) -> Result<Q::Answers> {
        let state = serde_json::to_value(state).context("serialize systemone state")?;
        self.evaluate_map::<Q::Answers>(&state, &Q::questions())
            .await
    }

    /// Evaluate after rendering `Q`'s state from `text` + `prompt_context`.
    ///
    /// Convenience over [`Self::evaluate`] using [`Questions::render_state`].
    pub async fn evaluate_text<Q: Questions>(
        &self,
        text: &str,
        prompt_context: &str,
    ) -> Result<Q::Answers> {
        self.evaluate::<Q>(Q::render_state(text, prompt_context)?)
            .await
    }

    /// Core request: evaluate `questions` against `state`, deserialize `answers` into `A`.
    async fn evaluate_map<A: for<'de> Deserialize<'de>>(
        &self,
        state: &Value,
        questions: &QuestionMap,
    ) -> Result<A> {
        let _permit = self
            .permits
            .acquire()
            .await
            .context("systemone semaphore closed")?;
        let url = format!("{}/v1/systemone", self.base_url);
        let body = serde_json::to_vec(&RequestBody {
            model: &self.model,
            state,
            questions,
        })
        .context("serialize systemone request body")?;
        let text = timeout(
            self.limits.call_timeout,
            self.transport.post(&url, &self.api_key, &body),
        )
        .await
        .context("systemone call timed out")?
        .context("systemone transport failed")?;
        let response: RawResponse = serde_json::from_str(&text)
            .with_context(|| format!("parse systemone response envelope: {text}"))?;
        let answers = response.answers;
        serde_json::from_value::<A>(answers.clone())
            .with_context(|| format!("parse systemone answers: {answers}"))
    }

    /// Domain-aware healthcheck, mirroring [`crate::llm_cli::SharedLlm::verify`].
    ///
    /// Renders `E::HEALTHCHECK_TEXT` through `E`'s real template, evaluates
    /// `E`'s real questions, and validates the result via [`Evaluatable::verify`].
    /// Cheap enough to run once per batch (not per item); catches broken
    /// auth/model/template/shape drift before a whole pass burns.
    pub async fn verify<E: Evaluatable>(&self) -> Result<()> {
        let state = E::render_state(E::HEALTHCHECK_TEXT, "healthcheck")?;
        self.evaluate_map::<E>(&serde_json::json!(state), &E::questions())
            .await?
            .verify()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[derive(Debug, Clone)]
    struct Recorded {
        url: String,
        api_key: String,
        body: Value,
    }

    struct FakeTransport {
        responses: Mutex<VecDeque<Result<String>>>,
        requests: Mutex<Vec<Recorded>>,
    }

    impl FakeTransport {
        fn new(responses: Vec<String>) -> Arc<Self> {
            Arc::new(Self {
                responses: Mutex::new(responses.into_iter().map(Ok).collect()),
                requests: Mutex::new(Vec::new()),
            })
        }

        fn failing(message: &str) -> Arc<Self> {
            Arc::new(Self {
                responses: Mutex::new(VecDeque::from([Err(anyhow::Error::msg(
                    message.to_owned(),
                ))])),
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
        ) -> BoxFuture<'a, Result<String>> {
            self.requests.lock().expect("lock requests").push(Recorded {
                url: url.to_owned(),
                api_key: api_key.to_owned(),
                body: serde_json::from_slice(body).expect("request body is JSON"),
            });
            let next = self.responses.lock().expect("lock responses").pop_front();
            Box::pin(async move {
                match next {
                    Some(result) => result,
                    None => anyhow::bail!("no fake response queued"),
                }
            })
        }
    }

    struct SleepTransport {
        delay: Duration,
    }

    impl Transport for SleepTransport {
        fn post<'a>(
            &'a self,
            _url: &'a str,
            _api_key: &'a str,
            _body: &'a [u8],
        ) -> BoxFuture<'a, Result<String>> {
            let delay = self.delay;
            Box::pin(async move {
                tokio::time::sleep(delay).await;
                Ok(String::new())
            })
        }
    }

    #[derive(Clone)]
    struct ConcurrencyTransport {
        inner: Arc<ConcurrencyInner>,
    }

    struct ConcurrencyInner {
        active: AtomicUsize,
        max: AtomicUsize,
        delay: Duration,
    }

    impl ConcurrencyTransport {
        fn new(delay: Duration) -> Self {
            Self {
                inner: Arc::new(ConcurrencyInner {
                    active: AtomicUsize::new(0),
                    max: AtomicUsize::new(0),
                    delay,
                }),
            }
        }

        fn max_seen(&self) -> usize {
            self.inner.max.load(Ordering::SeqCst)
        }
    }

    impl Transport for ConcurrencyTransport {
        fn post<'a>(
            &'a self,
            _url: &'a str,
            _api_key: &'a str,
            _body: &'a [u8],
        ) -> BoxFuture<'a, Result<String>> {
            let inner = self.inner.clone();
            Box::pin(async move {
                let now = inner.active.fetch_add(1, Ordering::SeqCst) + 1;
                inner.max.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(inner.delay).await;
                inner.active.fetch_sub(1, Ordering::SeqCst);
                Ok(r#"{"answers":{"urgency":{"type":"noul","noul":0.5}}}"#.to_owned())
            })
        }
    }

    define_questions! {
        NoulOut: "test_input.md" {
            urgency: noul("Is it urgent?"),
        }
    }

    define_questions! {
        Mixed: "test_input.md" {
            is_urgent: noul("Urgent?"),
            severity:  score("Severity?", ["low", "high"]),
            team:      choice("Team?", ["billing", ("other", "none of these")]),
        }
    }

    impl Evaluatable for NoulOut {
        const HEALTHCHECK_TEXT: &'static str = "Healthcheck text.";

        fn verify(&self) -> Result<()> {
            anyhow::ensure!(
                (0.0..=1.0).contains(&self.urgency.noul),
                "noul out of range: {}",
                self.urgency.noul
            );
            Ok(())
        }
    }

    fn client(transport: Arc<dyn Transport>) -> SharedSystemOne {
        SharedSystemOne::with_transport(
            transport,
            "https://example.test".to_owned(),
            "secret".to_owned(),
            "jev-test".to_owned(),
            ConcurrencyLimits::default(),
        )
    }

    const NOUL_RESPONSE: &str = r#"{"answers":{"urgency":{"type":"noul","noul":0.7}},"model":"jev-test","usage":{"input_tokens":3,"output_tokens":1}}"#;

    #[test]
    fn noul_serializes_without_criteria() {
        assert_eq!(
            serde_json::to_value(noul("Is it urgent?")).expect("serialize noul"),
            json!({"type": "noul", "instructions": "Is it urgent?"})
        );
    }

    #[test]
    fn choice_serializes_null_for_undescribed_options() {
        let question = choice(
            "Which team?",
            [("billing", Some("Payments")), ("sales", None::<&str>)],
        );
        assert_eq!(
            serde_json::to_value(question).expect("serialize choice"),
            json!({
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "Payments", "sales": null}
            })
        );
    }

    #[test]
    fn score_serializes_levels_in_order() {
        let question = score("Rate frustration", ["Calm", "Frustrated", "Very angry"]);
        assert_eq!(
            serde_json::to_value(question).expect("serialize score"),
            json!({
                "type": "score",
                "instructions": "Rate frustration",
                "criteria": ["Calm", "Frustrated", "Very angry"]
            })
        );
    }

    #[test]
    fn structured_criteria_are_accepted() {
        let choice_question = choice(
            "Pick",
            [("a", Some(json!({"when": "x"}))), ("b", None::<Value>)],
        );
        assert_eq!(
            serde_json::to_value(choice_question).expect("serialize choice"),
            json!({
                "type": "choice",
                "instructions": "Pick",
                "criteria": {"a": {"when": "x"}, "b": null}
            })
        );
        let score_question = score("Rate", [json!({"level": "low"}), json!(["mid"])]);
        assert_eq!(
            serde_json::to_value(score_question).expect("serialize score"),
            json!({
                "type": "score",
                "instructions": "Rate",
                "criteria": [{"level": "low"}, ["mid"]]
            })
        );
    }

    #[test]
    fn noul_answer_deserializes_from_tagged_object() {
        let answer: Noul =
            serde_json::from_value(json!({"type": "noul", "noul": 0.9})).expect("deserialize");
        assert!((answer.noul - 0.9).abs() < 1e-6);
    }

    #[test]
    fn score_answer_deserializes_level_indexed_probabilities() {
        let answer: Score = serde_json::from_value(json!({
            "type": "score",
            "score": 1.5,
            "confidence": 0.8,
            "probabilities": {"0": 0.2, "1": 0.8},
            "legend": {"0": "low", "1": "high"}
        }))
        .expect("deserialize");
        assert!((answer.expected() - 0.8).abs() < 1e-6, "expected position");
        assert_eq!(answer.argmax_level(), Some(1));
        assert!((answer.confidence - 0.8).abs() < 1e-6);
        let level1 = answer.probabilities.get(&1).expect("level 1 present");
        assert!((*level1 - 0.8).abs() < 1e-6);
    }

    #[test]
    fn choice_answer_deserializes_probabilities() {
        let answer: Choice = serde_json::from_value(json!({
            "type": "choice",
            "choice": "technical",
            "confidence": 0.82,
            "probabilities": {"billing": 0.08, "technical": 0.85}
        }))
        .expect("deserialize");
        assert_eq!(answer.choice, "technical");
        assert!((answer.confidence - 0.82).abs() < 1e-6);
        let technical = answer
            .probabilities
            .get("technical")
            .expect("technical present");
        assert!((*technical - 0.85).abs() < 1e-6);
    }

    #[test]
    fn define_questions_builds_typed_map_and_answers() {
        let questions = <Mixed as Questions>::questions();
        assert_eq!(
            serde_json::to_value(&questions).expect("serialize questions"),
            json!({
                "is_urgent": {"type": "noul", "instructions": "Urgent?"},
                "severity": {"type": "score", "instructions": "Severity?", "criteria": ["low", "high"]},
                "team": {"type": "choice", "instructions": "Team?",
                         "criteria": {"billing": null, "other": "none of these"}}
            })
        );
        let answers: Mixed = serde_json::from_value(json!({
            "is_urgent": {"type": "noul", "noul": 0.5},
            "severity": {"type": "score", "confidence": 0.5, "probabilities": {"0": 0.5, "1": 0.5}},
            "team": {"type": "choice", "choice": "billing", "confidence": 0.5,
                     "probabilities": {"billing": 1.0, "other": 0.0}}
        }))
        .expect("deserialize answers");
        assert!((answers.is_urgent.noul - 0.5).abs() < 1e-6);
        assert!((answers.severity.expected() - 0.5).abs() < 1e-6);
        assert_eq!(answers.team.choice, "billing");
    }

    #[test]
    fn define_questions_render_state_uses_template() {
        let noul = NoulOut::render_state("N-TEXT", "N-CTX").expect("render noul state");
        assert!(noul.contains("Input: N-TEXT"));
        assert!(noul.contains("Context: N-CTX"));
        let mixed = Mixed::render_state("M-TEXT", "M-CTX").expect("render mixed state");
        assert!(mixed.contains("Input: M-TEXT"));
        assert!(mixed.contains("Context: M-CTX"));
    }

    #[tokio::test]
    async fn evaluate_sends_expected_request_and_parses() {
        let transport = FakeTransport::new(vec![NOUL_RESPONSE.to_owned()]);
        let handle = client(transport.clone());
        let out: NoulOut = handle
            .evaluate::<NoulOut>(&json!({"message": "hello"}))
            .await
            .expect("evaluate");
        assert!((out.urgency.noul - 0.7).abs() < 1e-6);

        let request = transport.last_request();
        assert_eq!(request.url, "https://example.test/v1/systemone");
        assert_eq!(request.api_key, "secret");
        assert_eq!(
            request.body,
            json!({
                "model": "jev-test",
                "state": {"message": "hello"},
                "questions": {"urgency": {"type": "noul", "instructions": "Is it urgent?"}}
            })
        );
    }

    #[tokio::test]
    async fn evaluate_error_includes_raw_answers() {
        let transport = FakeTransport::new(vec![
            r#"{"answers":{"urgency":{"type":"noul","noul":"not-a-number"}}}"#.to_owned(),
        ]);
        let handle = client(transport);
        let error = handle
            .evaluate::<NoulOut>(&json!({}))
            .await
            .expect_err("parse must fail");
        let message = format!("{error:#}");
        assert!(
            message.contains("parse systemone answers"),
            "message: {message}"
        );
        assert!(message.contains("not-a-number"), "message: {message}");
    }

    #[tokio::test]
    async fn evaluate_propagates_transport_error() {
        let transport = FakeTransport::failing("connection reset");
        let handle = client(transport);
        let error = handle
            .evaluate::<NoulOut>(&json!({}))
            .await
            .expect_err("transport must fail");
        assert!(
            format!("{error:#}").contains("connection reset"),
            "error: {error:#}"
        );
    }

    #[tokio::test]
    async fn evaluate_applies_per_call_timeout() {
        let transport = Arc::new(SleepTransport {
            delay: Duration::from_millis(200),
        });
        let limits = ConcurrencyLimits {
            call_timeout: Duration::from_millis(10),
            ..ConcurrencyLimits::default()
        };
        let handle = SharedSystemOne::with_transport(
            transport,
            "https://example.test".to_owned(),
            "secret".to_owned(),
            "jev-test".to_owned(),
            limits,
        );
        let error = handle
            .evaluate::<NoulOut>(&json!({}))
            .await
            .expect_err("must time out");
        assert!(
            format!("{error:#}").contains("timed out"),
            "error: {error:#}"
        );
    }

    #[tokio::test]
    async fn evaluate_caps_concurrency() {
        let transport = ConcurrencyTransport::new(Duration::from_millis(50));
        let limits = ConcurrencyLimits {
            max_concurrent_calls: 2,
            ..ConcurrencyLimits::default()
        };
        let handle = Arc::new(SharedSystemOne::with_transport(
            Arc::new(transport.clone()),
            "https://example.test".to_owned(),
            "secret".to_owned(),
            "jev-test".to_owned(),
            limits,
        ));
        let mut tasks = Vec::new();
        for _ in 0..8 {
            let handle = handle.clone();
            tasks.push(tokio::spawn(async move {
                let _: NoulOut = handle
                    .evaluate::<NoulOut>(&json!({}))
                    .await
                    .expect("evaluate");
            }));
        }
        for task in tasks {
            task.await.expect("join");
        }
        assert_eq!(
            transport.max_seen(),
            2,
            "never more than 2 concurrent calls"
        );
    }

    #[tokio::test]
    async fn verify_round_trips_a_noul() {
        let transport = FakeTransport::new(vec![NOUL_RESPONSE.to_owned()]);
        let handle = client(transport.clone());
        handle.verify::<NoulOut>().await.expect("verify");
        assert_eq!(transport.call_count(), 1, "exactly one round-trip");
    }

    #[tokio::test]
    async fn verify_rejects_missing_answer() {
        let transport = FakeTransport::new(vec![r#"{"answers":{}}"#.to_owned()]);
        let handle = client(transport);
        assert!(handle.verify::<NoulOut>().await.is_err());
    }

    #[tokio::test]
    async fn evaluate_text_renders_state_and_parses() {
        let transport = FakeTransport::new(vec![NOUL_RESPONSE.to_owned()]);
        let handle = client(transport.clone());
        let out = handle
            .evaluate_text::<NoulOut>("hello", "ctx")
            .await
            .expect("evaluate_text");
        assert!((out.urgency.noul - 0.7).abs() < 1e-6);
        let body = transport.last_request().body;
        let state = body.get("state").expect("state field present");
        let state = state.as_str().expect("state is a string");
        assert!(state.contains("hello"), "state: {state}");
    }

    #[tokio::test]
    async fn verify_sends_string_state_from_template() {
        let transport = FakeTransport::new(vec![NOUL_RESPONSE.to_owned()]);
        let handle = client(transport.clone());
        handle.verify::<NoulOut>().await.expect("verify");
        let body = transport.last_request().body;
        let state = body.get("state").expect("state field present");
        assert!(
            state.is_string(),
            "state must be a JSON string, got: {state}"
        );
    }

    #[test]
    fn debug_redacts_api_key() {
        let handle = client(FakeTransport::new(vec![]));
        let rendered = format!("{handle:?}");
        assert!(rendered.contains("<redacted>"), "debug: {rendered}");
        assert!(!rendered.contains("secret"), "debug: {rendered}");
    }

    #[tokio::test]
    #[ignore = "live: requires TYPESAFE_API_KEY"]
    async fn live_verify() {
        let api_key = std::env::var("TYPESAFE_API_KEY").expect("TYPESAFE_API_KEY set");
        let base_url = std::env::var("TYPESAFE_BASE_URL")
            .unwrap_or_else(|_| "https://api.typesafe.ai".to_owned());
        let model =
            std::env::var("TYPESAFE_DEFAULT_MODEL").unwrap_or_else(|_| "jev-latest".to_owned());
        let handle = SharedSystemOne::new(base_url, api_key, model, ConcurrencyLimits::default());
        handle.verify::<NoulOut>().await.expect("live verify");
    }
}
