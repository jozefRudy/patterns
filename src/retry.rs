//! Shared transient-failure retry policy for bounded HTTP gateways
//! (`embed_api`, `systemone`).

use anyhow::Result;
use std::future::Future;
use std::time::Duration;
use tokio::time::timeout;

/// HTTP status + body, returned for both success and error so a [`RetryPolicy`]
/// can classify transient failures.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// HTTP status code.
    pub status: u16,
    /// Response body (used verbatim in error messages).
    pub body: String,
}

/// Transient-failure retry policy for a bounded HTTP gateway.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Total attempts including the first; `1` disables retries.
    pub max_attempts: usize,
    /// First backoff, doubled per attempt.
    pub base_backoff: Duration,
    /// Cap on backoff.
    pub max_backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_backoff: Duration::from_millis(250),
            max_backoff: Duration::from_secs(2),
        }
    }
}

impl RetryPolicy {
    /// No retries — a single attempt.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            max_attempts: 1,
            base_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
        }
    }

    /// Backoff before the given 1-based attempt: `base * 2^(attempt-1)`, capped.
    #[must_use]
    pub fn backoff(&self, attempt: usize) -> Duration {
        let shift = attempt.saturating_sub(1).min(31);
        let factor = 1u32 << shift;
        self.base_backoff
            .saturating_mul(factor)
            .min(self.max_backoff)
    }

    /// Whether an HTTP status is transient and worth retrying (429, 5xx).
    #[must_use]
    pub fn is_transient(status: u16) -> bool {
        status == 429 || (500..600).contains(&status)
    }

    /// Run `call` until it yields a 2xx [`HttpResponse`] or the attempt budget
    /// is exhausted.
    ///
    /// Each attempt is bounded by `call_timeout`. Transient outcomes (per
    /// [`Self::is_transient`]) and `Err` results (including a timed-out attempt)
    /// are retried with [`Self::backoff`]; any other non-2xx fails immediately.
    /// `label` prefixes error messages.
    pub async fn run<F, Fut>(
        &self,
        label: &str,
        call_timeout: Duration,
        mut call: F,
    ) -> Result<HttpResponse>
    where
        F: FnMut() -> Fut + Send,
        Fut: Future<Output = Result<HttpResponse>> + Send,
    {
        let mut attempt = 0usize;
        loop {
            attempt += 1;
            match timeout(call_timeout, call()).await {
                Ok(Ok(response)) if (200..300).contains(&response.status) => return Ok(response),
                Ok(Ok(response)) if Self::is_transient(response.status) => {
                    if attempt >= self.max_attempts {
                        anyhow::bail!(
                            "{label} returned HTTP {} after {attempt} attempts: {}",
                            response.status,
                            response.body
                        );
                    }
                }
                Ok(Ok(response)) => {
                    anyhow::bail!(
                        "{label} returned HTTP {}: {}",
                        response.status,
                        response.body
                    );
                }
                Ok(Err(error)) => {
                    if attempt >= self.max_attempts {
                        return Err(error.context(format!("{label} transport failed")));
                    }
                }
                Err(_elapsed) => {
                    if attempt >= self.max_attempts {
                        anyhow::bail!("{label} call timed out after {attempt} attempts");
                    }
                }
            }
            tokio::time::sleep(self.backoff(attempt)).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn instant(attempts: usize) -> RetryPolicy {
        RetryPolicy {
            max_attempts: attempts,
            base_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
        }
    }

    #[test]
    fn default_is_three_attempts_with_backoff() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_attempts, 3);
        assert_eq!(policy.base_backoff, Duration::from_millis(250));
        assert_eq!(policy.max_backoff, Duration::from_secs(2));
    }

    #[test]
    fn none_has_single_attempt() {
        assert_eq!(RetryPolicy::none().max_attempts, 1);
    }

    #[test]
    fn backoff_doubles_and_caps() {
        let policy = RetryPolicy {
            max_attempts: 10,
            base_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(500),
        };
        assert_eq!(policy.backoff(1), Duration::from_millis(100));
        assert_eq!(policy.backoff(2), Duration::from_millis(200));
        assert_eq!(policy.backoff(3), Duration::from_millis(400));
        assert_eq!(policy.backoff(4), Duration::from_millis(500));
        assert_eq!(policy.backoff(100), Duration::from_millis(500));
    }

    #[test]
    fn classifies_transient_statuses() {
        assert!(RetryPolicy::is_transient(429));
        assert!(RetryPolicy::is_transient(500));
        assert!(RetryPolicy::is_transient(503));
        assert!(!RetryPolicy::is_transient(400));
        assert!(!RetryPolicy::is_transient(404));
    }

    #[tokio::test]
    async fn run_retries_transient_then_succeeds() {
        let calls = AtomicUsize::new(0);
        let response = instant(3)
            .run("test", Duration::from_secs(1), || {
                let previous = calls.fetch_add(1, Ordering::SeqCst);
                async move {
                    if previous == 0 {
                        Ok(HttpResponse {
                            status: 503,
                            body: "busy".to_owned(),
                        })
                    } else {
                        Ok(HttpResponse {
                            status: 200,
                            body: "ok".to_owned(),
                        })
                    }
                }
            })
            .await
            .expect("succeeds");
        assert_eq!(response.status, 200);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn run_fails_fast_on_payload_status() {
        let calls = AtomicUsize::new(0);
        let error = instant(3)
            .run("test", Duration::from_secs(1), || {
                calls.fetch_add(1, Ordering::SeqCst);
                async {
                    Ok(HttpResponse {
                        status: 400,
                        body: "bad".to_owned(),
                    })
                }
            })
            .await
            .expect_err("must fail");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "no retry on 4xx");
        assert!(format!("{error:#}").contains("400"), "error: {error:#}");
    }

    #[tokio::test]
    async fn run_exhausts_attempts_on_persistent_transient() {
        let calls = AtomicUsize::new(0);
        let error = instant(2)
            .run("test", Duration::from_secs(1), || {
                calls.fetch_add(1, Ordering::SeqCst);
                async {
                    Ok(HttpResponse {
                        status: 500,
                        body: "err".to_owned(),
                    })
                }
            })
            .await
            .expect_err("must fail");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(
            format!("{error:#}").contains("after 2 attempts"),
            "error: {error:#}"
        );
    }

    #[tokio::test]
    async fn run_reports_timeout_separately_from_transport() {
        let error = instant(1)
            .run("test", Duration::from_millis(1), || async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(HttpResponse {
                    status: 200,
                    body: String::new(),
                })
            })
            .await
            .expect_err("must time out");
        assert!(
            format!("{error:#}").contains("timed out"),
            "error: {error:#}"
        );
    }

    #[tokio::test]
    async fn run_contextualizes_transport_error() {
        let error = instant(1)
            .run("test", Duration::from_secs(1), || async {
                Err(anyhow::anyhow!("connection reset"))
            })
            .await
            .expect_err("must fail");
        let message = format!("{error:#}");
        assert!(
            message.contains("test transport failed"),
            "error: {message}"
        );
        assert!(message.contains("connection reset"), "error: {message}");
    }
}
