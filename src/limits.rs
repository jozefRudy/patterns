//! Shared concurrency/time limits for bounded gateways (`llm_cli`, `systemone`).

use std::time::Duration;

/// Default per-call timeout for a gateway.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default cap on concurrent calls across a process.
pub const DEFAULT_MAX_CONCURRENT_CALLS: usize = 2;

/// Limits for a shared gateway: how many calls may run at once, and how long
/// each may take.
///
/// App-specific policy is passed in by the consumer; see `DEFAULT_*` constants
/// for the generic defaults. Feature-specific limits (e.g. prompt text length)
/// live with their feature, not here.
#[derive(Debug, Clone)]
pub struct ConcurrencyLimits {
    /// Cap on concurrent calls across all holders of the handle.
    pub max_concurrent_calls: usize,
    /// Per-call timeout.
    pub call_timeout: Duration,
}

impl Default for ConcurrencyLimits {
    fn default() -> Self {
        Self {
            max_concurrent_calls: DEFAULT_MAX_CONCURRENT_CALLS,
            call_timeout: DEFAULT_TIMEOUT,
        }
    }
}
