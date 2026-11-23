//! Shared concurrency/time limits for bounded gateways (`llm`/`embed_api`, `systemone`).

use std::time::Duration;

/// Default per-call timeout for a gateway.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default cap on concurrent calls for one shared gateway handle.
pub const DEFAULT_MAX_CONCURRENT_CALLS: usize = 2;

/// Limits for one shared gateway handle: how many calls it may run at once, and
/// how long each may take.
///
/// The cap belongs to the handle, not the process: every gateway instance builds
/// its own semaphore, so two handles (say `llm` and `systemone`) each get their
/// own budget. Clones of one handle share its semaphore, so pass the same handle
/// around rather than building one per task.
///
/// App-specific policy is passed in by the consumer; see `DEFAULT_*` constants
/// for the generic defaults. Feature-specific limits (e.g. prompt text length)
/// live with their feature, not here.
#[derive(Debug, Clone)]
pub struct ConcurrencyLimits {
    /// Cap on concurrent calls across all clones of one handle.
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
