//! Query/document prefixes for asymmetric embedding models.
//!
//! Shared by the in-process `embed` and remote `embed_api` backends so both
//! apply the same formatting. The prefixes are **model configuration** — a
//! fixed prepend chosen once per corpus (e.g. `"search_query: "` /
//! `"search_document: "`, or Qwen's `"Instruct: <task>\nQuery: "` / `""`).

/// Model's query/document prefixes (empty strings when the model uses none).
#[derive(Clone, Debug)]
pub struct Prefixes {
    /// Prepended to queries, e.g. `"search_query: "`.
    pub query: String,
    /// Prepended to documents, e.g. `"search_document: "`.
    pub document: String,
}

impl Prefixes {
    /// No prefixes (symmetric models, e.g. BGE-M3).
    #[must_use]
    pub const fn none() -> Self {
        Self {
            query: String::new(),
            document: String::new(),
        }
    }

    /// Explicit query and document prefixes.
    #[must_use]
    pub fn new(query: impl Into<String>, document: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            document: document.into(),
        }
    }

    /// Query prefix only; documents are embedded raw (e.g. Qwen3-Embedding).
    #[must_use]
    pub fn query_only(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            document: String::new(),
        }
    }
}
