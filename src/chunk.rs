//! Token-window chunking shared by the in-process `embed` and remote
//! `embed_api` backends.
//!
//! Pure over a pre-tokenized span list, so both backends chunk identically.
//! Lives outside `embed` (feature-gated) so the remote client reuses it;
//! re-exported as `patterns::embed::*` for back-compat.

use anyhow::Result;
use tokenizers::Tokenizer;

/// Byte span of a single token within its source text.
pub(crate) type TokenSpan = std::ops::Range<usize>;

/// Tokens the tokenizer adds around every input ([CLS]/[SEP]) — excluded from
/// chunk content so the model never truncates a chunk we built.
pub(crate) const SPECIAL_TOKEN_HEADROOM: usize = 2;

/// Chunking policy for document embedding (model-agnostic limits).
#[derive(Clone, Copy, Debug)]
pub struct ChunkOptions {
    /// Content tokens per chunk, excluding special tokens ([CLS]/[SEP]).
    pub max_tokens: usize,
    /// Token overlap between consecutive chunks.
    pub overlap_tokens: usize,
    /// Texts below this token count are not worth embedding.
    pub min_tokens: usize,
}

impl ChunkOptions {
    /// Options at `max_tokens` with the default overlap/minimum.
    #[must_use]
    pub const fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            overlap_tokens: 64,
            min_tokens: 5,
        }
    }
}

/// A slice of a source text, aligned to token boundaries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextChunk {
    pub text: String,
    pub tokens: usize,
    /// `source[byte_start..byte_end]`, outer whitespace trimmed.
    pub byte_start: usize,
    pub byte_end: usize,
}

/// One embedded chunk — a storage row: `(doc_ix → content id, chunk_ix, vector)`.
#[derive(Clone, Debug)]
pub struct EmbeddedChunk {
    /// Index of the source text in the batch's input slice (0 for single-doc calls).
    pub doc_ix: usize,
    /// Position of this chunk within its document.
    pub chunk_ix: usize,
    pub chunk: TextChunk,
    pub embedding: Vec<f32>,
}

/// Byte span of every token in `text`, empty spans filtered.
///
/// The tokenizer must already have truncation disabled (callers clone the
/// shared instance and `with_truncation(None)` it first) so counts aren't
/// silently capped.
pub(crate) fn token_spans(tokenizer: &Tokenizer, text: &str) -> Result<Vec<TokenSpan>> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    Ok(encoding
        .get_offsets()
        .iter()
        .filter(|(start, end)| start != end)
        .map(|(start, end)| *start..*end)
        .collect())
}

/// Whitespace-word spans — the `Fake` embedder's documented token approximation.
#[cfg(feature = "embed")]
pub(crate) fn fake_token_spans(text: &str) -> Vec<TokenSpan> {
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    for (ix, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(s) = start.take() {
                spans.push(s..ix);
            }
        } else if start.is_none() {
            start = Some(ix);
        }
    }
    if let Some(s) = start {
        spans.push(s..text.len());
    }
    spans
}

/// Split pre-tokenized `spans` into chunks of at most `opts.max_tokens`,
/// preferring paragraph, then sentence, then whitespace boundaries.
///
/// Invariant: every non-whitespace byte of `text` is inside exactly one chunk
/// span's coverage (no tail loss); consecutive chunks overlap by at most
/// `opts.overlap_tokens`.
pub(crate) fn chunk_spans(text: &str, spans: &[TokenSpan], opts: &ChunkOptions) -> Vec<TextChunk> {
    let max = opts.max_tokens.max(1);
    let mut out = Vec::new();
    if spans.is_empty() {
        return out;
    }
    let mut start = 0_usize;
    while start < spans.len() {
        let remaining = spans.len() - start;
        if remaining <= max {
            if let Some(chunk) = make_chunk(text, spans, start, spans.len()) {
                out.push(chunk);
            }
            break;
        }
        let end = start + max;
        let break_tok = find_break(text, spans, start, end);
        let break_tok = if break_tok > start { break_tok } else { end };
        if let Some(chunk) = make_chunk(text, spans, start, break_tok) {
            out.push(chunk);
        }
        let next = break_tok.saturating_sub(opts.overlap_tokens).max(start + 1);
        start = next;
    }
    out
}

/// Last natural break inside `spans[start..end]`, as a token index.
fn find_break(text: &str, spans: &[TokenSpan], start: usize, end: usize) -> usize {
    let (Some(first), Some(last)) = (spans.get(start), spans.get(end.saturating_sub(1))) else {
        return end;
    };
    let Some(window) = text.get(first.start..last.end) else {
        return end;
    };
    let break_pos = ["\n\n", ". ", "! ", "? "]
        .iter()
        .filter_map(|sep| window.rfind(sep).map(|ix| ix + sep.len()))
        .max()
        .or_else(|| window.rfind(char::is_whitespace));
    let Some(pos) = break_pos else {
        return end;
    };
    let abs = first.start + pos;
    let ix = spans.get(start..end).map_or(0, |window| {
        window.iter().take_while(|span| span.end <= abs).count()
    });
    start + ix
}

fn make_chunk(text: &str, spans: &[TokenSpan], start: usize, end: usize) -> Option<TextChunk> {
    let raw_start = spans.get(start)?.start;
    let raw_end = spans.get(end.checked_sub(1)?)?.end;
    let (byte_start, byte_end) = trim_span(text, raw_start, raw_end);
    if byte_start >= byte_end {
        return None;
    }
    Some(TextChunk {
        text: text.get(byte_start..byte_end)?.to_string(),
        tokens: end - start,
        byte_start,
        byte_end,
    })
}

/// Trim ASCII/Unicode whitespace from a byte span.
fn trim_span(text: &str, start: usize, end: usize) -> (usize, usize) {
    let slice = text.get(start..end).unwrap_or_default();
    let trimmed = slice.trim();
    if trimmed.is_empty() {
        return (start, start);
    }
    let lead = slice.len() - slice.trim_start().len();
    let new_start = start + lead;
    (new_start, new_start + trimmed.len())
}
