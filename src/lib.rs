//! Personal pattern library: reusable, strictly-linted building blocks
//! shared across projects via a pinned git dependency.

/// Re-exported so macro-generated derives resolve without a direct `serde` dep.
pub use serde;

#[cfg(feature = "embed")]
pub mod embed;
#[cfg(feature = "language")]
pub mod language;
pub mod limits;
#[cfg(feature = "llm_cli")]
pub mod llm_cli;
#[cfg(feature = "systemone")]
pub mod systemone;
// Reserved: pub mod lance_store;

/// Re-exported so embed consumers don't declare fastembed/ort separately
/// (single version, enforced).
#[cfg(feature = "embed")]
pub use fastembed;
#[cfg(feature = "embed")]
pub use ort;

// Guard: the ONNX Runtime C API level `ort` asks for is the *union* of every
// crate's `api-*` features — the highest one wins, and features can only be
// added, never subtracted. This crate links the system onnxruntime (via
// `ORT_LIB_LOCATION`) instead of ort's bundled one, so a silent bump past its
// supported API version only fails at *runtime* (`GetApi` returns null ->
// panic). Assert the known-good level here so any dependency bump fails the
// build instead.
#[cfg(feature = "embed")]
const _: () = assert!(
    ort::sys::ORT_API_VERSION == 24,
    "ort API level changed: re-check the linked onnxruntime version (deploy pins 1.26, which supports API <= 26)"
);

/// Re-exported so `define_prompts!`/`define_questions!` consumers don't need
/// own askama/paste deps.
#[cfg(any(feature = "llm_cli", feature = "systemone"))]
pub use askama;
#[cfg(any(feature = "llm_cli", feature = "systemone"))]
#[doc(hidden)]
pub use pastey;

/// Define an extraction-prompt enum backed by askama templates.
///
/// Template paths resolve against the *consumer* crate's template dirs
/// (its `askama.toml` / `templates/`).
#[cfg(feature = "llm_cli")]
#[macro_export]
macro_rules! define_prompts {
    ($(($variant:ident, $path:literal)),* $(,)?) => {
        #[derive(Copy, Clone, Debug)]
        pub enum PromptKind {
            $($variant,)*
        }

        $crate::pastey::paste! {
            $(
                #[derive($crate::askama::Template)]
                #[template(path = $path, ext = "md", askama = $crate::askama)]
                struct [<$variant Prompt>]<'a> {
                    schema: &'a str,
                    text: &'a str,
                    prompt_context: &'a str,
                }

                impl<'a> [<$variant Prompt>]<'a> {
                    fn render_prompt(
                        schema: &'a str,
                        text: &'a str,
                        prompt_context: &'a str,
                    ) -> ::anyhow::Result<String> {
                        use $crate::askama::Template;
                        Self { schema, text, prompt_context }
                            .render()
                            .map_err(Into::into)
                    }
                }
            )*
        }

        impl PromptKind {
            pub fn render_prompt(
                self,
                schema: &str,
                text: &str,
                prompt_context: &str,
            ) -> ::anyhow::Result<String> {
                $crate::pastey::paste! {
                    match self {
                        $(Self::$variant => [<$variant Prompt>]::render_prompt(schema, text, prompt_context),)*
                    }
                }
            }
        }
    };
}
