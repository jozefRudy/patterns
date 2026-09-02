//! Personal pattern library: reusable, strictly-linted building blocks
//! shared across projects via a pinned git dependency.

pub mod llm_cli;
// Reserved: pub mod lance_store;

/// Re-exported so `define_prompts!` consumers don't need own askama/paste deps.
pub use askama;
#[doc(hidden)]
pub use pastey;

/// Define an extraction-prompt enum backed by askama templates.
///
/// Template paths resolve against the *consumer* crate's template dirs
/// (its `askama.toml` / `templates/`).
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
