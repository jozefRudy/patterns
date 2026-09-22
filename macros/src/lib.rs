//! `#[derive(SystemOne)]` — annotated question structs for
//! `patterns::systemone`.
//!
//! The consumer struct is the source of truth (mirroring
//! `#[derive(JsonSchema)]` + `#[schemars(description)]` in `llm_cli`): field
//! attributes declare each question, container attributes bind the state
//! template and the optional healthcheck fixture. The derive emits the askama
//! input template struct, `impl Questions` (`questions()` + `render_state`),
//! and — when `healthcheck` is present — `impl Evaluatable` (forwarding
//! `verify` to your inherent method of that name).
//!
//! ```ignore
//! #[derive(SystemOne, Debug, serde::Deserialize)]
//! #[systemone(
//!     template = "prompts/job_input.md",
//!     healthcheck = "Senior Rust dev, fully remote, EUR 80k-100k",
//! )]
//! struct JobAssessment {
//!     #[noul("Is the role fully remote, with no onsite or region restriction?")]
//!     is_remote: Noul,
//!     #[choice("Which seniority level does the posting target?",
//!              junior = "0-2 years", senior = "6+ years, leads work")]
//!     seniority: Choice,
//!     #[score("How strong is the match?", "Poor", "Weak", "Fair", "Strong", "Excellent")]
//!     match_score: Score,
//! }
//! ```
//!
//! Template paths resolve against the *consumer* crate's template dirs; the
//! template receives exactly `{{ text }}` and `{{ prompt_context }}`. Generated
//! code refers to `::patterns::` re-exports, so consumers never depend on
//! syn/quote/askama directly.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Error, Field, Fields, Ident, LitStr, Result, Token};

mod common;

use common::{healthcheck_methods, parse_container, render_fn, template_struct};

/// Derive `Questions` (and optionally `Evaluatable`) for an annotated struct.
#[proc_macro_derive(SystemOne, attributes(systemone, noul, choice, score))]
pub fn derive_systemone(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as DeriveInput);
    expand(&input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

// TODO(extractable-derive): replace this block with the real derive (Phase 3).
//
//   #[proc_macro_derive(Extractable, attributes(extract))]
//   pub fn derive_extractable(input: TokenStream) -> TokenStream { ... }
//
// Container attr: `#[extract(template = "...", healthcheck = "...")]` — same
// keys as `#[systemone(...)]`, so reuse `parse_container` (already generic).
// `healthcheck` is required here (Extractable::HEALTHCHECK_TEXT has no default).
//
// Emits (replaces `define_prompts!` + a hand-written `Extractable` impl):
//   #[derive(::patterns::askama::Template)]
//   #[template(path = <template>, ext = "md", askama = ::patterns::askama)]
//   struct <Name>Prompt<'a> { schema: &'a str, text: &'a str, prompt_context: &'a str }
//
//   impl ::patterns::llm_cli::Extractable for <Name> {
//       const HEALTHCHECK_TEXT: &'static str = <healthcheck literal>;
//       fn render_prompt(schema, text, prompt_context) -> anyhow::Result<String>
//       fn verify(&self) -> anyhow::Result<()> { Self::verify(self) }  // inherent
//   }

/// Derive `Extractable` for `#[extract(template = "...", healthcheck = "...")]`.
#[proc_macro_derive(Extractable, attributes(extract))]
pub fn derive_extractable(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as DeriveInput);
    expand_extractable(&input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn expand_extractable(input: &DeriveInput) -> Result<proc_macro2::TokenStream> {
    if !input.generics.params.is_empty() {
        return Err(Error::new(
            input.generics.span(),
            "#[derive(Extractable)] does not support generics",
        ));
    }
    if !matches!(input.data, Data::Struct(_)) {
        return Err(Error::new(
            input.span(),
            "#[derive(Extractable)] needs a struct",
        ));
    }
    let container_attr = input
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("extract"))
        .ok_or_else(|| {
            Error::new(
                Span::call_site(),
                "missing #[extract(template = \"...\", healthcheck = \"...\")] container attribute",
            )
        })?;
    let container = parse_container(container_attr)?;
    let healthcheck = container.healthcheck.as_ref().ok_or_else(|| {
        Error::new(
            container_attr.span(),
            "missing `healthcheck = \"...\"` (Extractable::HEALTHCHECK_TEXT has no default)",
        )
    })?;

    let name = &input.ident;
    let prompt_struct = format_ident!("{}Prompt", name);
    let template = &container.template;
    let template_item = template_struct(
        &prompt_struct,
        template,
        &["schema", "text", "prompt_context"],
    );
    let render_prompt = render_fn(
        &format_ident!("render_prompt"),
        &prompt_struct,
        &["schema", "text", "prompt_context"],
    );
    let methods = healthcheck_methods(healthcheck);

    Ok(quote! {
        #template_item

        impl ::patterns::llm_cli::Extractable for #name {
            #methods

            #render_prompt
        }
    })
}

/// `#[score("instructions", "level0", "level1", ...)]`.
#[derive(Debug)]
struct ScoreArgs {
    instructions: LitStr,
    levels: Vec<LitStr>,
}

/// One `#[choice]` option: bare `name` (undescribed) or `name = "rubric"`.
#[derive(Debug)]
struct ChoiceOpt {
    name: Ident,
    description: Option<LitStr>,
}

/// `#[choice("instructions", name, name = "rubric", ...)]`.
#[derive(Debug)]
struct ChoiceArgs {
    instructions: LitStr,
    options: Vec<ChoiceOpt>,
}

impl syn::parse::Parse for ScoreArgs {
    fn parse(input: syn::parse::ParseStream<'_>) -> Result<Self> {
        let instructions: LitStr = input.parse()?;
        let mut levels = Vec::new();
        while input.peek(Token![,]) {
            let _comma: Token![,] = input.parse()?;
            if input.is_empty() {
                break;
            }
            levels.push(input.parse::<LitStr>()?);
        }
        Ok(Self {
            instructions,
            levels,
        })
    }
}

impl syn::parse::Parse for ChoiceOpt {
    fn parse(input: syn::parse::ParseStream<'_>) -> Result<Self> {
        let name: Ident = input.parse()?;
        if input.peek(Token![=]) {
            let _eq: Token![=] = input.parse()?;
            let description: LitStr = input.parse()?;
            Ok(Self {
                name,
                description: Some(description),
            })
        } else {
            Ok(Self {
                name,
                description: None,
            })
        }
    }
}

impl syn::parse::Parse for ChoiceArgs {
    fn parse(input: syn::parse::ParseStream<'_>) -> Result<Self> {
        let instructions: LitStr = input.parse()?;
        let mut options = Vec::new();
        while input.peek(Token![,]) {
            let _comma: Token![,] = input.parse()?;
            if input.is_empty() {
                break;
            }
            options.push(input.parse::<ChoiceOpt>()?);
        }
        Ok(Self {
            instructions,
            options,
        })
    }
}

/// The question one field declares.
#[derive(Debug)]
enum FieldQuestion {
    Noul(LitStr),
    Score(ScoreArgs),
    Choice(ChoiceArgs),
}

impl FieldQuestion {
    /// The answer type this question requires on the field.
    const fn expected_type(&self) -> &'static str {
        match self {
            Self::Noul(_) => "Noul",
            Self::Score(_) => "Score",
            Self::Choice(_) => "Choice",
        }
    }
}

/// Extract the question attr from a field; error on none, duplicates, or
/// mismatched field type. Foreign attrs (serde, …) are ignored.
fn field_question(field: &Field) -> Result<FieldQuestion> {
    let mut question: Option<FieldQuestion> = None;
    for attr in &field.attrs {
        let parsed = if attr.path().is_ident("noul") {
            FieldQuestion::Noul(attr.parse_args::<LitStr>().map_err(|error| {
                Error::new(
                    attr.span(),
                    format!("expected #[noul(\"instructions\")]: {error}"),
                )
            })?)
        } else if attr.path().is_ident("score") {
            FieldQuestion::Score(attr.parse_args::<ScoreArgs>()?)
        } else if attr.path().is_ident("choice") {
            FieldQuestion::Choice(attr.parse_args::<ChoiceArgs>()?)
        } else {
            continue;
        };
        if question.is_some() {
            return Err(Error::new(
                attr.span(),
                "field has more than one question attribute",
            ));
        }
        question = Some(parsed);
    }
    let question = question.ok_or_else(|| {
        Error::new(
            field.span(),
            "field needs exactly one of #[noul(..)], #[choice(..)], #[score(..)]",
        )
    })?;
    if let Some(actual) = path_last_segment(&field.ty) {
        let expected = question.expected_type();
        if actual != expected {
            return Err(Error::new(
                field.ty.span(),
                format!("question kind needs field type `{expected}`, found `{actual}`"),
            ));
        }
    }
    Ok(question)
}

/// Last path segment of a type, if it is a plain path (`Noul`,
/// `patterns::systemone::Noul`, …).
fn path_last_segment(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    type_path
        .path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}

/// Build the `(id, Question)` map entry for one field, validating kind-specific
/// rules (score levels, choice options, duplicates).
fn build_entry(field: &Field) -> Result<proc_macro2::TokenStream> {
    let id = field
        .ident
        .as_ref()
        .ok_or_else(|| Error::new(field.span(), "field needs a name"))?;
    Ok(match &field_question(field)? {
        FieldQuestion::Noul(instructions) => quote! {
            (
                ::std::string::String::from(::std::stringify!(#id)),
                ::patterns::systemone::noul(#instructions),
            )
        },
        FieldQuestion::Score(args) => {
            if args.levels.len() < 2 {
                return Err(Error::new(
                    field.span(),
                    "#[score(..)] needs at least two levels",
                ));
            }
            let instructions = &args.instructions;
            let levels = &args.levels;
            quote! {
                (
                    ::std::string::String::from(::std::stringify!(#id)),
                    ::patterns::systemone::score(#instructions, [#(#levels),*]),
                )
            }
        }
        FieldQuestion::Choice(args) => {
            if args.options.len() < 2 {
                return Err(Error::new(
                    field.span(),
                    "#[choice(..)] needs at least two options",
                ));
            }
            let mut names: Vec<String> = Vec::new();
            for option in &args.options {
                let option_name = option.name.to_string();
                if names.contains(&option_name) {
                    return Err(Error::new(
                        option.name.span(),
                        format!("duplicate option `{option_name}`"),
                    ));
                }
                names.push(option_name);
            }
            let instructions = &args.instructions;
            let options = args.options.iter().map(|option| {
                let name = &option.name;
                let value = option.description.as_ref().map_or_else(
                    || quote! { ::core::option::Option::None },
                    |description| quote! { ::core::option::Option::Some(#description) },
                );
                quote! {
                    (::std::string::String::from(::std::stringify!(#name)), #value)
                }
            });
            quote! {
                (
                    ::std::string::String::from(::std::stringify!(#id)),
                    ::patterns::systemone::choice(#instructions, [#(#options),*]),
                )
            }
        }
    })
}

fn expand(input: &DeriveInput) -> Result<proc_macro2::TokenStream> {
    if !input.generics.params.is_empty() {
        return Err(Error::new(
            input.generics.span(),
            "#[derive(SystemOne)] does not support generics",
        ));
    }
    let container_attr = input
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("systemone"))
        .ok_or_else(|| {
            Error::new(
                Span::call_site(),
                "missing #[systemone(template = \"...\")] container attribute",
            )
        })?;
    let container = parse_container(container_attr)?;

    let Data::Struct(data) = &input.data else {
        return Err(Error::new(
            input.span(),
            "#[derive(SystemOne)] needs a struct",
        ));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(Error::new(
            input.span(),
            "#[derive(SystemOne)] needs named fields",
        ));
    };
    if fields.named.is_empty() {
        return Err(Error::new(
            input.span(),
            "#[derive(SystemOne)] needs fields",
        ));
    }

    let entries = fields
        .named
        .iter()
        .map(build_entry)
        .collect::<Result<Vec<_>>>()?;

    let name = &input.ident;
    let input_struct = format_ident!("{}Input", name);
    let template = &container.template;
    let template_item = template_struct(&input_struct, template, &["text", "prompt_context"]);
    let render_state = render_fn(
        &format_ident!("render_state"),
        &input_struct,
        &["text", "prompt_context"],
    );

    let healthcheck_impl = container.healthcheck.as_ref().map(|fixture| {
        let methods = healthcheck_methods(fixture);
        quote! {
            impl ::patterns::systemone::Evaluatable for #name {
                #methods
            }
        }
    });

    Ok(quote! {
        #template_item

        impl ::patterns::systemone::Questions for #name {
            type Answers = Self;

            fn questions() -> ::patterns::systemone::QuestionMap {
                ::patterns::systemone::QuestionMap::from([#(#entries),*])
            }

            #render_state
        }

        #healthcheck_impl
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_question_requires_attr() {
        let field: syn::Field = syn::parse_quote! { a: Noul };
        let err = field_question(&field).expect_err("must fail");
        assert!(err.to_string().contains("exactly one of"), "{err}");
    }

    #[test]
    fn field_question_rejects_type_mismatch() {
        let field: syn::Field = syn::parse_quote! { #[noul("q")] a: Score };
        let err = field_question(&field).expect_err("must fail");
        assert!(err.to_string().contains("needs field type `Noul`"), "{err}");
    }

    #[test]
    fn field_question_rejects_duplicate_attrs() {
        let field: syn::Field = syn::parse_quote! { #[noul("a")] #[noul("b")] a: Noul };
        let err = field_question(&field).expect_err("must fail");
        assert!(err.to_string().contains("more than one"), "{err}");
    }

    #[test]
    fn build_entry_rejects_short_score() {
        let field: syn::Field = syn::parse_quote! { #[score("s", "only")] a: Score };
        let err = build_entry(&field).expect_err("must fail");
        assert!(err.to_string().contains("at least two levels"), "{err}");
    }

    #[test]
    fn build_entry_rejects_short_choice() {
        let field: syn::Field = syn::parse_quote! { #[choice("c", only)] a: Choice };
        let err = build_entry(&field).expect_err("must fail");
        assert!(err.to_string().contains("at least two options"), "{err}");
    }

    #[test]
    fn build_entry_rejects_duplicate_option() {
        let field: syn::Field = syn::parse_quote! { #[choice("c", a, a)] a: Choice };
        let err = build_entry(&field).expect_err("must fail");
        assert!(err.to_string().contains("duplicate option"), "{err}");
    }

    #[test]
    fn build_entry_accepts_valid_choice() {
        let field: syn::Field = syn::parse_quote! { #[choice("c", a, b = "desc")] a: Choice };
        assert!(build_entry(&field).is_ok());
    }

    #[test]
    fn systemone_rejects_generics() {
        let input: DeriveInput = syn::parse_quote! {
            #[systemone(template = "t.md")]
            struct Probe<T> { #[noul("q")] a: Noul }
        };
        let err = expand(&input).expect_err("must fail");
        assert!(
            err.to_string().contains("does not support generics"),
            "{err}"
        );
    }

    #[test]
    fn systemone_rejects_tuple_struct() {
        let input: DeriveInput = syn::parse_quote! {
            #[systemone(template = "t.md")]
            struct Probe(Noul);
        };
        let err = expand(&input).expect_err("must fail");
        assert!(err.to_string().contains("named fields"), "{err}");
    }

    #[test]
    fn systemone_rejects_empty_struct() {
        let input: DeriveInput = syn::parse_quote! {
            #[systemone(template = "t.md")]
            struct Probe {}
        };
        let err = expand(&input).expect_err("must fail");
        assert!(err.to_string().contains("needs fields"), "{err}");
    }

    #[test]
    fn systemone_requires_container_attr() {
        let input: DeriveInput = syn::parse_quote! {
            struct Probe { #[noul("q")] a: Noul }
        };
        let err = expand(&input).expect_err("must fail");
        assert!(err.to_string().contains("missing #[systemone"), "{err}");
    }

    #[test]
    fn systemone_expands_valid() {
        let input: DeriveInput = syn::parse_quote! {
            #[systemone(template = "t.md", healthcheck = "hc")]
            struct Probe { #[noul("q")] a: Noul }
        };
        let tokens = expand(&input).expect("expand").to_string();
        assert!(tokens.contains("Questions"));
        assert!(tokens.contains("Evaluatable"));
    }

    #[test]
    fn extractable_requires_healthcheck() {
        let input: DeriveInput = syn::parse_quote! {
            #[extract(template = "t.md")]
            struct Probe { value: String }
        };
        let err = expand_extractable(&input).expect_err("must fail");
        assert!(err.to_string().contains("missing `healthcheck"), "{err}");
    }

    #[test]
    fn extractable_rejects_enum() {
        let input: DeriveInput = syn::parse_quote! {
            #[extract(template = "t.md", healthcheck = "hc")]
            enum Probe { A }
        };
        let err = expand_extractable(&input).expect_err("must fail");
        assert!(err.to_string().contains("needs a struct"), "{err}");
    }

    #[test]
    fn extractable_expands_valid() {
        let input: DeriveInput = syn::parse_quote! {
            #[extract(template = "t.md", healthcheck = "hc")]
            struct Probe { value: String }
        };
        let tokens = expand_extractable(&input).expect("expand").to_string();
        assert!(tokens.contains("Extractable"));
        assert!(tokens.contains("HEALTHCHECK_TEXT"));
    }
}
