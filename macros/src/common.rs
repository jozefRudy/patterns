//! Shared codegen for the derive macros (`SystemOne`, `Extractable`).

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::spanned::Spanned;
use syn::{Error, Expr, Ident, Lit, LitStr, Meta, Result, Token};

/// Container args: `#[<derive>(template = "...", healthcheck = "...")]`.
#[derive(Debug)]
pub struct Container {
    pub template: LitStr,
    pub healthcheck: LitStr,
}

pub fn parse_container(attr: &syn::Attribute) -> Result<Container> {
    let mut template = None;
    let mut healthcheck = None;
    let metas =
        attr.parse_args_with(syn::punctuated::Punctuated::<Meta, Token![,]>::parse_terminated)?;
    for meta in metas {
        let Meta::NameValue(nv) = &meta else {
            return Err(Error::new(
                meta.span(),
                "expected `template = \"...\"` or `healthcheck = \"...\"`",
            ));
        };
        let Expr::Lit(expr) = &nv.value else {
            return Err(Error::new(nv.value.span(), "expected a string literal"));
        };
        let value = match &expr.lit {
            Lit::Str(s) => s.clone(),
            other => return Err(Error::new(other.span(), "expected a string literal")),
        };
        if nv.path.is_ident("template") {
            if template.is_some() {
                return Err(Error::new(nv.path.span(), "duplicate `template`"));
            }
            template = Some(value);
        } else if nv.path.is_ident("healthcheck") {
            if healthcheck.is_some() {
                return Err(Error::new(nv.path.span(), "duplicate `healthcheck`"));
            }
            healthcheck = Some(value);
        } else {
            return Err(Error::new(
                nv.path.span(),
                "unknown key; expected `template` or `healthcheck`",
            ));
        }
    }
    let template =
        template.ok_or_else(|| Error::new(attr.span(), "missing `template = \"...\"`"))?;
    let healthcheck =
        healthcheck.ok_or_else(|| Error::new(attr.span(), "missing `healthcheck = \"...\"`"))?;
    Ok(Container {
        template,
        healthcheck,
    })
}

/// Emit a `fn <fn_name>(<params>..) -> Result<String>` that declares the
/// askama input struct locally (inside the fn body — no module-level symbol to
/// collide with consumer types) and renders it.
pub fn render_fn(fn_name: &Ident, template: &LitStr, params: &[&str]) -> TokenStream {
    let params = params
        .iter()
        .map(|param| Ident::new(param, Span::call_site()))
        .collect::<Vec<_>>();
    quote! {
        fn #fn_name(#(#params: &str),*) -> ::anyhow::Result<String> {
            #[derive(::patterns::askama::Template)]
            #[template(path = #template, ext = "md", askama = ::patterns::askama)]
            struct Input<'a> {
                #(#params: &'a str,)*
            }

            use ::patterns::askama::Template as _;
            Input { #(#params),* }
                .render()
                .map_err(::core::convert::Into::into)
        }
    }
}

/// Emit `healthcheck_text` (renders the fixture through askama, so its path
/// resolves from the same configured dirs as `template`) + `verify` forwarding
/// used by both `Evaluatable` and `Extractable` healthcheck impls.
pub fn healthcheck_methods(healthcheck: &LitStr) -> TokenStream {
    quote! {
        fn healthcheck_text() -> ::anyhow::Result<::std::string::String> {
            #[derive(::patterns::askama::Template)]
            #[template(path = #healthcheck, ext = "md", askama = ::patterns::askama)]
            struct Healthcheck;

            use ::patterns::askama::Template as _;
            Healthcheck
                .render()
                .map_err(::core::convert::Into::into)
        }

        fn verify(&self) -> ::anyhow::Result<()> {
            Self::verify(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    fn attr(tokens: proc_macro2::TokenStream) -> syn::Attribute {
        use syn::parse::Parser as _;
        syn::Attribute::parse_outer
            .parse2(tokens)
            .expect("parse attribute")
            .pop()
            .expect("exactly one attribute")
    }

    #[test]
    fn parses_template_and_healthcheck() {
        let attr = attr(quote! { #[systemone(template = "t.md", healthcheck = "hc.md")] });
        let container = parse_container(&attr).expect("valid");
        assert_eq!(container.template.value(), "t.md");
        assert_eq!(container.healthcheck.value(), "hc.md");
    }

    #[test]
    fn rejects_missing_healthcheck() {
        let attr = attr(quote! { #[systemone(template = "t.md")] });
        let err = parse_container(&attr).expect_err("must fail");
        assert!(err.to_string().contains("missing `healthcheck"), "{err}");
    }

    #[test]
    fn rejects_missing_template() {
        let attr = attr(quote! { #[systemone(healthcheck = "hc.md")] });
        let err = parse_container(&attr).expect_err("must fail");
        assert!(err.to_string().contains("missing `template"), "{err}");
    }

    #[test]
    fn rejects_unknown_key() {
        let attr = attr(quote! { #[systemone(template = "t.md", boom = "x")] });
        let err = parse_container(&attr).expect_err("must fail");
        assert!(err.to_string().contains("unknown key"), "{err}");
    }

    #[test]
    fn rejects_duplicate_key() {
        let attr = attr(quote! { #[systemone(template = "a", template = "b")] });
        let err = parse_container(&attr).expect_err("must fail");
        assert!(err.to_string().contains("duplicate `template`"), "{err}");
    }

    #[test]
    fn rejects_non_string_value() {
        let attr = attr(quote! { #[systemone(template = 3)] });
        let err = parse_container(&attr).expect_err("must fail");
        assert!(err.to_string().contains("string literal"), "{err}");
    }

    #[test]
    fn healthcheck_methods_emit_render_and_verify() {
        let healthcheck: LitStr = syn::parse_quote!("hc.md");
        let tokens = healthcheck_methods(&healthcheck).to_string();
        assert!(tokens.contains("healthcheck_text"));
        assert!(tokens.contains("askama"));
        assert!(tokens.contains("fn verify"));
    }

    #[test]
    fn render_fn_emits_local_template_struct() {
        let fn_name = quote::format_ident!("render_state");
        let template: LitStr = syn::parse_quote!("t.md");
        let tokens = render_fn(&fn_name, &template, &["text", "prompt_context"]).to_string();
        assert!(tokens.contains("struct Input"), "{tokens}");
        assert!(tokens.contains("text"), "{tokens}");
        assert!(tokens.contains("prompt_context"), "{tokens}");
        assert!(tokens.contains("askama :: Template"), "{tokens}");
    }
}
