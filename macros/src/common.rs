//! Shared codegen for the derive macros (`SystemOne`, `Extractable`).

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::spanned::Spanned;
use syn::{Error, Expr, Ident, Lit, LitStr, Meta, Result, Token};

/// Container args: `#[<derive>(template = "...", healthcheck = "...")]`.
#[derive(Debug)]
pub struct Container {
    pub template: LitStr,
    pub healthcheck: Option<LitStr>,
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
    Ok(Container {
        template,
        healthcheck,
    })
}

/// Emit the askama template struct with one `&str` field per slot.
pub fn template_struct(name: &Ident, template: &LitStr, slots: &[&str]) -> TokenStream {
    let slots = slots.iter().map(|slot| Ident::new(slot, Span::call_site()));
    quote! {
        #[derive(::patterns::askama::Template)]
        #[template(path = #template, ext = "md", askama = ::patterns::askama)]
        struct #name<'a> {
            #(#slots: &'a str,)*
        }
    }
}

/// Emit a `fn <fn_name>(<params>..) -> Result<String>` rendering `prompt_struct`.
pub fn render_fn(fn_name: &Ident, prompt_struct: &Ident, params: &[&str]) -> TokenStream {
    let params = params
        .iter()
        .map(|param| Ident::new(param, Span::call_site()))
        .collect::<Vec<_>>();
    quote! {
        fn #fn_name(#(#params: &str),*) -> ::anyhow::Result<String> {
            use ::patterns::askama::Template as _;
            #prompt_struct { #(#params),* }
                .render()
                .map_err(::core::convert::Into::into)
        }
    }
}

/// Emit the `HEALTHCHECK_TEXT` const + `verify` forwarding used by both
/// `Evaluatable` and `Extractable` healthcheck impls.
pub fn healthcheck_methods(healthcheck: &LitStr) -> TokenStream {
    quote! {
        const HEALTHCHECK_TEXT: &'static str = #healthcheck;

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
        let attr = attr(quote! { #[systemone(template = "t.md", healthcheck = "hc")] });
        let container = parse_container(&attr).expect("valid");
        assert_eq!(container.template.value(), "t.md");
        assert_eq!(
            container.healthcheck.as_ref().map(syn::LitStr::value),
            Some("hc".to_owned())
        );
    }

    #[test]
    fn healthcheck_is_optional() {
        let attr = attr(quote! { #[systemone(template = "t.md")] });
        let container = parse_container(&attr).expect("valid");
        assert!(container.healthcheck.is_none());
    }

    #[test]
    fn rejects_missing_template() {
        let attr = attr(quote! { #[systemone(healthcheck = "hc")] });
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
    fn healthcheck_methods_emit_const_and_verify() {
        let healthcheck: LitStr = syn::parse_quote!("hc");
        let tokens = healthcheck_methods(&healthcheck).to_string();
        assert!(tokens.contains("HEALTHCHECK_TEXT"));
        assert!(tokens.contains("fn verify"));
    }

    #[test]
    fn template_struct_emits_all_slots() {
        let name = quote::format_ident!("ProbePrompt");
        let template: LitStr = syn::parse_quote!("t.md");
        let tokens =
            template_struct(&name, &template, &["schema", "text", "prompt_context"]).to_string();
        assert!(tokens.contains("schema"));
        assert!(tokens.contains("prompt_context"));
        assert!(tokens.contains("askama :: Template"));
    }
}
