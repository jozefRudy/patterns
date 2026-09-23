//! Consumer-style check that `#[derive(Extractable)]` works from outside the
//! crate: the askama prompt template compiles, `render_prompt` substitutes all
//! three slots, and `healthcheck` renders the fixture into
//! `healthcheck_text`.

use patterns::Extractable;
use patterns::llm_cli::Extractable as _;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema, Extractable)]
#[extract(template = "test_fields.md", healthcheck = "test_healthcheck.md")]
struct Test {
    value: String,
}

impl Test {
    fn verify(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.value.is_empty(), "empty value");
        Ok(())
    }
}

#[test]
fn derive_renders_prompt_with_all_slots() {
    let prompt = Test::render_prompt("SCHEMA", "TEXT", "CONTEXT").expect("render_prompt failed");
    assert!(prompt.contains("JSON schema:\nSCHEMA"));
    assert!(prompt.contains("Context:\nCONTEXT"));
    assert!(prompt.contains("Input:\nTEXT"));
}

#[test]
fn derive_exposes_healthcheck_text() {
    assert_eq!(
        Test::healthcheck_text().expect("render").trim(),
        "Healthcheck text."
    );
}
