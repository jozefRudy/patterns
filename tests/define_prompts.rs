//! Compile-time + runtime check that `define_prompts!` generates a working
//! prompt enum: template derives compile, rendering substitutes all fields.

use patterns::define_prompts;

define_prompts! {
    (Test, TestPrompt, "test_fields.md"),
}

#[test]
fn test_define_prompts_renders_all_fields() {
    let prompt = PromptKind::Test
        .render_prompt("SCHEMA", "TEXT", "CONTEXT")
        .expect("template render failed");
    assert!(prompt.contains("JSON schema:\nSCHEMA"));
    assert!(prompt.contains("Context:\nCONTEXT"));
    assert!(prompt.contains("Input:\nTEXT"));
}
