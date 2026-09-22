//! Compile-time + runtime check that `define_questions!` works from a consumer
//! crate: the askama template derives compile and `render_state` substitutes
//! both fields.

use patterns::define_questions;
use patterns::systemone::Questions;

define_questions! {
    Test: "test_input.md" {
        urgent:   noul("Is it urgent?"),
        severity: score("Severity?", ["low", "high"]),
        team:     choice("Team?", ["billing", ("other", "none of these")]),
    }
}

#[test]
fn test_define_questions_renders_state() {
    let state = Test::render_state("TEXT", "CONTEXT").expect("render_state failed");
    assert!(state.contains("Input: TEXT"));
    assert!(state.contains("Context: CONTEXT"));
}
