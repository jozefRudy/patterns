//! Consumer-style check that `#[derive(SystemOne)]` works from outside the
//! crate: the askama input template compiles and `render_state` substitutes
//! both fields; `questions()` builds the typed map; the healthcheck template
//! forwards to the inherent `verify`.

use patterns::SystemOne;
use patterns::systemone::{Choice, Noul, Questions, Score};

#[derive(SystemOne, Debug, serde::Deserialize)]
#[systemone(template = "test_input.md", healthcheck = "test_healthcheck.md")]
struct Test {
    #[noul("Is it urgent?")]
    urgent: Noul,
    #[score("Severity?", "low", "high")]
    severity: Score,
    #[choice("Team?", billing, other = "none of these")]
    team: Choice,
}

impl Test {
    fn verify(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.urgent.noul > 0.5, "not urgent");
        Ok(())
    }
}

#[test]
fn derive_renders_state_from_template() {
    let state = Test::render_state("TEXT", "CONTEXT").expect("render_state failed");
    assert!(state.contains("Input: TEXT"));
    assert!(state.contains("Context: CONTEXT"));
}

#[test]
fn derive_builds_typed_questions() {
    let questions = Test::questions();
    assert_eq!(
        serde_json::to_value(&questions).expect("serialize questions"),
        serde_json::json!({
            "urgent": {"type": "noul", "instructions": "Is it urgent?"},
            "severity": {"type": "score", "instructions": "Severity?", "criteria": ["low", "high"]},
            "team": {"type": "choice", "instructions": "Team?",
                     "criteria": {"billing": null, "other": "none of these"}}
        })
    );
}

#[test]
fn derive_parses_typed_answers() {
    let answers: Test = serde_json::from_value(serde_json::json!({
        "urgent": {"type": "noul", "noul": 0.9},
        "severity": {"type": "score", "confidence": 0.5, "probabilities": {"0": 0.5, "1": 0.5}},
        "team": {"type": "choice", "choice": "billing", "confidence": 0.7,
                 "probabilities": {"billing": 0.7, "other": 0.3}}
    }))
    .expect("deserialize answers");
    assert!((answers.urgent.noul - 0.9).abs() < 1e-6);
    assert!((answers.severity.expected() - 0.5).abs() < 1e-6);
    assert_eq!(answers.team.choice, "billing");
    assert!((answers.team.confidence - 0.7).abs() < 1e-6);
}
