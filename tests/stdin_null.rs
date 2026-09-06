//! Child stdin must be /dev/null, not inherited.
//!
//! Parents launched detached (nohup, systemd) can hold a CLOSED fd 0 — with
//! inherited stdin, node-based CLIs crash with `EBADF: bad file descriptor,
//! read` on their stdin `ReadStream` (seen live with `pi --print` under a
//! nohup'd poller).
//!
//! This test closes fd 0 in its own process (a tests/ binary — separate from
//! the lib test process, so no other tests are affected) and runs one
//! extraction against a fake LLM CLI ("pi") that drains stdin: `head -c 0` succeeds on
//! a readable-to-EOF stdin (/dev/null) and fails with EBADF on a closed one.

use patterns::llm_cli::{Extractable, SharedLimits, SharedLlm};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Dummy {
    value: String,
}

impl Extractable for Dummy {
    const HEALTHCHECK_TEXT: &'static str = "healthcheck";

    fn render_prompt(schema: &str, text: &str, _ctx: &str) -> anyhow::Result<String> {
        Ok(format!("{schema}\n{text}"))
    }

    fn verify(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn child_stdin_is_null_not_inherited_when_parent_fd0_closed() {
    /// restores fd 0 on scope exit — even on assertion failure
    struct Guard(i32);
    impl Drop for Guard {
        fn drop(&mut self) {
            unsafe {
                libc::dup2(self.0, 0);
                libc::close(self.0);
            }
        }
    }
    // save fd 0, close it, restore on exit — even on assertion failure
    let saved = unsafe { libc::dup(0) };
    assert!(saved >= 0, "dup fd 0");
    unsafe { libc::close(0) };
    let _g = Guard(saved);

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fake.sh");
    // `head -c 0` drains stdin to EOF: /dev/null → exit 0, closed fd → EBADF
    std::fs::write(&path, "#!/bin/sh\nhead -c 0 && echo '{\"value\":\"ok\"}'\n").expect("write");
    // `sh script` stands in for the real LLM CLI (pi): any child works, the
    // test asserts stdin behavior, not LLM behavior.
    let llm = SharedLlm::new(
        "sh".to_owned(),
        vec![path.display().to_string()],
        SharedLimits::default(),
    );
    let d: Dummy = llm
        .extract::<Dummy>("text", String::new())
        .await
        .expect("extraction with parent fd 0 closed must still work (child stdin = /dev/null)");
    assert_eq!(d.value, "ok");
}
