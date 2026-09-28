//! Bakes crate versions into the model identity: the tokenizer (which decides the token stream) is
//! part of `id()`; the rest are recorded for inspection only, so an execution-crate bump does not
//! force a re-embed. Reads `Cargo.lock` because versions are not available at runtime.

use std::env;
use std::fs;
use std::path::Path;

/// Execution crates: cannot change the vector beyond float-level noise (inspection only).
const ENGINE: [&str; 3] = ["fastembed", "ort", "ndarray"];

/// The one engine crate in `id()`: a bump can change the token stream, not just float bits.
const TOKENIZER: &str = "tokenizers";

fn main() {
    println!("cargo:rerun-if-changed=Cargo.lock");
    let manifest = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo");
    let lock = fs::read_to_string(Path::new(&manifest).join("Cargo.lock"))
        .expect("Cargo.lock sits next to Cargo.toml");
    println!(
        "cargo:rustc-env=PATTERNS_ENGINE_FINGERPRINT={}",
        fingerprint(&lock, &ENGINE)
    );
    println!(
        "cargo:rustc-env=PATTERNS_TOKENIZER_FINGERPRINT={}",
        fingerprint(&lock, &[TOKENIZER])
    );
}

/// `name=version@checksum8` per name, joined by `;`.
fn fingerprint(lock: &str, names: &[&str]) -> String {
    names
        .iter()
        .map(|name| {
            let (version, checksum) = package(lock, name);
            let short = checksum.map_or_else(String::new, |hash| hash.chars().take(8).collect());
            format!("{name}={version}@{short}")
        })
        .collect::<Vec<_>>()
        .join(";")
}

/// `(version, checksum)` for `name`, from its `[[package]]` block (empty when absent).
fn package(lock: &str, name: &str) -> (String, Option<String>) {
    for block in lock.split("[[package]]") {
        let mut block_name = String::new();
        let mut version = String::new();
        let mut checksum = None;
        for line in block.lines().map(str::trim) {
            if let Some(value) = field(line, "name") {
                block_name = value;
            } else if let Some(value) = field(line, "version") {
                version = value;
            } else if let Some(value) = field(line, "checksum") {
                checksum = Some(value);
            }
        }
        if block_name == name {
            return (version, checksum);
        }
    }
    (String::new(), None)
}

/// Value of a `key = "value"` lockfile line, when `line` is that key.
fn field(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?.strip_prefix(" = ")?;
    Some(rest.trim_matches('"').to_string())
}
