//! Bakes crate versions into the model identity: the tokenizer (which decides the token stream) is
//! part of `id()`; the rest are recorded for inspection only, so an execution-crate bump does not
//! force a re-embed.
//!
//! Reads `Cargo.toml`, not `Cargo.lock`: a vendored dependency (cargo vendor, nix/crane) ships the
//! manifest but never a lockfile, so a lockfile read fails every sandboxed consumer build. The
//! engine/tokenizer deps are exact-pinned (`=x.y.z`) in the manifest, so the declared version
//! equals the resolved one — deterministic without a lockfile.

use std::env;
use std::fs;
use std::path::Path;

/// Execution crates: cannot change the vector beyond float-level noise (inspection only).
const ENGINE: [&str; 2] = ["fastembed", "ort"];

/// The one engine crate in `id()`: a bump can change the token stream, not just float bits.
const TOKENIZER: &str = "tokenizers";

fn main() -> Result<(), String> {
    println!("cargo:rerun-if-changed=Cargo.toml");
    let manifest_dir =
        env::var("CARGO_MANIFEST_DIR").map_err(|err| format!("CARGO_MANIFEST_DIR: {err}"))?;
    let text = fs::read_to_string(Path::new(&manifest_dir).join("Cargo.toml"))
        .map_err(|err| format!("read Cargo.toml: {err}"))?;
    let manifest: toml::Value =
        toml::from_str(&text).map_err(|err| format!("parse Cargo.toml: {err}"))?;
    println!(
        "cargo:rustc-env=PATTERNS_ENGINE_FINGERPRINT={}",
        fingerprint(&manifest, &ENGINE)?
    );
    println!(
        "cargo:rustc-env=PATTERNS_TOKENIZER_FINGERPRINT={}",
        fingerprint(&manifest, &[TOKENIZER])?
    );
    Ok(())
}

/// `name=version` per name, joined by `;`.
fn fingerprint(manifest: &toml::Value, names: &[&str]) -> Result<String, String> {
    names
        .iter()
        .map(|name| Ok(format!("{name}={}", exact_version(manifest, name)?)))
        .collect::<Result<Vec<_>, String>>()
        .map(|parts| parts.join(";"))
}

/// The exact version declared for `dependencies.<name>` (a leading `=` is fine).
///
/// Errors unless the crate is a direct dependency with an exact (`x.y.z`) version:
/// a caret/range would not fix the resolved crate, so the identity would not be
/// reproducible.
fn exact_version(manifest: &toml::Value, name: &str) -> Result<String, String> {
    let dep = manifest
        .get("dependencies")
        .and_then(|deps| deps.get(name))
        .ok_or_else(|| format!("`{name}` must be a direct dependency"))?;
    let declared = match dep {
        toml::Value::String(version) => version.clone(),
        toml::Value::Table(table) => table
            .get("version")
            .and_then(toml::Value::as_str)
            .ok_or_else(|| format!("`{name}` needs an explicit `version`"))?
            .to_string(),
        other => {
            return Err(format!(
                "`{name}` has an unsupported dependency form: {other:?}"
            ));
        }
    };
    let version = declared.strip_prefix('=').unwrap_or(&declared);
    let parts: Vec<&str> = version.split('.').collect();
    let exact = parts.len() >= 3
        && parts
            .iter()
            .take(3)
            .all(|part| part.chars().next().is_some_and(|c| c.is_ascii_digit()));
    if !exact {
        return Err(format!(
            "`{name}` must be exact-pinned (`=x.y.z`), got `{version}`"
        ));
    }
    Ok(version.to_string())
}
