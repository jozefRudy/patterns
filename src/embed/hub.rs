//! All network I/O for the load path: fetch an artifact at a pinned revision, verify its bytes
//! against the hub's own sha256, read the metadata the load-time guards need.
//!
//! Runs inside `spawn_blocking` (hf-hub's API is sync). The `HubClient` seam exists so this
//! module's tests run fully offline.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, ensure};
use fastembed::TokenizerFiles;
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::embed::spec::{MetadataSource, ModelSpec, PoolingMeta};

/// `1_Pooling/config.json` — present in sentence-transformers-format repos only.
const POOLING_FILE: &str = "1_Pooling/config.json";

/// Everything `TextEmbedding::try_new_from_user_defined` and the load-time guards need.
#[derive(Debug)]
pub struct Artifact {
    pub(crate) onnx: Vec<u8>,
    /// `(file_name, bytes)` for external initializers — plain tuples, since fastembed's
    /// `ExternalInitializerFile` is not nameable from outside the crate.
    pub(crate) external: Vec<(String, Vec<u8>)>,
    pub(crate) tokenizer_files: TokenizerFiles,
    pub(crate) pooling_meta: Option<PoolingMeta>,
    pub(crate) matryoshka_dims: Option<Vec<usize>>,
    /// Tokenizer window, from `tokenizer_config.json`'s `model_max_length`.
    pub(crate) max_length: usize,
}

/// Hub seam, injectable so tests run without network.
pub trait HubClient {
    /// `path -> sha256` for LFS files (`None` for small git-tracked files), at that revision.
    fn tree(&self, repo: &str, revision: &str) -> Result<BTreeMap<String, Option<String>>>;

    /// File contents; `Err` when the file is absent at that revision.
    fn bytes(&self, repo: &str, revision: &str, file: &str) -> Result<Vec<u8>>;
}

/// Real client: hf-hub with a persistent cache dir.
pub struct HubFiles {
    api: Api,
}

impl HubFiles {
    pub fn new(cache_dir: PathBuf, show_download_progress: bool) -> Result<Self> {
        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir)
            .with_progress(show_download_progress)
            .build()
            .context("initialise the HuggingFace hub client")?;
        Ok(Self { api })
    }

    fn repo(&self, repo: &str, revision: &str) -> hf_hub::api::sync::ApiRepo {
        self.api.repo(Repo::with_revision(
            repo.to_string(),
            RepoType::Model,
            revision.to_string(),
        ))
    }
}

impl HubClient for HubFiles {
    fn tree(&self, repo: &str, revision: &str) -> Result<BTreeMap<String, Option<String>>> {
        let mut response = self
            .repo(repo, revision)
            .info_request()
            .query("blobs", "true")
            .call()
            .with_context(|| format!("list {repo}@{revision}"))?;
        let listing: RepoListing = response
            .body_mut()
            .read_json()
            .with_context(|| format!("parse the file listing of {repo}@{revision}"))?;
        Ok(listing
            .siblings
            .into_iter()
            .map(|entry| (entry.rfilename, entry.lfs.map(|lfs| lfs.sha256)))
            .collect())
    }

    fn bytes(&self, repo: &str, revision: &str, file: &str) -> Result<Vec<u8>> {
        let path = self
            .repo(repo, revision)
            .get(file)
            .with_context(|| format!("fetch {repo}/{file}@{revision}"))?;
        std::fs::read(&path).with_context(|| format!("read the cached file {}", path.display()))
    }
}

/// `/api/models/{repo}/revision/{rev}?blobs=true` — only the fields we need.
#[derive(Deserialize)]
struct RepoListing {
    siblings: Vec<RepoEntry>,
}

#[derive(Deserialize)]
struct RepoEntry {
    rfilename: String,
    lfs: Option<LfsEntry>,
}

#[derive(Deserialize)]
struct LfsEntry {
    sha256: String,
}

/// Fetch and verify everything the load path needs for `spec`.
///
/// # Errors
/// A missing required file, a sha256 mismatch against the hub's own listing, or unparseable
/// `config.json` / `tokenizer_config.json` / `1_Pooling/config.json`. Never falls back to another
/// model: rows are keyed by the identity of what was actually loaded.
pub fn fetch<C: HubClient>(client: &C, spec: &ModelSpec) -> Result<Artifact> {
    let tree = client.tree(spec.repo(), spec.revision())?;
    ensure!(
        !tree.is_empty(),
        "{}@{}: empty file listing",
        spec.repo(),
        spec.revision()
    );

    let read = |file: &str| -> Result<Vec<u8>> {
        ensure!(
            tree.contains_key(file),
            "{}@{}: `{file}` is not in the repo",
            spec.repo(),
            spec.revision()
        );
        let bytes = client.bytes(spec.repo(), spec.revision(), file)?;
        verify(file, &bytes, tree.get(file).and_then(Option::as_deref))?;
        Ok(bytes)
    };

    let onnx = read(spec.file())?;
    let mut external = Vec::with_capacity(spec.additional().len());
    for file in spec.additional() {
        external.push((file.clone(), read(file)?));
    }

    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: read("special_tokens_map.json")?,
        tokenizer_config_file: read("tokenizer_config.json")?,
    };

    Ok(Artifact {
        onnx,
        external,
        max_length: parse_max_length(&tokenizer_files.tokenizer_config_file)?,
        matryoshka_dims: parse_matryoshka(&tokenizer_files.config_file)?,
        pooling_meta: read_pooling_meta(client, spec, &tree)?,
        tokenizer_files,
    })
}

/// `max_length` comes from the artifact, not from fastembed's private `DEFAULT_MAX_LENGTH`.
fn parse_max_length(tokenizer_config: &[u8]) -> Result<usize> {
    let config: serde_json::Value =
        serde_json::from_slice(tokenizer_config).context("parse tokenizer_config.json")?;
    let value = config
        .get("model_max_length")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow!("tokenizer_config.json has no numeric `model_max_length`"))?;
    usize::try_from(value).map_err(|err| anyhow!("model_max_length {value} is out of range: {err}"))
}

/// MRL widths declared by the model, if any (`config.json`'s `matryoshka_dimensions`).
fn parse_matryoshka(config: &[u8]) -> Result<Option<Vec<usize>>> {
    let config: serde_json::Value = serde_json::from_slice(config).context("parse config.json")?;
    let Some(dims) = config
        .get("matryoshka_dimensions")
        .and_then(serde_json::Value::as_array)
    else {
        return Ok(None);
    };
    let widths = dims
        .iter()
        .filter_map(serde_json::Value::as_u64)
        .filter_map(|dim| usize::try_from(dim).ok())
        .collect();
    Ok(Some(widths))
}

/// `1_Pooling/config.json` from the artifact repo, else from a declared metadata repo, else `None`.
fn read_pooling_meta<C: HubClient>(
    client: &C,
    spec: &ModelSpec,
    tree: &BTreeMap<String, Option<String>>,
) -> Result<Option<PoolingMeta>> {
    if tree.contains_key(POOLING_FILE) {
        let bytes = client.bytes(spec.repo(), spec.revision(), POOLING_FILE)?;
        verify(
            POOLING_FILE,
            &bytes,
            tree.get(POOLING_FILE).and_then(Option::as_deref),
        )?;
        return Ok(Some(parse_pooling_meta(&bytes)?));
    }

    let Some(MetadataSource { repo, revision }) = spec.pooling_metadata_from() else {
        return Ok(None);
    };
    let other_tree = client.tree(repo, revision)?;
    if !other_tree.contains_key(POOLING_FILE) {
        return Ok(None);
    }
    let bytes = client.bytes(repo, revision, POOLING_FILE)?;
    verify(
        POOLING_FILE,
        &bytes,
        other_tree.get(POOLING_FILE).and_then(Option::as_deref),
    )?;
    Ok(Some(parse_pooling_meta(&bytes)?))
}

#[derive(Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors the 1_Pooling/config.json flag set verbatim"
)]
struct PoolingConfig {
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_max_tokens: bool,
    #[serde(default)]
    pooling_mode_weightedmean_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
    include_prompt: Option<bool>,
}

fn parse_pooling_meta(bytes: &[u8]) -> Result<PoolingMeta> {
    let config: PoolingConfig =
        serde_json::from_slice(bytes).context("parse 1_Pooling/config.json")?;
    Ok(PoolingMeta {
        cls: config.pooling_mode_cls_token,
        mean: config.pooling_mode_mean_tokens,
        max: config.pooling_mode_max_tokens,
        weightedmean: config.pooling_mode_weightedmean_tokens,
        mean_sqrt_len: config.pooling_mode_mean_sqrt_len_tokens,
        lasttoken: config.pooling_mode_lasttoken,
        include_prompt: config.include_prompt,
    })
}

/// Always-on artifact verification: bytes must hash to the sha256 the hub reports for that path.
/// Files the hub does not track as LFS have no hash to compare — revision pinning covers them.
fn verify(file: &str, bytes: &[u8], expected: Option<&str>) -> Result<()> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let actual = hex(&Sha256::digest(bytes));
    ensure!(
        actual == expected,
        "sha256 mismatch for `{file}`: got {actual}, hub reports {expected}"
    );
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        write!(out, "{byte:02x}").expect("writing to a String is infallible");
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
    const ONNX: &str = "onnx/model_quantized.onnx";

    /// In-memory hub: files plus their (real) sha256, so verification is exercised for real.
    #[derive(Default)]
    struct FakeHub {
        files: HashMap<(String, String), Vec<u8>>,
        /// When false, the listing omits `lfs` hashes (git-tracked files).
        hashed: bool,
    }

    impl FakeHub {
        fn with(mut self, repo: &str, file: &str, body: &[u8]) -> Self {
            self.files
                .insert((repo.to_string(), file.to_string()), body.to_vec());
            self
        }
    }

    impl HubClient for FakeHub {
        fn tree(&self, repo: &str, revision: &str) -> Result<BTreeMap<String, Option<String>>> {
            let _ = revision;
            Ok(self
                .files
                .iter()
                .filter(|((owner, _), _)| owner == repo)
                .map(|((_, file), body)| {
                    let entry = self.hashed.then(|| hex(&Sha256::digest(body)));
                    (file.clone(), entry)
                })
                .collect())
        }

        fn bytes(&self, repo: &str, revision: &str, file: &str) -> Result<Vec<u8>> {
            let _ = revision;
            self.files
                .get(&(repo.to_string(), file.to_string()))
                .cloned()
                .ok_or_else(|| anyhow!("{repo}/{file} not found"))
        }
    }

    fn tokenizer_files(hub: FakeHub) -> FakeHub {
        hub.with("org/repo", "tokenizer.json", b"{\"v\":1}")
            .with(
                "org/repo",
                "config.json",
                b"{\"matryoshka_dimensions\":[256]}",
            )
            .with("org/repo", "special_tokens_map.json", b"{}")
            .with(
                "org/repo",
                "tokenizer_config.json",
                b"{\"model_max_length\":512}",
            )
    }

    fn spec() -> ModelSpec {
        ModelSpec::new("org/repo", SHA, ONNX, 768).expect("valid spec")
    }

    #[test]
    fn fetch_reads_artifact_and_metadata() {
        let hub = tokenizer_files(FakeHub {
            hashed: true,
            ..FakeHub::default()
        })
        .with("org/repo", ONNX, b"onnx-bytes");
        let artifact = fetch(&hub, &spec()).expect("fetch");
        assert_eq!(artifact.onnx, b"onnx-bytes");
        assert_eq!(artifact.max_length, 512);
        assert_eq!(artifact.matryoshka_dims, Some(vec![256]));
        assert!(artifact.pooling_meta.is_none(), "no 1_Pooling/config.json");
        assert_eq!(artifact.external, []);
    }

    #[test]
    fn fetch_rejects_a_missing_required_file() {
        let hub = FakeHub {
            hashed: true,
            ..FakeHub::default()
        }
        .with("org/repo", ONNX, b"onnx-bytes")
        .with("org/repo", "config.json", b"{}")
        .with("org/repo", "special_tokens_map.json", b"{}")
        .with(
            "org/repo",
            "tokenizer_config.json",
            b"{\"model_max_length\":512}",
        );
        let error = fetch(&hub, &spec()).expect_err("tokenizer.json is missing");
        assert!(error.to_string().contains("tokenizer.json"), "{error}");
    }

    /// Serves `body` while the listing advertises a different hash, so verification must fail.
    struct Tampered {
        inner: FakeHub,
        body: Vec<u8>,
    }

    impl HubClient for Tampered {
        fn tree(&self, repo: &str, revision: &str) -> Result<BTreeMap<String, Option<String>>> {
            let mut tree = self.inner.tree(repo, revision)?;
            tree.insert(ONNX.to_string(), Some(hex(&Sha256::digest(b"expected"))));
            Ok(tree)
        }

        fn bytes(&self, repo: &str, revision: &str, file: &str) -> Result<Vec<u8>> {
            if file == ONNX {
                return Ok(self.body.clone());
            }
            self.inner.bytes(repo, revision, file)
        }
    }

    #[test]
    fn fetch_rejects_a_sha256_mismatch() {
        let hub = tokenizer_files(FakeHub {
            hashed: true,
            ..FakeHub::default()
        })
        .with("org/repo", ONNX, b"onnx-bytes");
        let tampered = Tampered {
            inner: hub,
            body: b"tampered".to_vec(),
        };
        let error = fetch(&tampered, &spec()).expect_err("sha256 must mismatch");
        assert!(error.to_string().contains("sha256 mismatch"), "{error}");
    }

    #[test]
    fn fetch_skips_verification_for_unhashed_files() {
        let hub = tokenizer_files(FakeHub {
            hashed: false,
            ..FakeHub::default()
        })
        .with("org/repo", ONNX, b"onnx-bytes");
        let artifact = fetch(&hub, &spec()).expect("non-LFS files have no hash to check");
        assert_eq!(artifact.onnx, b"onnx-bytes");
    }

    #[test]
    fn fetch_parses_pooling_metadata() {
        let hub = tokenizer_files(FakeHub {
            hashed: true,
            ..FakeHub::default()
        })
        .with("org/repo", ONNX, b"onnx-bytes")
        .with(
            "org/repo",
            POOLING_FILE,
            b"{\"pooling_mode_cls_token\":true,\"include_prompt\":false}",
        );
        let artifact = fetch(&hub, &spec()).expect("fetch");
        let meta = artifact.pooling_meta.expect("1_Pooling parsed");
        assert!(meta.cls);
        assert!(!meta.mean);
        assert_eq!(meta.include_prompt, Some(false));
    }

    #[test]
    fn fetch_falls_back_to_the_declared_metadata_repo_for_pooling() {
        let hub = tokenizer_files(FakeHub {
            hashed: true,
            ..FakeHub::default()
        })
        .with("org/repo", ONNX, b"onnx-bytes")
        .with(
            "upstream/repo",
            "1_Pooling/config.json",
            b"{\"pooling_mode_mean_tokens\":true}",
        );
        let spec = spec().with_pooling_metadata_from("upstream/repo", SHA);
        let meta = fetch(&hub, &spec)
            .expect("fetch")
            .pooling_meta
            .expect("pooling metadata came from the metadata repo");
        assert!(meta.mean, "metadata repo's flags are used");
    }

    #[test]
    fn fetch_rejects_an_empty_listing() {
        let error = fetch(&FakeHub::default(), &spec()).expect_err("empty repo");
        assert!(error.to_string().contains("empty file listing"), "{error}");
    }
}
