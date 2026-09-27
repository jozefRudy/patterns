//! All network I/O for the load path: fetch an artifact at a pinned revision, verify its bytes
//! against the hub's own content hashes (sha256 for LFS files, git blob sha1 for the rest), read
//! the metadata the load-time guards need.
//!
//! The file listing (path → sha256) for a pinned revision is immutable, so it is cached on disk;
//! a warm load needs no network at all (the artifact bytes come from hf-hub's own blob cache).
//!
//! Runs inside `spawn_blocking` (hf-hub's API is sync). The `HubClient` seam exists so this
//! module's tests run fully offline.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use fastembed::TokenizerFiles;
use hf_hub::api::sync::{Api, ApiBuilder};
use hf_hub::{Repo, RepoType};
use serde::{Deserialize, Serialize};
use sha1::Sha1;
use sha2::{Digest, Sha256};

use crate::embed::PoolingMeta;
use crate::embed::spec::ModelSpec;

/// `1_Pooling/config.json` — present in sentence-transformers-format repos only.
const POOLING_FILE: &str = "1_Pooling/config.json";

/// Everything `TextEmbedding::try_new_from_user_defined` and the load-time guards need.
pub(super) struct Artifact {
    pub(super) onnx: Vec<u8>,
    pub(super) tokenizer_files: TokenizerFiles,
    pub(super) pooling_meta: Option<PoolingMeta>,
    pub(super) matryoshka_dims: Option<Vec<usize>>,
    /// Trained token window: tokenizer `model_max_length` clamped by the model's `max_position_embeddings`.
    pub(super) max_length: usize,
}

/// Expected content hash for one file in a repo tree.
///
/// LFS files carry a sha256; git-tracked files (configs, tokenizer, vocab) have no LFS entry but
/// do have a git blob id — so every file is verifiable, not just the large ones.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum TreeHash {
    /// `lfs.sha256`: sha256 of the file's content.
    Sha256(String),
    /// `blobId`: sha1 of the git blob (`sha1("blob {len}\0" + content)`).
    GitBlobSha1(String),
}

/// Hub seam, injectable so tests run without network.
pub(super) trait HubClient {
    /// `path -> content hash` for the repo at that revision.
    fn tree(&self, repo: &str, revision: &str) -> Result<BTreeMap<String, TreeHash>>;

    /// File contents; `Err` when the file is absent at that revision.
    fn bytes(&self, repo: &str, revision: &str, file: &str) -> Result<Vec<u8>>;
}

/// Real client: hf-hub with a persistent cache dir.
pub(super) struct HubFiles {
    api: Api,
    /// Where per-revision file listings are cached; see [`write_tree_cache`].
    tree_cache_dir: PathBuf,
}

impl HubFiles {
    pub(super) fn new(cache_dir: PathBuf, show_download_progress: bool) -> Result<Self> {
        let tree_cache_dir = cache_dir.join("hub-trees");
        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir)
            .with_progress(show_download_progress)
            .build()
            .context("initialise the HuggingFace hub client")?;
        Ok(Self {
            api,
            tree_cache_dir,
        })
    }

    fn repo(&self, repo: &str, revision: &str) -> hf_hub::api::sync::ApiRepo {
        self.api.repo(Repo::with_revision(
            repo.to_string(),
            RepoType::Model,
            revision.to_string(),
        ))
    }

    /// `{tree_cache_dir}/{repo--}@{revision}.json` (revision is a pinned sha, so the name is stable).
    fn tree_cache_path(&self, repo: &str, revision: &str) -> PathBuf {
        self.tree_cache_dir
            .join(format!("{}@{revision}.json", repo.replace('/', "--")))
    }
}

impl HubClient for HubFiles {
    fn tree(&self, repo: &str, revision: &str) -> Result<BTreeMap<String, TreeHash>> {
        let path = self.tree_cache_path(repo, revision);
        if let Some(cached) = read_tree_cache(&path) {
            return Ok(cached);
        }
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
        let tree = listing
            .siblings
            .into_iter()
            .map(|entry| {
                // LFS entries expose `lfs.sha256`; every entry has `blobId`.
                let hash = match entry.lfs {
                    Some(lfs) => TreeHash::Sha256(lfs.sha256),
                    None => TreeHash::GitBlobSha1(entry.blob_id),
                };
                (entry.rfilename, hash)
            })
            .collect();
        write_tree_cache(&path, &tree);
        Ok(tree)
    }

    fn bytes(&self, repo: &str, revision: &str, file: &str) -> Result<Vec<u8>> {
        let path = self
            .repo(repo, revision)
            .get(file)
            .with_context(|| format!("fetch {repo}/{file}@{revision}"))?;
        std::fs::read(&path).with_context(|| format!("read the cached file {}", path.display()))
    }
}

/// Read a cached listing; `None` when absent or unparseable (falls back to the network).
fn read_tree_cache(path: &Path) -> Option<BTreeMap<String, TreeHash>> {
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Persist a listing for a pinned revision.
///
/// Best-effort: a cache is an optimization, so any failure here (unwritable dir, full disk) is
/// swallowed — the listing is already in hand and the load must not fail because of it.
fn write_tree_cache(path: &Path, tree: &BTreeMap<String, TreeHash>) {
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let Ok(bytes) = serde_json::to_vec(tree) else {
        return;
    };
    // temp + rename so a concurrent reader never sees a half-written file
    let tmp = path.with_extension("json.tmp");
    if std::fs::write(&tmp, bytes).is_err() {
        return;
    }
    std::fs::rename(&tmp, path).unwrap_or_default();
}

/// `/api/models/{repo}/revision/{rev}?blobs=true` — only the fields we need.
#[derive(Deserialize)]
struct RepoListing {
    siblings: Vec<RepoEntry>,
}

#[derive(Deserialize)]
struct RepoEntry {
    rfilename: String,
    /// Git blob oid; present for every file.
    #[serde(rename = "blobId")]
    blob_id: String,
    lfs: Option<LfsEntry>,
}

#[derive(Deserialize)]
struct LfsEntry {
    sha256: String,
}

/// Fetch and verify everything the load path needs for `spec`.
///
/// # Errors
/// A missing required file, a content-hash mismatch against the hub's own listing, or unparseable
/// `config.json` / `tokenizer_config.json` / `1_Pooling/config.json`. Never falls back to another
/// model: rows are keyed by the identity of what was actually loaded.
pub(super) fn fetch<C: HubClient>(client: &C, spec: &ModelSpec) -> Result<Artifact> {
    let tree = client.tree(spec.repo(), spec.revision())?;
    ensure!(
        !tree.is_empty(),
        "{}@{}: empty file listing",
        spec.repo(),
        spec.revision()
    );

    let read = |file: &str| -> Result<Vec<u8>> {
        let expected = tree.get(file).ok_or_else(|| {
            anyhow!(
                "{}@{}: `{file}` is not in the repo",
                spec.repo(),
                spec.revision()
            )
        })?;
        let bytes = client.bytes(spec.repo(), spec.revision(), file)?;
        verify(file, &bytes, expected)?;
        Ok(bytes)
    };

    let onnx = read(spec.file())?;
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read("tokenizer.json")?,
        config_file: read("config.json")?,
        special_tokens_map_file: read("special_tokens_map.json")?,
        tokenizer_config_file: read("tokenizer_config.json")?,
    };

    Ok(Artifact {
        onnx,
        max_length: parse_max_length(
            &tokenizer_files.tokenizer_config_file,
            &tokenizer_files.config_file,
        )?,
        matryoshka_dims: parse_matryoshka(&tokenizer_files.config_file)?,
        pooling_meta: read_pooling_meta(client, spec, &tree)?,
        tokenizer_files,
    })
}

/// Trained token window: tokenizer `model_max_length` clamped to the model's `max_position_embeddings`.
///
/// The tokenizer value is often a placeholder (e.g. `8192` without dynamic `RoPE` scaling, or `BGE`'s
/// huge sentinel); the positional config is what the model actually trained on. `Fastembed`'s private
/// `DEFAULT_MAX_LENGTH` is never consulted.
fn parse_max_length(tokenizer_config: &[u8], model_config: &[u8]) -> Result<usize> {
    let config: serde_json::Value =
        serde_json::from_slice(tokenizer_config).context("parse tokenizer_config.json")?;
    let value = config
        .get("model_max_length")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow!("tokenizer_config.json has no numeric `model_max_length`"))?;
    let tokenizer_max = usize::try_from(value)
        .map_err(|err| anyhow!("model_max_length {value} is out of range: {err}"))?;

    let model: serde_json::Value =
        serde_json::from_slice(model_config).context("parse config.json")?;
    let trained = model
        .get("max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());

    Ok(trained.map_or(tokenizer_max, |trained| tokenizer_max.min(trained)))
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

/// `1_Pooling/config.json` from the artifact repo, when it ships one (converted exports do not).
fn read_pooling_meta<C: HubClient>(
    client: &C,
    spec: &ModelSpec,
    tree: &BTreeMap<String, TreeHash>,
) -> Result<Option<PoolingMeta>> {
    let Some(expected) = tree.get(POOLING_FILE) else {
        return Ok(None);
    };
    let bytes = client.bytes(spec.repo(), spec.revision(), POOLING_FILE)?;
    verify(POOLING_FILE, &bytes, expected)?;
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

/// Always-on artifact verification: bytes must hash to the hash the hub reports for that path.
/// LFS files are checked against their sha256; git-tracked files against their git blob sha1 —
/// so every file the loader reads is covered.
fn verify(file: &str, bytes: &[u8], expected: &TreeHash) -> Result<()> {
    let (kind, actual, expected) = match expected {
        TreeHash::Sha256(expected) => ("sha256", hex(&Sha256::digest(bytes)), expected),
        TreeHash::GitBlobSha1(expected) => ("git blob sha1", git_blob_sha1(bytes), expected),
    };
    ensure!(
        actual == *expected,
        "{kind} mismatch for `{file}`: got {actual}, hub reports {expected}"
    );
    Ok(())
}

/// Git object id of a blob: `sha1("blob {len}\0" + content)`, matching the hub's `blobId`.
fn git_blob_sha1(bytes: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(format!("blob {}\0", bytes.len()).as_bytes());
    hasher.update(bytes);
    hex(&hasher.finalize())
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
        fn tree(&self, repo: &str, revision: &str) -> Result<BTreeMap<String, TreeHash>> {
            let _ = revision;
            Ok(self
                .files
                .iter()
                .filter(|((owner, _), _)| owner == repo)
                .map(|((_, file), body)| {
                    // `hashed` selects the LFS (sha256) vs git-blob (sha1) branch, as the hub does.
                    let hash = if self.hashed {
                        TreeHash::Sha256(hex(&Sha256::digest(body)))
                    } else {
                        TreeHash::GitBlobSha1(git_blob_sha1(body))
                    };
                    (file.clone(), hash)
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
    fn tree_cache_round_trips_and_ignores_corruption() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nested/org--repo@sha.json");
        let tree: BTreeMap<String, TreeHash> = BTreeMap::from([
            (
                "onnx/model.onnx".to_string(),
                TreeHash::Sha256("abc123".to_string()),
            ),
            (
                "config.json".to_string(),
                TreeHash::GitBlobSha1("def456".to_string()),
            ),
        ]);

        assert_eq!(read_tree_cache(&path), None, "absent cache misses");

        write_tree_cache(&path, &tree);
        assert_eq!(
            read_tree_cache(&path),
            Some(tree),
            "written tree round-trips"
        );

        std::fs::write(&path, b"not json").expect("overwrite");
        assert_eq!(read_tree_cache(&path), None, "corrupt cache is ignored");
    }

    #[test]
    fn hub_files_serves_tree_from_disk_cache_without_network() {
        let dir = tempfile::tempdir().expect("tempdir");
        let hub = HubFiles::new(dir.path().to_path_buf(), false).expect("build hub client");
        let tree: BTreeMap<String, TreeHash> =
            BTreeMap::from([(ONNX.to_string(), TreeHash::Sha256("deadbeef".to_string()))]);

        // a repo that does not exist: a network call would fail, so a cache hit proves offline
        let repo = "example/nonexistent";
        write_tree_cache(&hub.tree_cache_path(repo, SHA), &tree);
        assert_eq!(hub.tree(repo, SHA).expect("served from cache"), tree);
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
    }

    #[test]
    fn parse_max_length_clamps_to_trained_window() {
        // tokenizer claims 8192 but the model was trained at 2048 (no dynamic RoPE scaling)
        let got = parse_max_length(
            br#"{"model_max_length":8192}"#,
            br#"{"max_position_embeddings":2048}"#,
        )
        .expect("parse");
        assert_eq!(got, 2048);

        // absurd placeholder: the positional config wins
        let got = parse_max_length(
            br#"{"model_max_length":18446744073709551615}"#,
            br#"{"max_position_embeddings":512}"#,
        )
        .expect("parse");
        assert_eq!(got, 512);

        // no positional field: trust the tokenizer
        let got = parse_max_length(br#"{"model_max_length":512}"#, b"{}").expect("parse");
        assert_eq!(got, 512);
    }

    #[test]
    fn fetch_clamps_window_to_trained_positions() {
        let hub = tokenizer_files(FakeHub {
            hashed: true,
            ..FakeHub::default()
        })
        .with("org/repo", ONNX, b"onnx-bytes")
        .with(
            "org/repo",
            "config.json",
            br#"{"max_position_embeddings":2048,"matryoshka_dimensions":[256]}"#,
        )
        .with(
            "org/repo",
            "tokenizer_config.json",
            br#"{"model_max_length":8192}"#,
        );
        let artifact = fetch(&hub, &spec()).expect("fetch");
        assert_eq!(artifact.max_length, 2048);
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
        let error = fetch(&hub, &spec())
            .err()
            .expect("tokenizer.json is missing");
        assert!(error.to_string().contains("tokenizer.json"), "{error}");
    }

    /// Serves `body` while the listing advertises a different hash, so verification must fail.
    struct Tampered {
        inner: FakeHub,
        body: Vec<u8>,
    }

    impl HubClient for Tampered {
        fn tree(&self, repo: &str, revision: &str) -> Result<BTreeMap<String, TreeHash>> {
            let mut tree = self.inner.tree(repo, revision)?;
            tree.insert(
                ONNX.to_string(),
                TreeHash::Sha256(hex(&Sha256::digest(b"expected"))),
            );
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
        let error = fetch(&tampered, &spec())
            .err()
            .expect("sha256 must mismatch");
        assert!(error.to_string().contains("sha256 mismatch"), "{error}");
    }

    #[test]
    fn fetch_verifies_git_tracked_files_via_blob_sha1() {
        // `hashed: false` → the listing carries git blob ids (sha1), as for non-LFS files.
        let hub = tokenizer_files(FakeHub {
            hashed: false,
            ..FakeHub::default()
        })
        .with("org/repo", ONNX, b"onnx-bytes");
        let artifact = fetch(&hub, &spec()).expect("git-tracked files verify via blob sha1");
        assert_eq!(artifact.onnx, b"onnx-bytes");
    }

    /// Serves git-tracked files whose content does not match the listed `blobId`.
    struct TamperedBlob {
        inner: FakeHub,
    }

    impl HubClient for TamperedBlob {
        fn tree(&self, repo: &str, revision: &str) -> Result<BTreeMap<String, TreeHash>> {
            self.inner.tree(repo, revision)
        }

        fn bytes(&self, repo: &str, revision: &str, file: &str) -> Result<Vec<u8>> {
            if file == "tokenizer.json" {
                return Ok(b"tampered".to_vec());
            }
            self.inner.bytes(repo, revision, file)
        }
    }

    #[test]
    fn fetch_rejects_a_git_blob_sha1_mismatch() {
        let hub = tokenizer_files(FakeHub {
            hashed: false,
            ..FakeHub::default()
        })
        .with("org/repo", ONNX, b"onnx-bytes");
        let error = fetch(&TamperedBlob { inner: hub }, &spec())
            .err()
            .expect("git blob sha1 must mismatch");
        assert!(
            error.to_string().contains("git blob sha1 mismatch"),
            "{error}"
        );
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
    fn fetch_rejects_an_empty_listing() {
        let error = fetch(&FakeHub::default(), &spec())
            .err()
            .expect("an empty listing must fail");
        assert!(error.to_string().contains("empty file listing"), "{error}");
    }
}
