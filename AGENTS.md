# Project Rules

## Panics

- `unwrap` denied everywhere. `expect("reason")` allowed — documented invariant panic on `Result`/`Option`.
- Warranted invariant panics on bool conditions: `assert!`/`assert_eq!` with message.
- Example: wrong Lance schema at startup, impossible internal states — fail fast.

## After Code Changes

Don't relax clippy rules -> #[allow(clippy::*)]
After completing code changes, run validation:
```bash
cargo build --workspace && cargo clippy --workspace --all-targets && cargo test --workspace && cargo fmt --all
```

Integration tests, run after changes related to api clients

```bash
cargo test --workspace -- --include-ignored
```

## Style: functional core

Prefer functional style where idiomatic:

- take by value, return results — avoid `&mut` out-params
- read-only params borrow, never force a caller copy: `&T`/`&[T]`, or a generic (`S: AsRef<str>`) when callers hold varying types — not owned `String`/`Vec<T>`; `ptr_arg` misses this class, so clippy won't flag it
- mutation confined inside functions; immutable data across boundaries
  (accumulate into a local struct like `RefillOut`, return it)
- pure leaf fns get `const` (`cargo clippy --fix` adds it)

## Concurrency

- Shared mutation via message-passing (writer actor), not `Arc<Mutex<T>>` — never
  introduce mutexes around store access
- Tasks own `Arc<Store>`/`Arc<Config>`; no `Send`/`Sync` gymnastics (no `Rc`→`Arc`
  reshaping, no clone-to-satisfy-`spawn`, no lock-scope restructuring), no locks
  across `.await`
- Lance writes only through `writer::run` commands; queries are read-only DataFusion

## Documentation

- Check `md_docs/` when using unfamiliar APIs, 3rd party crates, or trait/method signature errors. Note: `md_docs/` is auto-generated and gitignored — don't edit it.

## Extraction rules

How patterns get into this crate, and how modules are shaped:

1. **Context-free patterns**: one consumer suffices if the API is stable.
2. **Context-coupled patterns**: wait for a second real consumer; extract from
   the diff between the implementations, not from one.
3. **One module per pattern.** Split into additional crates only on evidence
   (heavy/conflicting deps, or a required proc-macro crate — e.g. the internal
   `patterns-macros`). Never a catch-all `utils` module.
4. **Public repo + git deps with rev pins.** Publish to crates.io only once
   APIs stabilize and external use matters.
5. **Constraints travel with the crate README** (invariant panics,
   message-passing ownership, no locks across `.await`).
