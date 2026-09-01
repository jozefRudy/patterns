# patterns

Personal pattern library: reusable building blocks shared across my projects
via a pinned git dependency. One crate, module per pattern.

Modules:
- `llm_cli` — generic structured extraction from text via a local LLM CLI,
  with one-repair-retry semantics. Prompt templating stays in consumers.
- `lance_store` — reserved.

Usage:
`patterns = { git = "https://github.com/jozefRudy/patterns", rev = "<sha>" }`

License: MIT.
