# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-19

### Added

- Shared result contract: `RESULT_SCHEMA_VERSION = "1.1"`, `ToolResult`
  exported from the package, and `result_schema_version` / `skipped` /
  `skip_reason` in every `run()` summary.
- `--min-confidence F` (config `min_confidence`, default `0.2`) replaces the
  hard-coded gate; option values are validated and CLI config type errors exit 2.
- `finding_kind` metadata (`language_mismatch`, `weak_model`, `code_switching`,
  `mixed_language_unscorable`) and up to two findings per row, so a model
  mismatch and a code-switching signal can coexist.
- Per-record `language_distribution` in the summary and on findings, plus a
  medium `mixed_language_unscorable` finding when no share reaches 0.8.
- Model coverage matrix with `detected` / `assumed` / `count` / `risk` /
  `action` (`ok`, `reroute`, `translate-then-score`, `mark-unscorable`).
- ISO-639-3 → ISO-639-1 normalization for common language codes.
- Unicode script-transition code switching (Devanagari, Arabic, CJK, Greek,
  Cyrillic, Latin, and more) with the sentence-chunk stopword fallback.

### Fixed

- `weak_models` given as a string no longer iterates characters; malformed
  option values never raise from `run()` or `main()`.
- Runner-up ranking keeps only languages with stopword hits; `margin` is the
  raw hit difference (0 without a runner-up) and the README example was updated.
- Short-text confidence uses scaled coverage and a 0.85 cap; skipped rows get
  an explicit low-confidence warning.

### Inspiration

- [Code-switching cascades into downstream NLP](https://deepgram.com/blog/what-is-code-switching-asr-guide)
- [GLUECoS / code-switched sentiment benchmarks](https://www.nature.com/articles/s41598-024-60210-7)
- [COSMUS "mixed" class](https://aclanthology.org/2025.unlp-1.18.pdf)
- [Gold-label noise in code-switched LID](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.44)

## [0.1.0] - 2026-10-19

### Added

- Initial release.
- Dependency-free language detection: script identification (CJK, Cyrillic,
  Arabic, Hebrew, Greek, Thai, Devanagari, and low-resource Indic scripts)
  plus stopword scoring for 30+ Latin and 6 Cyrillic languages.
- Confidence-scored detection with runner-up and margin, so thin evidence is
  skipped instead of guessed.
- Language-mismatch flags with severity tiers (high for known-weak models and
  CJK tokenization risk, medium otherwise).
- Built-in weak-model list (VADER, AFINN, TextBlob, and friends) with
  per-dataset override.
- Code-switching detection for mixed-language records.
- Per-language summary with flag rates and low-resource coverage reporting.
- CLI with `table`, `json`, `csv`, and `markdown` output, `--fail-on` exit
  codes for CI, and JSON config.
- Importable library API: `linguacheck.run(rows, options)`.
