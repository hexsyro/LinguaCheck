# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
