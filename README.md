# LinguaCheck

Sentiment scores computed in a language the model doesn't speak are still
numbers on a dashboard. This catches them.

## Why this exists

Sentiment tools lose ten or more accuracy points on non-English content. An
English-tuned model gets quietly applied to Spanish, Arabic, or Japanese text,
the score comes out wrong, and nothing in the pipeline says so. The error
blends into the dashboard as data.

LinguaCheck reads a dataset of text plus sentiment metadata, detects the real
language of each record, and flags every row where the model's assumed
language doesn't match reality. It also knows which models are weak outside
English (VADER, AFINN, TextBlob, and friends), when a record switches
languages mid-sentence, and which low-resource languages your model coverage
is missing.

## What it does

- **Language detection per record.** A fast, dependency-free classifier:
  script identification (CJK, Cyrillic, Arabic, and more) plus stopword
  scoring for Latin and Cyrillic languages. No GPU, no model download.
- **Mismatch flags.** "Detected `es` but scored with an `en`-tuned model
  (vader)." Severity rises when the model is known-weak or the text is CJK,
  where word-boundary tokenization breaks the score outright.
- **Weak-model detection.** A built-in list of models that only really work
  on English. Replace it with `--weak-models` if your stack differs.
- **Code-switching detection.** Records switching between Unicode scripts
  (Latin, Cyrillic, Greek, Arabic, Devanagari, CJK, …) or between two
  confident languages in one script — the case a single-language model cannot
  score correctly.
- **Per-record language distribution.** Mixed rows get a language/share
  breakdown; when no language holds at least 0.8 of the record, the sentiment
  label is marked `mixed_language_unscorable` instead of trusted.
- **Coverage reporting.** Per-language totals, flag rates, a per-record
  language distribution, a detected-vs-assumed model coverage matrix with a
  recommended action (`ok`, `reroute`, `translate-then-score`,
  `mark-unscorable`), plus which low-resource languages showed up.

Low-confidence detections (gibberish, a single unknown word) are skipped with
a warning, not guessed at. The threshold is configurable with
`--min-confidence` (default `0.2`). A wrong accusation is worse than no
accusation. Every finding carries `metadata["finding_kind"]` (`language_mismatch`,
`weak_model`, `code_switching`, or `mixed_language_unscorable`) so downstream
tools can tell the signals apart, and a row may carry both a model problem and
a code-switching/unscorable warning.

## What it is not

It is not a sentiment model and not a translation service. It never scores
your text — it audits the scoring your pipeline already did.

## Install

Python 3.10+, no runtime dependencies.

```bash
git clone https://github.com/hexsyro/LinguaCheck.git
cd LinguaCheck
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Usage

```bash
linguacheck examples/sample.csv
```

```
linguacheck 0.2.0 — 6 finding(s), 10 row(s) checked

  [high  ] 1              model_language
           Detected 'es' but scored with an 'en'-tuned model (vader).

  [high  ] 3              model_language
           CJK text detected as 'ja' but scored with an 'en'-tuned model (vader) — word-boundary
           tokenization breaks this model.

  [medium] 4              model_language
           Detected 'de' but scored with an 'en'-tuned model (custom-bert).

  [high  ] 6              model_language
           Detected 'ru' but scored with an 'en'-tuned model (vader).

  [medium] 8              model_language
           Code-switching detected (en / es in one record) — a single-language sentiment model
           cannot score this correctly.

  [medium] 9              model_language
           Detected 'ar' but scored with an 'en'-tuned model (custom-bert).

Warnings:
  - 1 row(s) skipped due to low confidence: too little evidence to attribute a language.
```

It looks for a text column among `text`, `content`, `message`, `body`,
`post_text`, `tweet`, `comment`, and `review`; a model column among `model`,
`sentiment_model`, `tool`, `model_name`, and `classifier`; and an assumed
language from `model_language`, `assumed_language`, `language_model`, `lang`,
or `expected_language` — all matched case-insensitively. With no language
column, the model name decides: `vader` means English-tuned. An unknown model
with no language column means nothing, and the tool stays quiet.

## Options

| Flag | What it does |
| --- | --- |
| `--weak-models LIST` | Comma-separated model names to treat as known-weak (replaces the built-in list) |
| `--min-confidence F` | Minimum detection confidence to trust a row, 0–1 (default `0.2`); config key `min_confidence` |
| `--config FILE` | JSON file with the same option keys; explicit flags win. Invalid value types exit 2 |
| `--format {table,json,csv,md}` | Output format (default `table`) |
| `--output FILE` | Write the report to a file instead of stdout |
| `--fail-on {none,low,medium,high}` | Exit 1 if any finding is at or above this severity |
| `--version` | Print version and exit |

## Output formats

`json` emits the machine contract this tool family shares:

```json
{
  "tool": "linguacheck",
  "version": "0.2.0",
  "status": "success",
  "findings": [
    {
      "severity": "high",
      "field": "model_language",
      "message": "Detected 'es' but scored with an 'en'-tuned model (vader).",
      "record_id": "1",
      "metadata": {
        "detected": "es",
        "script": "latin",
        "assumed": "en",
        "model": "vader",
        "score": -0.82,
        "detection_confidence": 0.615,
        "runner_up": "ca",
        "margin": 2.0,
        "language_distribution": {"es": 1.0},
        "cjk_risk": false,
        "finding_kind": "language_mismatch"
      }
    }
  ],
  "warnings": [],
  "summary": {
    "result_schema_version": "1.1",
    "skipped": false,
    "skip_reason": null,
    "checked": 10,
    "flagged": 6,
    "percent_flagged": 60.0,
    "weak_model_hits": 4,
    "min_confidence": 0.2,
    "by_language": {"es": {"total": 2, "flagged": 1, "percent_flagged": 50.0}},
    "low_resource_languages": ["cy"],
    "low_resource_records": 1,
    "language_distribution": [
      {"record_id": "8", "lang": "en", "share": 0.5, "languages": ["en", "es"]},
      {"record_id": "8", "lang": "es", "share": 0.5, "languages": ["en", "es"]}
    ],
    "coverage_matrix": [
      {"detected": "es", "assumed": "en", "count": 1, "risk": "medium", "action": "reroute"},
      {"detected": "ja", "assumed": "en", "count": 1, "risk": "high", "action": "translate-then-score"},
      {"detected": "cy", "assumed": "unknown", "count": 1, "risk": "high", "action": "mark-unscorable"}
    ]
  }
}
```

Feed it to WhyChanged, ReadyPack, or anything else that reads findings files.

## Using it in CI

```bash
linguacheck export.csv --fail-on high || exit 1
```

Exit codes: `0` ran fine, `1` findings tripped the gate, `2` the input was
broken.

## Library use

```python
from linguacheck import RESULT_SCHEMA_VERSION, ToolResult, run

result: ToolResult = run(
    rows,
    {"weak_models": ["vader", "my-internal-model"], "min_confidence": 0.3},
)
assert result.summary["result_schema_version"] == RESULT_SCHEMA_VERSION
print(result.summary["percent_flagged"], "% flagged")
```

## Limitations, honestly

- The detector is heuristic. Stopword scoring covers 40+ languages well;
  short texts, heavy slang, and single words get skipped rather than
  guessed. That's the right trade — this tool exists to prevent false
  confidence, not create it.
- Script evidence is decisive but coarse: any CJK text is tagged `zh`, `ja`,
  or `ko` by character set, which fails on rare mixed-script edge cases.
- The weak-model list is a judgment call maintained in `logic.py`. If your
  pipeline uses a model that should be on it, open an issue or pass
  `--weak-models`.
- Code-switching detection uses script transitions plus sentence-level Latin
  stopwords; two languages inside one script and one sentence can still slip by.
- The coverage matrix recommends an action but does not reroute or translate
  anything — that is your pipeline's job.

## [0.2.0] - 2026-11-19

### Added

- `--min-confidence F` (config key `min_confidence`, default `0.2`) replaces the
  hard-coded detection gate; invalid config value types exit 2 with a clear
  stderr message instead of raising.
- Per-record `language_distribution` in the summary and on finding metadata,
  with a `mixed_language_unscorable` finding when no language share reaches 0.8.
- Model coverage matrix (`detected`, `assumed`, `count`, `risk`, `action`) with
  `ok` / `reroute` / `translate-then-score` / `mark-unscorable` actions.
- `finding_kind` on every finding, so a row can report a model problem and a
  code-switching/unscorable problem at the same time.
- ISO-639-3 → ISO-639-1 normalization for common language labels.
- Script-transition code switching for Devanagari, Arabic, CJK, Greek, Latin,
  Cyrillic, and the other supported scripts.
- Shared result contract: `RESULT_SCHEMA_VERSION = "1.1"`, exported `ToolResult`,
  and `result_schema_version` / `skipped` / `skip_reason` in every summary.

### Fixed

- `weak_models` supplied as a string no longer iterates character by character.
- Runner-up detection ignores zero-hit languages; `margin` is the raw hit
  difference (0 with no runner-up) and the README example matches the output.
- Short texts use scaled stopword coverage and a 0.85 confidence cap; skipped
  rows get an explicit low-confidence warning.
- Malformed option values can no longer raise out of `run()` or `main()`.

### Inspiration

- [Code-switching cascades into downstream NLP](https://deepgram.com/blog/what-is-code-switching-asr-guide)
  — why a switch inside one record breaks the next stage, not just the transcriber.
- [GLUECoS and code-switched sentiment benchmarks](https://www.nature.com/articles/s41598-024-60210-7)
  — mixed-language sentiment is a benchmarked failure mode, not an edge case.
- [COSMUS and the "mixed" class](https://aclanthology.org/2025.unlp-1.18.pdf)
  — a label that refuses to pick one language is a legitimate outcome.
- [Gold-label noise in code-switched LID](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.44)
  — language labels on mixed records are themselves uncertain; report that.

## License

MIT. See [LICENSE](LICENSE).
