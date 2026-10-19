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
- **Code-switching detection.** Records mixing two confident languages in one
  text — the case a single-language model cannot score correctly.
- **Coverage reporting.** Per-language totals and flag rates, plus which
  low-resource languages showed up, so you know what your model lineup
  actually needs to support.

Low-confidence detections (gibberish, a single unknown word) are skipped with
a warning, not guessed at. A wrong accusation is worse than no accusation.

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
linguacheck 0.1.0 — 6 finding(s), 10 row(s) checked

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
  - 1 row(s) had too little evidence to attribute a language and were skipped.
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
| `--config FILE` | JSON file with the same option keys; explicit flags win |
| `--format {table,json,csv,md}` | Output format (default `table`) |
| `--output FILE` | Write the report to a file instead of stdout |
| `--fail-on {none,low,medium,high}` | Exit 1 if any finding is at or above this severity |
| `--version` | Print version and exit |

## Output formats

`json` emits the machine contract this tool family shares:

```json
{
  "tool": "linguacheck",
  "version": "0.1.0",
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
        "cjk_risk": false,
        "detection_confidence": 0.917,
        "runner_up": "it",
        "margin": 0.833
      }
    }
  ],
  "warnings": [],
  "summary": {
    "checked": 10,
    "flagged": 5,
    "percent_flagged": 50.0,
    "by_language": {"es": {"total": 2, "flagged": 1, "percent_flagged": 50.0}},
    "low_resource_languages": ["cy"],
    "low_resource_records": 1
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
from linguacheck import run

result = run(rows, {"weak_models": ["vader", "my-internal-model"]})
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
- Code-switching detection works on Latin and Cyrillic scripts only.

## License

MIT. See [LICENSE](LICENSE).
