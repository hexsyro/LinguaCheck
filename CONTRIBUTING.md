# Contributing

Thanks for considering a contribution. This is a small tool on purpose, and
that's a feature worth protecting.

## Ground rules

- **Keep it stdlib-only at runtime.** No third-party dependencies in
  `src/linguacheck/`. The detector ships with its own stopword tables and
  script ranges; it does not pull in `langdetect`, `fasttext`, or a model
  download.
- **A new language needs evidence.** Add a stopword table (15–30 high-signal
  words), a test with a real sentence that must detect, and a test with a
  neighboring language's sentence that must not false-positive on it.
- **No silent behavior changes.** If a detection result, severity, or
  warning changes, that goes in the changelog.

## Workflow

1. Open an issue describing the bug or the new language before starting work
   on anything bigger than a typo fix.
2. Fork, branch, make the change, add tests.
3. Run `python -m pytest` and make sure it's green.
4. Open a pull request that says what changed and why, in plain language.

## Adding a weak model

Weak models live in `WEAK_MODELS` in `logic.py`. Adding one is a one-line
change plus a test — but only if the model genuinely fails outside English
(or wherever its training data ends). Cite something better than a hunch.
