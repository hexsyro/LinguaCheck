"""LinguaCheck — catch sentiment scored with the wrong language model.

Detects the actual language of each record and flags every row where the
sentiment model's assumed language doesn't match reality, plus known trouble
spots like CJK tokenization failures on English-only models (e.g. VADER),
code-switching, and records too mixed to score reliably.
"""

from __future__ import annotations

from typing import Any

from linguacheck.common import (
    RESULT_SCHEMA_VERSION,
    ToolResult,
    as_str,
    find_id,
    get,
    opt_float,
    opt_list,
    to_float,
)
from linguacheck.languages import (
    detect_code_switching,
    detect_language_scored,
    is_low_resource,
    language_distribution,
    normalize_lang,
)

WEAK_MODELS = [
    "vader",
    "vadersentiment",
    "afinn",
    "textblob",
    "sentiwordnet",
    "pattern",
    "liwc",
    "roberta-base",
    "twitter-roberta",
    "nlptown",
    "distilbert",
]

MODEL_LANGUAGE: dict[str, str] = {
    "vader": "en",
    "vadersentiment": "en",
    "afinn": "en",
    "textblob": "en",
    "sentiwordnet": "en",
    "pattern": "en",
    "liwc": "en",
}

TEXT_COLUMNS = ["text", "content", "message", "body", "post_text", "tweet", "comment", "review"]
MODEL_COLUMNS = ["model", "sentiment_model", "tool", "model_name", "classifier"]
ASSUMED_COLUMNS = ["model_language", "assumed_language", "language_model", "lang", "expected_language"]
SCORE_COLUMNS = ["sentiment_score", "score", "polarity", "sentiment"]

CJK_LANGS = frozenset({"zh", "ja", "ko"})
DEFAULT_MIN_CONFIDENCE = 0.2
DOMINANT_SHARE = 0.8


def _model_name(row: dict) -> str:
    raw = get(row, MODEL_COLUMNS)
    return as_str(raw).lower()


def _assumed_language(row: dict, model: str) -> str | None:
    raw = get(row, ASSUMED_COLUMNS)
    if raw is not None and as_str(raw):
        return normalize_lang(as_str(raw))
    if model in MODEL_LANGUAGE:
        return MODEL_LANGUAGE[model]
    # An unknown model with no language column tells us nothing about the model's
    # language, so stay quiet rather than assume an English default (many feeds
    # carry unrelated `score` columns that would otherwise cause false positives).
    return None


class LinguaCheckLogic:
    """Language-mismatch scanner."""

    def run(self, rows: list[dict], options: dict[str, Any] | None = None) -> ToolResult:
        options = options or {}
        weak_models = {name.lower() for name in opt_list(options, "weak_models", WEAK_MODELS) if name}
        min_confidence = opt_float(options, "min_confidence", DEFAULT_MIN_CONFIDENCE, low=0.0, high=1.0)

        findings: list[dict] = []
        warnings: list[str] = []
        checked = 0
        analyzed = 0
        flagged = 0
        skipped_low_confidence = 0
        by_language: dict[str, dict[str, int]] = {}
        low_resource_langs: dict[str, int] = {}
        weak_hits = 0
        language_distribution_out: list[dict[str, Any]] = []
        coverage_counts: dict[tuple[str, str], int] = {}
        coverage_mixed: set[tuple[str, str]] = set()

        for index, row in enumerate(rows):
            text = as_str(get(row, TEXT_COLUMNS))
            if not text:
                continue
            checked += 1
            detected, script, confidence, runner_up, margin = detect_language_scored(text)
            if detected == "unknown" or confidence < min_confidence:
                # No stopword evidence (e.g. gibberish or a single unknown
                # word): don't accuse a model mismatch on a guess.
                skipped_low_confidence += 1
                continue
            analyzed += 1

            entry = by_language.setdefault(detected, {"total": 0, "flagged": 0})
            entry["total"] += 1
            if is_low_resource(detected):
                low_resource_langs[detected] = low_resource_langs.get(detected, 0) + 1

            record_id = find_id(row) or f"row-{index + 1}"
            model = _model_name(row)
            weak = model in weak_models
            assumed = _assumed_language(row, model)
            score = to_float(get(row, SCORE_COLUMNS))
            cjk = detected in CJK_LANGS

            distribution = language_distribution(text)
            shares = dict(distribution)
            is_mixed = len(shares) >= 2 and max(shares.values()) < DOMINANT_SHARE
            if len(shares) >= 2:
                for lang, share in shares.items():
                    language_distribution_out.append(
                        {
                            "record_id": record_id,
                            "lang": lang,
                            "share": share,
                            "languages": sorted(shares),
                        }
                    )

            coverage_key = (detected, assumed or "unknown")
            coverage_counts[coverage_key] = coverage_counts.get(coverage_key, 0) + 1
            if is_mixed:
                coverage_mixed.add(coverage_key)

            common_metadata: dict[str, Any] = {
                "detected": detected,
                "script": script,
                "assumed": assumed,
                "model": model or None,
                "score": round(score, 4) if score is not None else None,
                "detection_confidence": confidence,
                "runner_up": runner_up or None,
                "margin": margin,
                "language_distribution": shares,
            }
            row_findings: list[dict] = []

            # One primary finding: a language mismatch wins over a weak-model
            # warning so the same problem isn't reported twice.
            if assumed and assumed != detected:
                severity = "high" if weak or cjk else "medium"
                cjk_risk = bool(cjk and weak)
                if cjk_risk:
                    message = (
                        f"CJK text detected as '{detected}' but scored with an '{assumed}'-tuned model"
                        f" ({model or 'unknown'}) — word-boundary tokenization breaks this model."
                    )
                else:
                    message = (
                        f"Detected '{detected}' but scored with an '{assumed}'-tuned model ({model or 'unknown'})."
                    )
                metadata = {
                    **common_metadata,
                    "cjk_risk": cjk_risk,
                }
                row_findings.append(
                    self._finding(
                        severity,
                        "model_language",
                        message,
                        row,
                        metadata,
                        finding_kind="language_mismatch",
                    )
                )
            elif weak and detected != "en":
                severity = "high" if cjk else "medium"
                if cjk:
                    message = (
                        f"Known-weak model '{model}' applied to CJK text ('{detected}') — "
                        "tokenization will break the score."
                    )
                else:
                    message = f"Known-weak model '{model}' applied to non-English text ('{detected}')."
                metadata = {
                    **common_metadata,
                    "cjk_risk": cjk,
                }
                row_findings.append(
                    self._finding(
                        severity,
                        "model_language",
                        message,
                        row,
                        metadata,
                        finding_kind="weak_model",
                    )
                )

            # A second finding may coexist with the primary one: code-switching
            # tells us the model can't score the row, mixed-language dominance
            # tells us the gold label itself is unreliable.
            mixed_languages = detect_code_switching(text, script)
            if mixed_languages:
                row_findings.append(
                    self._finding(
                        "medium",
                        "model_language",
                        f"Code-switching detected ({' / '.join(mixed_languages)} in one record) — "
                        "a single-language sentiment model cannot score this correctly.",
                        row,
                        {**common_metadata, "languages": mixed_languages},
                        finding_kind="code_switching",
                    )
                )
            elif is_mixed:
                row_findings.append(
                    self._finding(
                        "medium",
                        "language_distribution",
                        "No language holds >=0.8 of this record "
                        f"({', '.join(f'{lang}={share}' for lang, share in shares.items())}) — "
                        "mark the sentiment label unscorable rather than trusting it.",
                        row,
                        common_metadata,
                        finding_kind="mixed_language_unscorable",
                    )
                )

            if row_findings:
                flagged += 1
                entry["flagged"] += 1
                if weak:
                    weak_hits += 1
                findings.extend(row_findings)

        if checked == 0:
            warnings.append("No rows carried a text field (text, content, message, body, tweet, review).")
        if skipped_low_confidence:
            warnings.append(
                f"{skipped_low_confidence} row(s) skipped due to low confidence: "
                "too little evidence to attribute a language."
            )

        by_language_out: dict[str, dict[str, Any]] = {}
        for lang, counts in by_language.items():
            by_language_out[lang] = {
                "total": counts["total"],
                "flagged": counts["flagged"],
                "percent_flagged": round(counts["flagged"] / counts["total"] * 100, 2),
            }

        coverage_matrix: list[dict[str, Any]] = []
        for (detected, assumed), count in sorted(
            coverage_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        ):
            if (detected, assumed) in coverage_mixed:
                risk, action = "high", "mark-unscorable"
            elif detected == assumed:
                risk, action = "low", "ok"
            elif detected in CJK_LANGS:
                risk, action = "high", "translate-then-score"
            elif is_low_resource(detected):
                risk, action = "high", "mark-unscorable"
            else:
                risk, action = "medium", "reroute"
            coverage_matrix.append(
                {
                    "detected": detected,
                    "assumed": assumed,
                    "count": count,
                    "risk": risk,
                    "action": action,
                }
            )

        skipped = analyzed == 0
        if checked == 0:
            skip_reason: str | None = "no rows carried a text field"
        elif skipped:
            skip_reason = "no rows had enough language evidence"
        else:
            skip_reason = None

        summary: dict[str, Any] = {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "skipped": skipped,
            "skip_reason": skip_reason,
            "checked": checked,
            "flagged": flagged,
            "percent_flagged": round(flagged / checked * 100, 2) if checked else 0.0,
            "weak_model_hits": weak_hits,
            "min_confidence": min_confidence,
            "by_language": by_language_out,
            # Extended low-resource language coverage: which ones showed up in
            # this dataset (so a team knows which models it actually needs).
            "low_resource_languages": sorted(low_resource_langs),
            "low_resource_records": sum(low_resource_langs.values()),
            "language_distribution": language_distribution_out,
            "coverage_matrix": coverage_matrix,
        }
        return ToolResult(
            status="success",
            findings=findings,
            warnings=warnings,
            summary=summary,
        )

    @staticmethod
    def _finding(
        severity: str,
        field: str,
        message: str,
        row: dict,
        metadata: dict,
        *,
        finding_kind: str = "language_mismatch",
    ) -> dict:
        return {
            "severity": severity,
            "field": field,
            "message": message,
            "record_id": find_id(row),
            "metadata": {**metadata, "finding_kind": finding_kind},
        }


def run(rows: list[dict], options: dict[str, Any] | None = None) -> ToolResult:
    """Library entry point: scan ``rows`` for language/model mismatches."""
    return LinguaCheckLogic().run(rows, options or {})
