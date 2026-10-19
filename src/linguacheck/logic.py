"""LinguaCheck — catch sentiment scored with the wrong language model.

Detects the actual language of each record and flags every row where the
sentiment model's assumed language doesn't match reality, plus known trouble
spots like CJK tokenization failures on English-only models (e.g. VADER).
"""

from __future__ import annotations

from typing import Any

from linguacheck.common import ToolResult, as_str, find_id, get, to_float
from linguacheck.languages import (
    detect_code_switching,
    detect_language_scored,
    is_low_resource,
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

    def run(self, rows: list[dict], options: dict[str, Any]) -> ToolResult:
        weak_models = {as_str(m).lower() for m in options.get("weak_models") or WEAK_MODELS}

        findings: list[dict] = []
        warnings: list[str] = []
        checked = 0
        flagged = 0
        skipped_low_confidence = 0
        by_language: dict[str, dict[str, int]] = {}
        low_resource_langs: dict[str, int] = {}
        weak_hits = 0

        for row in rows:
            text = as_str(get(row, TEXT_COLUMNS))
            if not text:
                continue
            checked += 1
            detected, script, confidence, runner_up, margin = detect_language_scored(text)
            if detected == "unknown":
                continue
            if confidence < 0.2:
                # No stopword evidence (e.g. gibberish or a single unknown
                # word): don't accuse a model mismatch on a guess.
                skipped_low_confidence += 1
                continue

            entry = by_language.setdefault(detected, {"total": 0, "flagged": 0})
            entry["total"] += 1
            if is_low_resource(detected):
                low_resource_langs[detected] = low_resource_langs.get(detected, 0) + 1

            model = _model_name(row)
            weak = model in weak_models
            assumed = _assumed_language(row, model)
            score = to_float(get(row, SCORE_COLUMNS))
            cjk = detected in ("zh", "ja", "ko")

            finding = None
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
                        f"Detected '{detected}' but scored with an '{assumed}'-tuned model"
                        f" ({model or 'unknown'})."
                    )
                finding = self._finding(
                    severity,
                    "model_language",
                    message,
                    row,
                    {
                        "detected": detected,
                        "script": script,
                        "assumed": assumed,
                        "model": model or None,
                        "score": round(score, 4) if score is not None else None,
                        "cjk_risk": cjk_risk,
                        "detection_confidence": confidence,
                        "runner_up": runner_up or None,
                        "margin": margin,
                    },
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
                finding = self._finding(
                    severity,
                    "model_language",
                    message,
                    row,
                    {
                        "detected": detected,
                        "script": script,
                        "model": model or None,
                        "score": round(score, 4) if score is not None else None,
                        "cjk_risk": cjk,
                        "detection_confidence": confidence,
                        "runner_up": runner_up or None,
                        "margin": margin,
                    },
                )

            if finding is None:
                mixed = detect_code_switching(text, script)
                if mixed:
                    finding = self._finding(
                        "medium",
                        "model_language",
                        f"Code-switching detected ({' / '.join(mixed)} in one record) — "
                        "a single-language sentiment model cannot score this correctly.",
                        row,
                        {
                            "detected": detected,
                            "script": script,
                            "model": model or None,
                            "languages": mixed,
                            "detection_confidence": confidence,
                        },
                    )

            if finding:
                flagged += 1
                entry["flagged"] += 1
                if weak:
                    weak_hits += 1
                findings.append(finding)

        if checked == 0:
            warnings.append(
                "No rows carried a text field (text, content, message, body, tweet, review)."
            )
        if skipped_low_confidence:
            warnings.append(
                f"{skipped_low_confidence} row(s) had too little evidence to attribute "
                "a language and were skipped."
            )

        by_language_out: dict[str, dict[str, Any]] = {}
        for lang, counts in by_language.items():
            by_language_out[lang] = {
                "total": counts["total"],
                "flagged": counts["flagged"],
                "percent_flagged": round(counts["flagged"] / counts["total"] * 100, 2),
            }

        summary: dict[str, Any] = {
            "checked": checked,
            "flagged": flagged,
            "percent_flagged": round(flagged / checked * 100, 2) if checked else 0.0,
            "weak_model_hits": weak_hits,
            "by_language": by_language_out,
            # Extended low-resource language coverage: which ones showed up in
            # this dataset (so a team knows which models it actually needs).
            "low_resource_languages": sorted(low_resource_langs),
            "low_resource_records": sum(low_resource_langs.values()),
        }
        return ToolResult(
            status="success",
            findings=findings,
            warnings=warnings,
            summary=summary,
        )

    @staticmethod
    def _finding(severity: str, field: str, message: str, row: dict, metadata: dict) -> dict:
        return {
            "severity": severity,
            "field": field,
            "message": message,
            "record_id": find_id(row),
            "metadata": metadata,
        }


def run(rows: list[dict], options: dict[str, Any] | None = None) -> ToolResult:
    """Library entry point: scan ``rows`` for language/model mismatches."""
    return LinguaCheckLogic().run(rows, options or {})
