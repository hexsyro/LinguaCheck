"""Contract helpers: option coercion, validation, and package exports."""

from __future__ import annotations

import pytest

import linguacheck
from linguacheck.common import RESULT_SCHEMA_VERSION, opt_float, opt_list, validate_options
from linguacheck.logic import run

SPANISH = "Esto es lo peor que he comprado y no lo recomiendo para nada"


def test_package_exports_shared_contract():
    assert linguacheck.__version__ == "0.2.0"
    assert linguacheck.RESULT_SCHEMA_VERSION == "1.1"
    assert RESULT_SCHEMA_VERSION == "1.1"
    assert linguacheck.ToolResult is not None
    for name in ("RESULT_SCHEMA_VERSION", "ToolResult", "run", "__version__"):
        assert name in linguacheck.__all__


def test_opt_list_never_iterates_characters():
    assert opt_list({"weak_models": "vader"}, "weak_models", []) == ["vader"]
    assert opt_list({"weak_models": ["a", "b"]}, "weak_models", []) == ["a", "b"]
    assert opt_list({"weak_models": 7}, "weak_models", ["fallback"]) == ["fallback"]
    assert opt_list({}, "weak_models", ["fallback"]) == ["fallback"]


def test_opt_float_rejects_bad_values():
    assert opt_float({"min_confidence": 0.5}, "min_confidence", 0.2) == 0.5
    assert opt_float({"min_confidence": "0.4"}, "min_confidence", 0.2) == 0.4
    assert opt_float({"min_confidence": "oops"}, "min_confidence", 0.2) == 0.2
    assert opt_float({"min_confidence": float("nan")}, "min_confidence", 0.2) == 0.2
    assert opt_float({"min_confidence": True}, "min_confidence", 0.2) == 0.2
    assert opt_float({"min_confidence": 2.0}, "min_confidence", 0.2, high=1.0) == 0.2


def test_validate_options_reports_wrong_types():
    assert validate_options({"weak_models": ["vader"], "min_confidence": 0.3}) == []
    errors = validate_options({"weak_models": "vader", "min_confidence": "high"})
    assert len(errors) == 2
    assert any("weak_models" in error and "list" in error for error in errors)
    assert any("min_confidence" in error for error in errors)
    assert validate_options({"min_confidence": -0.1})
    assert validate_options({"weak_models": 7})


def test_run_falls_back_without_raising_on_bad_config():
    rows = [{"id": "1", "text": SPANISH, "model": "vader"}]
    result = run(rows, {"weak_models": "vader", "min_confidence": "oops"})
    assert result.status == "success"
    assert result.findings[0]["metadata"]["finding_kind"] == "language_mismatch"
    assert result.summary["min_confidence"] == 0.2


def test_run_accepts_none_options():
    result = run([{"id": "1", "text": SPANISH, "model": "vader"}])
    assert result.status == "success"
    assert result.summary["checked"] == 1


@pytest.mark.parametrize("value", [None, 0, [], {}, object()])
def test_opt_helpers_tolerate_unexpected_types(value):
    assert isinstance(opt_list({"key": value}, "key", ["fallback"]), list)
    assert isinstance(opt_float({"key": value}, "key", 0.2), float)
