"""Rule-by-rule tests for the LinguaCheck scanner."""

from __future__ import annotations

from linguacheck.logic import LinguaCheckLogic

logic = LinguaCheckLogic()

SPANISH = "Esto es lo peor que he comprado y no lo recomiendo para nada"
ENGLISH = "This is the best product I have ever used and it keeps working"
JAPANESE = "この商品は本当に素晴らしい"
CHINESE = "这个产品非常好用"
RUSSIAN = "Этот продукт действительно ужасен и я его ненавижу"
WELSH = "Mae cynnyrch yn dda iawn ac rwyf wrth fy modd"
GIBBERISH = "xkcd qqq zzz www"


def test_vader_on_spanish_is_high():
    rows = [{"id": "1", "text": SPANISH, "model": "vader", "sentiment_score": -0.8}]
    result = logic.run(rows, {})
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding["severity"] == "high"
    assert finding["metadata"]["detected"] == "es"
    assert finding["metadata"]["assumed"] == "en"
    assert finding["metadata"]["model"] == "vader"
    assert "scored with an 'en'-tuned model" in finding["message"]


def test_vader_on_english_is_clean():
    rows = [{"id": "1", "text": ENGLISH, "model": "vader", "sentiment_score": 0.9}]
    result = logic.run(rows, {})
    assert result.findings == []
    assert result.summary["checked"] == 1
    assert result.summary["flagged"] == 0


def test_cjk_with_vader_flags_tokenization_risk():
    rows = [{"id": "1", "text": JAPANESE, "model": "vader", "sentiment_score": 0.5}]
    result = logic.run(rows, {})
    finding = result.findings[0]
    assert finding["severity"] == "high"
    assert finding["metadata"]["cjk_risk"] is True
    assert "word-boundary tokenization" in finding["message"]


def test_weak_model_cjk_message_when_assumed_matches():
    rows = [{"id": "1", "text": CHINESE, "model": "vader", "model_language": "zh"}]
    result = logic.run(rows, {})
    finding = result.findings[0]
    assert finding["severity"] == "high"
    assert "Known-weak model" in finding["message"]
    assert finding["metadata"]["cjk_risk"] is True


def test_custom_model_mismatch_is_medium():
    rows = [{"id": "1", "text": SPANISH, "model": "custom-bert", "model_language": "en"}]
    result = logic.run(rows, {})
    finding = result.findings[0]
    assert finding["severity"] == "medium"
    assert finding["metadata"]["model"] == "custom-bert"


def test_unknown_model_without_language_column_stays_quiet():
    # A feed with unrelated score columns must not produce false accusations.
    rows = [{"id": "1", "text": SPANISH, "model": "some-internal-model", "score": 0.4}]
    result = logic.run(rows, {})
    assert result.findings == []


def test_russian_with_vader():
    rows = [{"id": "1", "text": RUSSIAN, "model": "vader"}]
    result = logic.run(rows, {})
    assert result.findings[0]["severity"] == "high"
    assert result.findings[0]["metadata"]["detected"] == "ru"


def test_assumed_language_column_wins():
    rows = [{"id": "1", "text": SPANISH, "model": "custom", "assumed_language": "English"}]
    result = logic.run(rows, {})
    assert result.findings[0]["metadata"]["assumed"] == "en"


def test_code_switching_flagged_when_no_mismatch():
    rows = [{"id": "1", "text": "The product is good. El producto es bueno."}]
    result = logic.run(rows, {})
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding["severity"] == "medium"
    assert finding["metadata"]["languages"] == ["en", "es"]
    assert "Code-switching" in finding["message"]


def test_low_confidence_rows_skipped_with_warning():
    rows = [{"id": "1", "text": GIBBERISH, "model": "vader"}]
    result = logic.run(rows, {})
    assert result.findings == []
    assert result.summary["checked"] == 1
    assert result.summary["flagged"] == 0
    assert any("too little evidence" in w for w in result.warnings)


def test_rows_without_text_skipped():
    rows = [{"id": "1", "sentiment_score": 0.5}]
    result = logic.run(rows, {})
    assert result.summary["checked"] == 0
    assert any("No rows carried a text field" in w for w in result.warnings)


def test_summary_by_language():
    rows = [
        {"id": "1", "text": SPANISH, "model": "vader"},
        {"id": "2", "text": SPANISH, "model": "custom-bert", "model_language": "es"},
        {"id": "3", "text": ENGLISH, "model": "vader"},
    ]
    result = logic.run(rows, {})
    by_lang = result.summary["by_language"]
    assert by_lang["es"]["total"] == 2
    assert by_lang["es"]["flagged"] == 1
    assert by_lang["es"]["percent_flagged"] == 50.0
    assert by_lang["en"]["total"] == 1
    assert by_lang["en"]["flagged"] == 0


def test_low_resource_languages_reported():
    rows = [{"id": "1", "text": WELSH}]
    result = logic.run(rows, {})
    assert result.summary["low_resource_languages"] == ["cy"]
    assert result.summary["low_resource_records"] == 1
    assert result.findings == []  # no model on the row -> no mismatch to claim


def test_weak_models_option_replaces_defaults():
    rows = [{"id": "1", "text": SPANISH, "model": "vader"}]
    result = logic.run(rows, {"weak_models": ["custom-only"]})
    # vader no longer "weak": mismatch still real, but medium, not high.
    assert result.findings[0]["severity"] == "medium"
    assert result.summary["weak_model_hits"] == 0


def test_weak_model_hits_counted():
    rows = [
        {"id": "1", "text": SPANISH, "model": "vader"},
        {"id": "2", "text": RUSSIAN, "model": "textblob"},
    ]
    result = logic.run(rows, {})
    assert result.summary["weak_model_hits"] == 2
    assert result.summary["flagged"] == 2
    assert result.summary["percent_flagged"] == 100.0


def test_summary_has_shared_schema_fields():
    result = logic.run([{"id": "1", "text": SPANISH, "model": "vader"}], {})
    assert result.status == "success"
    assert result.summary["result_schema_version"] == "1.1"
    assert result.summary["skipped"] is False
    assert result.summary["skip_reason"] is None


def test_summary_skipped_when_no_text_rows():
    result = logic.run([{"id": "1", "score": 0.5}], {})
    assert result.summary["skipped"] is True
    assert "text" in result.summary["skip_reason"]
    assert result.summary["flagged"] == 0


def test_summary_skipped_when_all_rows_low_confidence():
    result = logic.run([{"id": "1", "text": GIBBERISH, "model": "vader"}], {})
    assert result.summary["checked"] == 1
    assert result.summary["skipped"] is True
    assert result.summary["skip_reason"] is not None
    assert any("low confidence" in warning for warning in result.warnings)
    assert any("too little evidence" in warning for warning in result.warnings)


def test_mismatch_and_code_switching_produce_two_findings():
    rows = [
        {
            "id": "mixed",
            "text": "The product is good. El producto es bueno.",
            "model": "vader",
            "model_language": "es",
        }
    ]
    result = logic.run(rows, {})
    kinds = [finding["metadata"]["finding_kind"] for finding in result.findings]
    assert kinds == ["language_mismatch", "code_switching"]
    assert [finding["severity"] for finding in result.findings] == ["high", "medium"]
    assert result.summary["flagged"] == 1  # one row, two findings


def test_weak_model_finding_kind():
    rows = [{"id": "1", "text": SPANISH, "model": "vader", "model_language": "es"}]
    result = logic.run(rows, {})
    assert len(result.findings) == 1
    assert result.findings[0]["metadata"]["finding_kind"] == "weak_model"


def test_language_distribution_summary_and_finding_metadata():
    rows = [
        {
            "id": "mixed",
            "text": "The product is good. El producto es bueno.",
            "model": "vader",
            "model_language": "es",
        }
    ]
    result = logic.run(rows, {})
    entries = [entry for entry in result.summary["language_distribution"] if entry["record_id"] == "mixed"]
    assert {entry["lang"] for entry in entries} == {"en", "es"}
    assert all(entry["share"] == 0.5 for entry in entries)
    assert all(entry["languages"] == ["en", "es"] for entry in entries)
    assert all(finding["metadata"]["language_distribution"] == {"en": 0.5, "es": 0.5} for finding in result.findings)


def test_mixed_language_unscorable_finding_when_no_dominant_share():
    rows = [{"id": "u1", "text": "the and el la de que", "model": "custom", "model_language": "en"}]
    result = logic.run(rows, {})
    kinds = [finding["metadata"]["finding_kind"] for finding in result.findings]
    assert kinds == ["language_mismatch", "mixed_language_unscorable"]
    unscorable = result.findings[-1]
    assert unscorable["severity"] == "medium"
    assert unscorable["field"] == "language_distribution"
    assert ">=0.8" in unscorable["message"]


def test_min_confidence_option_controls_gate():
    rows = [{"id": "1", "text": "the and", "model": "vader", "model_language": "es"}]
    assert logic.run(rows, {}).findings != []
    strict = logic.run(rows, {"min_confidence": 0.95})
    assert strict.findings == []
    assert strict.summary["min_confidence"] == 0.95
    assert strict.summary["skipped"] is True
    assert any("low confidence" in warning for warning in strict.warnings)


def test_coverage_matrix_actions_and_risk():
    rows = [
        {"id": "1", "text": SPANISH, "model": "custom", "model_language": "en"},
        {"id": "2", "text": JAPANESE, "model": "custom", "model_language": "en"},
        {"id": "3", "text": WELSH, "model": "custom", "model_language": "en"},
        {"id": "4", "text": ENGLISH, "model": "custom", "model_language": "en"},
    ]
    matrix = {(entry["detected"], entry["assumed"]): entry for entry in logic.run(rows, {}).summary["coverage_matrix"]}
    assert matrix[("es", "en")] == {
        "detected": "es",
        "assumed": "en",
        "count": 1,
        "risk": "medium",
        "action": "reroute",
    }
    assert matrix[("ja", "en")]["risk"] == "high"
    assert matrix[("ja", "en")]["action"] == "translate-then-score"
    assert matrix[("cy", "en")]["risk"] == "high"
    assert matrix[("cy", "en")]["action"] == "mark-unscorable"
    assert matrix[("en", "en")]["action"] == "ok"


def test_weak_models_string_is_not_iterated_by_character():
    rows = [{"id": "1", "text": SPANISH, "model": "vader"}]
    result = logic.run(rows, {"weak_models": "vader"})
    assert result.findings[0]["severity"] == "high"
    assert result.summary["weak_model_hits"] == 1


def test_malformed_options_never_raise_from_run():
    rows = [{"id": "1", "text": SPANISH, "model": "vader"}]
    for options in (
        {"weak_models": 42},
        {"weak_models": ["ok", 7, None]},
        {"min_confidence": "not-a-number"},
        {"min_confidence": None},
        {"min_confidence": float("nan")},
        {"min_confidence": 5.0},
    ):
        result = logic.run(rows, options)
        assert result.status == "success"
        assert result.summary["result_schema_version"] == "1.1"
        assert result.summary["min_confidence"] == 0.2
