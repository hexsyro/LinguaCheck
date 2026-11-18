"""End-to-end CLI tests: file in, report out, exit codes right."""

from __future__ import annotations

import csv
import json

import pytest

from linguacheck.cli import main

MIXED_CSV = """id,text,model,sentiment_score
1,"Esto es lo peor que he comprado y no lo recomiendo para nada",vader,-0.8
2,"This is the best product I have ever used and it keeps working",vader,0.9
3,"この商品は本当に素晴らしい",vader,0.5
"""


def write_csv(tmp_path, content: str):
    path = tmp_path / "data.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_json_output_envelope(tmp_path, capsys):
    path = write_csv(tmp_path, MIXED_CSV)
    assert main([str(path), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "linguacheck"
    assert payload["summary"]["checked"] == 3
    assert payload["summary"]["flagged"] == 2
    assert payload["summary"]["weak_model_hits"] == 2
    assert all(f["severity"] == "high" for f in payload["findings"])


def test_table_output(tmp_path, capsys):
    path = write_csv(tmp_path, MIXED_CSV)
    assert main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "finding(s)" in out
    assert "vader" in out


def test_csv_output_parses(tmp_path, capsys):
    path = write_csv(tmp_path, MIXED_CSV)
    assert main([str(path), "--format", "csv"]) == 0
    rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
    assert len(rows) == 2
    assert all(r["severity"] == "high" for r in rows)


def test_markdown_output(tmp_path, capsys):
    path = write_csv(tmp_path, MIXED_CSV)
    assert main([str(path), "--format", "md"]) == 0
    assert "| Severity | Field | Record | Message |" in capsys.readouterr().out


def test_fail_on_high(tmp_path):
    path = write_csv(tmp_path, MIXED_CSV)
    assert main([str(path), "--fail-on", "high"]) == 1
    assert main([str(path), "--fail-on", "none"]) == 0


def test_weak_models_flag_downgrades_non_cjk(tmp_path, capsys):
    path = write_csv(tmp_path, MIXED_CSV)
    assert main([str(path), "--weak-models", "custom-only", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    by_record = {f["record_id"]: f for f in payload["findings"]}
    # vader no longer weak: the Spanish mismatch drops to medium...
    assert by_record["1"]["severity"] == "medium"
    # ...but a CJK mismatch stays high regardless — tokenization breaks either way.
    assert by_record["3"]["severity"] == "high"
    assert payload["summary"]["weak_model_hits"] == 0


def test_config_file(tmp_path, capsys):
    path = write_csv(tmp_path, MIXED_CSV)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"weak_models": ["not-vader"]}), encoding="utf-8")
    assert main([str(path), "--config", str(config), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    by_record = {f["record_id"]: f for f in payload["findings"]}
    assert by_record["1"]["severity"] == "medium"
    assert by_record["3"]["severity"] == "high"


def test_output_file(tmp_path):
    path = write_csv(tmp_path, MIXED_CSV)
    report = tmp_path / "report.json"
    assert main([str(path), "--format", "json", "--output", str(report)]) == 0
    assert json.loads(report.read_text(encoding="utf-8"))["tool"] == "linguacheck"


def test_missing_file_exit_2(capsys):
    assert main(["nope.csv"]) == 2
    assert "file not found" in capsys.readouterr().err


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "linguacheck 0.2.0" in capsys.readouterr().out


def test_json_output_has_shared_schema_fields(tmp_path, capsys):
    path = write_csv(tmp_path, MIXED_CSV)
    assert main([str(path), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == "0.2.0"
    assert payload["summary"]["result_schema_version"] == "1.1"
    assert payload["summary"]["skipped"] is False
    assert payload["summary"]["skip_reason"] is None
    assert "language_distribution" in payload["summary"]
    assert "coverage_matrix" in payload["summary"]


def test_min_confidence_flag_gates_rows(tmp_path, capsys):
    path = write_csv(tmp_path, MIXED_CSV)
    assert main([str(path), "--min-confidence", "0.95", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["min_confidence"] == 0.95
    assert payload["summary"]["checked"] == 3
    assert payload["summary"]["flagged"] == 0
    assert payload["summary"]["skipped"] is True
    assert any("low confidence" in warning for warning in payload["warnings"])


def test_min_confidence_from_config(tmp_path, capsys):
    path = write_csv(tmp_path, MIXED_CSV)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"min_confidence": 0.95}), encoding="utf-8")
    assert main([str(path), "--config", str(config), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["min_confidence"] == 0.95
    assert payload["summary"]["flagged"] == 0


def test_explicit_flag_overrides_config_min_confidence(tmp_path, capsys):
    path = write_csv(tmp_path, MIXED_CSV)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"min_confidence": 0.95}), encoding="utf-8")
    assert main([str(path), "--config", str(config), "--min-confidence", "0.1", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["min_confidence"] == 0.1
    assert payload["summary"]["flagged"] == 2


def test_invalid_config_value_types_exit_2(tmp_path, capsys):
    path = write_csv(tmp_path, MIXED_CSV)
    for config_value in ({"weak_models": "vader"}, {"weak_models": 7}, {"min_confidence": "high"}):
        config = tmp_path / "bad-config.json"
        config.write_text(json.dumps(config_value), encoding="utf-8")
        capsys.readouterr()
        assert main([str(path), "--config", str(config)]) == 2
        error = capsys.readouterr().err
        assert "invalid config value" in error
        assert next(iter(config_value)) in error


def test_invalid_cli_min_confidence_exit_2(tmp_path, capsys):
    path = write_csv(tmp_path, MIXED_CSV)
    assert main([str(path), "--min-confidence", "1.5"]) == 2
    assert "min_confidence" in capsys.readouterr().err
