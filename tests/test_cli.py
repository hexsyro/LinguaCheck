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
    assert "linguacheck 0.1.0" in capsys.readouterr().out
