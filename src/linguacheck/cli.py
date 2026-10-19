"""Command-line interface for LinguaCheck.

Exit codes:
  0 — ran successfully (findings may or may not be present)
  1 — findings at or above the --fail-on severity (CI gate tripped)
  2 — usage or input error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from linguacheck import __version__
from linguacheck.common import SEVERITY_RANK, load_rows
from linguacheck.logic import LinguaCheckLogic
from linguacheck.output import render, worst_severity

TOOL = "linguacheck"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="Catch sentiment scores computed with the wrong language model: detects each "
        "record's actual language and flags mismatches, weak models, and code-switching.",
    )
    parser.add_argument("data", help="dataset file to scan (.csv, .json, or .jsonl)")
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json", "csv", "md", "markdown"],
        default="table",
        help="output format (default: table)",
    )
    parser.add_argument("--output", "-o", metavar="FILE", help="write the report to FILE instead of stdout")
    parser.add_argument(
        "--fail-on",
        choices=["none", "low", "medium", "high"],
        default="none",
        help="exit 1 if any finding is at or above this severity (default: none)",
    )
    parser.add_argument("--config", metavar="FILE", help="JSON file of options")
    parser.add_argument(
        "--weak-models",
        metavar="LIST",
        help="comma-separated model names to treat as known-weak "
        "(default: vader, afinn, textblob, and friends)",
    )
    parser.add_argument("--version", action="version", version=f"{TOOL} {__version__}")
    return parser


def _merge_options(args: argparse.Namespace) -> dict[str, Any]:
    """Config file first, then explicit CLI flags override it."""
    options: dict[str, Any] = {}
    if args.config:
        parsed = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"{args.config}: config must be a JSON object")
        options.update(parsed)
    if args.weak_models is not None:
        options["weak_models"] = [m.strip() for m in args.weak_models.split(",") if m.strip()]
    return options


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = _merge_options(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"{TOOL}: {exc}", file=sys.stderr)
        return 2

    try:
        rows = load_rows(args.data)
    except FileNotFoundError:
        print(f"{TOOL}: file not found: {args.data}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"{TOOL}: {exc}", file=sys.stderr)
        return 2

    result = LinguaCheckLogic().run(rows, options)
    text = render(result, args.format, tool=TOOL, version=__version__)

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="" if text.endswith("\n") else "\n")

    if args.fail_on != "none":
        worst = worst_severity(result)
        if worst and SEVERITY_RANK.get(worst, 0) >= SEVERITY_RANK[args.fail_on]:
            return 1
    return 0
