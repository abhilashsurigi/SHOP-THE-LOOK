"""Canonicalize a product-ID JSONL catalog and write traceable audit files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO


def _parse_record(line: str) -> tuple[str | None, str | None]:
    """Return (normalized product ID, invalid reason)."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None, "malformed_json"
    if not isinstance(record, dict):
        return None, "non_object_record"
    if "product" not in record:
        return None, "missing_product"
    product = record["product"]
    if not isinstance(product, str):
        return None, "non_string_product"
    product = product.strip()
    if not product:
        return None, "blank_product"
    return product, None


def _atomic_write(path: Path, writer: Any) -> None:
    """Write one output via a temporary sibling, then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_name = stream.name
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except OSError:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        raise


def _write_json(path: Path, value: Any) -> None:
    def write(stream: TextIO) -> None:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    _atomic_write(path, write)


def canonicalize_catalog(
    input_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Read, validate, and deduplicate a JSONL product catalog.

    Invalid lines are skipped and categorized. Product IDs are normalized by
    trimming surrounding whitespace only. Canonical order follows first source
    occurrence; duplicate provenance contains every original line number.
    """
    source = Path(input_path)
    output_root = Path(output_dir)
    canonical_path = output_root / "canonical_catalog.jsonl"
    duplicate_path = output_root / "duplicate_groups.json"
    report_path = output_root / "catalog_deduplication_report.json"

    rows_by_product: dict[str, list[int]] = {}
    invalid_lines: dict[str, list[int]] = {
        "malformed_json": [],
        "non_object_record": [],
        "missing_product": [],
        "non_string_product": [],
        "blank_product": [],
        "blank_line": [],
    }
    total_lines = 0

    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                total_lines = line_number
                if not line.strip():
                    # Blank physical lines are invalid records but are distinct
                    # from JSON objects whose product value is blank.
                    invalid_lines["blank_line"].append(line_number)
                    continue
                product, reason = _parse_record(line)
                if reason is not None:
                    invalid_lines[reason].append(line_number)
                    continue
                assert product is not None
                rows_by_product.setdefault(product, []).append(line_number)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Unable to read input file '{source}': {exc}") from exc

    # Keep an explicit zero count for blank physical lines in the report.
    invalid_counts = {reason: len(lines) for reason, lines in invalid_lines.items()}
    valid_rows = total_lines - sum(invalid_counts.values())
    duplicate_products = [
        {"product": product, "occurrence_count": len(lines), "source_lines": lines}
        for product, lines in rows_by_product.items()
        if len(lines) > 1
    ]
    duplicate_occurrences = sum(group["occurrence_count"] - 1 for group in duplicate_products)
    canonical_records = [{"product": product} for product in rows_by_product]

    duplicate_payload = {
        "source_file": str(source),
        "duplicate_group_count": len(duplicate_products),
        "duplicate_groups": duplicate_products,
    }
    report = {
        "input_file": str(source),
        "total_source_lines": total_lines,
        "valid_rows": valid_rows,
        "unique_valid_product_ids": len(rows_by_product),
        "duplicate_groups": len(duplicate_products),
        "duplicate_occurrences_beyond_first": duplicate_occurrences,
        "invalid_record_counts_by_reason": invalid_counts,
        "invalid_source_lines_by_reason": invalid_lines,
        "canonical_output_row_count": len(canonical_records),
        "output_paths": {
            "canonical_catalog": str(canonical_path),
            "duplicate_groups": str(duplicate_path),
            "report": str(report_path),
        },
        "execution_status": "success",
    }

    try:
        _atomic_write(
            canonical_path,
            lambda stream: [
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                for record in canonical_records
            ],
        )
        _write_json(duplicate_path, duplicate_payload)
        _write_json(report_path, report)
    except (OSError, UnicodeError, TypeError) as exc:
        raise ValueError(f"Unable to write outputs under '{output_root}': {exc}") from exc
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic canonical catalog and duplicate audit."
    )
    parser.add_argument(
        "--input", "-i", default="dataset/product_catelog.jsonl",
        help="Source product catalog JSONL (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", "-o", default="audit_output",
        help="Directory for generated artifacts (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = canonicalize_catalog(args.input, args.output_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "Catalog canonicalization complete: "
        f"source_lines={report['total_source_lines']}, "
        f"valid_rows={report['valid_rows']}, "
        f"canonical_rows={report['canonical_output_row_count']}, "
        f"duplicate_groups={report['duplicate_groups']}, "
        f"duplicate_occurrences_beyond_first={report['duplicate_occurrences_beyond_first']}, "
        f"invalid_rows={sum(report['invalid_record_counts_by_reason'].values())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
