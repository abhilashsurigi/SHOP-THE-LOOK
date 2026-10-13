from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shop_the_look.retrieval.evaluation import DEFAULT_VALIDATION
from shop_the_look.retrieval.faiss_index import DEFAULT_CATALOG, DEFAULT_OUTPUT
from shop_the_look.retrieval.visual_audit import DEFAULT_CACHE_ROOT, build_visual_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an offline visual inspection report from cached Shop-the-Look images.")
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--evaluation-dir", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "visual_audit")
    args = parser.parse_args()
    try:
        output = build_visual_audit(args.validation, args.catalog, args.evaluation_dir,
                                    args.index_dir, args.cache_dir, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    print(output / "visual_inspection.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
