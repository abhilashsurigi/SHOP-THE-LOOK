from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shop_the_look.retrieval.evaluation import (
    DEFAULT_EVALUATION_ROOT, DEFAULT_VALIDATION, run_evaluation,
)
from shop_the_look.retrieval.faiss_index import DEFAULT_CATALOG, DEFAULT_OUTPUT, RetrievalError


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved product retrievals against explicit validation associations.")
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retrieval-results", type=Path, default=None,
                        help="Saved scene_id/product_id/rank JSONL; defaults to active index all-scenes top-10 report")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()
    try:
        output = run_evaluation(args.validation, args.catalog, args.index_dir,
                                args.retrieval_results, args.output_dir, tuple(args.ks))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
