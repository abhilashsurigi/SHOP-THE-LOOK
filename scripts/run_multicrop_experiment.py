from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shop_the_look.retrieval.multicrop import (
    DEFAULT_OVERLAP, DEFAULT_SCALES, complete_multicrop_experiment, run_multicrop_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated FashionSigLIP multi-crop retrieval experiment.")
    parser.add_argument("--complete-run", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--refresh-embeddings", action="store_true",
                        help="Recompute crop embeddings even when a validated matching cache exists")
    args = parser.parse_args()
    try:
        if args.complete_run is not None:
            print(complete_multicrop_experiment(args.complete_run))
            return 0
        kwargs = {"device": args.device, "batch_size": args.batch_size, "top_k": args.top_k,
                  "scales": tuple(args.scales), "overlap": args.overlap,
                  "reuse_cache": not args.refresh_embeddings}
        if args.output_dir is not None:
            kwargs["output_root"] = args.output_dir
        output = run_multicrop_experiment(**kwargs)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
