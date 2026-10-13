from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shop_the_look.embedding_pipeline import (
    DEFAULT_ASSOCIATIONS, DEFAULT_CACHE, DEFAULT_CATALOG, DEFAULT_OUTPUT, generate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate resumable FashionSigLIP vectors from the local image cache.")
    parser.add_argument("--entity-type", choices=("all", "scenes", "products"), default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-products", type=int)
    parser.add_argument("--limit-scenes", type=int)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--associations", type=Path, default=DEFAULT_ASSOCIATIONS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    entity_types = {"all": ("scene", "product"), "scenes": ("scene",), "products": ("product",)}[args.entity_type]

    last_progress = time.monotonic()
    last_reported: dict[str, int] = {}

    def progress(value: dict[str, object]) -> None:
        nonlocal last_progress
        kind = str(value["entity_type"])
        attempted = int(value["attempted_pending"])
        pending = int(value["pending_total"])
        if attempted == pending or attempted - last_reported.get(kind, 0) >= args.batch_size * 20 or time.monotonic() - last_progress >= 30:
            print(json.dumps(value, sort_keys=True), flush=True)
            last_reported[kind] = attempted
            last_progress = time.monotonic()

    result = generate(args.catalog, args.associations, args.cache_dir, args.output_dir,
                      entity_types=entity_types, batch_size=args.batch_size,
                      limit_products=args.limit_products, limit_scenes=args.limit_scenes,
                      progress=progress)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(row["failed_embeddings"] == 0 and row["unresolved_embeddings"] == 0 for row in result.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
