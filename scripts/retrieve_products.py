from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shop_the_look.retrieval.faiss_index import (
    DEFAULT_CATALOG, DEFAULT_EMBEDDINGS, DEFAULT_OUTPUT, RetrievalEngine,
    build_product_index, discover_embedding_bundle, write_retrieval_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an exact vector index and retrieve products for saved scenes.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-index", help="Build exact IndexFlatIP or explicitly selected NumPy index")
    build.add_argument("--backend", choices=("faiss", "numpy"), default="faiss")
    build.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    build.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    one = subparsers.add_parser("search-scene", help="Search products for one saved validation scene")
    one.add_argument("scene_id")
    one.add_argument("--top-k", type=int, default=10)
    one.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    all_scenes = subparsers.add_parser("search-all-scenes", help="Search products for every saved validation scene")
    all_scenes.add_argument("--top-k", type=int, default=10)
    all_scenes.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        if args.command == "build-index":
            metadata = build_product_index(args.embeddings_dir, args.catalog, args.output_dir, args.backend)
            engine = RetrievalEngine.load(args.output_dir, args.embeddings_dir, args.catalog)
            _, scene_summary = discover_embedding_bundle(args.embeddings_dir, "scene")
            scene_map = Path(scene_summary["_mapping_path"])
            scene_ids = [row.get("scene_id", row.get("entity_id")) for row in
                         (json.loads(line) for line in scene_map.read_text(encoding="utf-8").splitlines())]
            chosen = scene_ids[:min(5, len(scene_ids))]
            run_dir = Path(metadata["metadata_path"]).parent
            summary = write_retrieval_report(engine, chosen, 10, run_dir / "sample_retrievals.jsonl")
            print(json.dumps({"index": metadata, "sample_retrieval_summary": summary}, indent=2, sort_keys=True))
            return 0
        engine = RetrievalEngine.load(args.output_dir)
        if args.command == "search-scene":
            started = time.perf_counter()
            rows = engine.search_scene(args.scene_id, args.top_k)
            elapsed = time.perf_counter() - started
            result = {"scene_id": args.scene_id, "top_k": args.top_k, "search_seconds": elapsed, "results": rows}
            run_dir = Path(engine.metadata["index_path"]).parent
            output_path = run_dir / f"search_{args.scene_id}_top_{args.top_k}.json"
            output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        scene_ids = engine.scene_ids
        output_path = Path(engine.metadata["index_path"]).parent / f"all_scenes_top_{args.top_k}.jsonl"
        summary = write_retrieval_report(engine, scene_ids, args.top_k, output_path)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
