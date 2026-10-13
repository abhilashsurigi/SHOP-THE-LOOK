"""Persistent FAISS-only subprocess for Streamlit's live upload search."""
from __future__ import annotations

import json
import sys

import numpy as np

from shop_the_look.retrieval.faiss_index import RetrievalEngine
from shop_the_look.retrieval.multicrop import aggregate_max_scores


def _handle(engine: RetrievalEngine, request: dict) -> dict:
    queries = np.asarray(request.get("queries"), dtype=np.float32)
    mode = request.get("mode")
    top_k = int(request.get("top_k", 10))
    if queries.ndim != 2 or queries.shape[0] == 0 or queries.shape[1] != int(engine.metadata["dimension"]):
        raise ValueError("Query embeddings do not match the product index dimension.")
    if mode == "whole":
        return {"results": engine.search_embedding(queries[0], top_k)}
    if mode != "multi" or queries.shape[0] == 0:
        raise ValueError("Invalid query mode or empty crop query.")
    crop_ids = request.get("crop_ids")
    if not isinstance(crop_ids, list) or len(crop_ids) != len(queries):
        raise ValueError("Crop IDs do not match query embeddings.")
    product_ids = engine.index.product_ids
    score_matrix = np.empty((len(queries), len(product_ids)), dtype=np.float32)
    for row, query in enumerate(queries):
        full_ranking = engine.search_embedding(query, top_k=engine.index.ntotal)
        scores_by_id = {item["product_id"]: item["cosine_similarity"] for item in full_ranking}
        score_matrix[row] = [scores_by_id[product_id] for product_id in product_ids]
    ranked, _, _ = aggregate_max_scores(score_matrix, product_ids, crop_ids)
    return {"results": ranked[:top_k]}


def main() -> None:
    try:
        engine = RetrievalEngine.load()
        print(json.dumps({"ready": True, "index_count": engine.index.ntotal,
                          "dimension": engine.index.dimension}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": f"Could not load saved product index: {exc}"}), flush=True)
        return
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = _handle(engine, request)
        except Exception as exc:
            response = {"error": str(exc)}
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
