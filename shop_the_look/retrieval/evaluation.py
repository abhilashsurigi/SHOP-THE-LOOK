"""Reproducible retrieval evaluation against explicit validation pairs."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .faiss_index import DEFAULT_CATALOG, DEFAULT_OUTPUT, RetrievalError, _atomic_json, _atomic_jsonl, _read_jsonl, sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALIDATION = ROOT / "dataset" / "validation.jsonl"
DEFAULT_EVALUATION_ROOT = ROOT / "artifacts" / "evaluation"
DEFAULT_KS = (1, 5, 10)


def read_validation(path: Path) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Read every validation row, rejecting malformed records and preserving pair duplicates in counts."""
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RetrievalError(f"Malformed validation JSONL at line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise RetrievalError(f"Expected validation object at line {line_no}")
            scene, product = row.get("scene"), row.get("product")
            if not isinstance(scene, str) or not scene.strip():
                raise RetrievalError(f"Missing or blank scene ID at validation line {line_no}")
            if not isinstance(product, str) or not product.strip():
                raise RetrievalError(f"Missing or blank product ID at validation line {line_no}")
            # Match the existing pipeline's whitespace cleanup while preserving case.
            pairs.append((scene.strip(), product.strip()))
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for pair in pairs:
        pair_counts[pair] += 1
    info = {
        "row_count": len(pairs),
        "unique_pair_count": len(pair_counts),
        "duplicate_pair_rows": len(pairs) - len(pair_counts),
        "duplicate_pairs": [
            {"scene_id": scene, "product_id": product, "row_count": count, "duplicate_rows": count - 1}
            for (scene, product), count in sorted(pair_counts.items()) if count > 1
        ],
    }
    return pairs, info


def read_id_set(path: Path, field: str) -> set[str]:
    values: set[str] = set()
    for line_no, row in enumerate(_read_jsonl(path), 1):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise RetrievalError(f"Missing {field} at {path}:{line_no}")
        value = value.strip()
        if value in values:
            raise RetrievalError(f"Duplicate {field} {value!r} in {path}")
        values.add(value)
    return values


def read_catalog_ids(path: Path) -> set[str]:
    values: set[str] = set()
    for line_no, row in enumerate(_read_jsonl(path), 1):
        value = row.get("product")
        if not isinstance(value, str) or not value.strip():
            raise RetrievalError(f"Missing product ID at {path}:{line_no}")
        value = value.strip()
        if value in values:
            raise RetrievalError(f"Duplicate canonical catalog product ID {value!r}")
        values.add(value)
    return values


def read_retrieval_results(path: Path, expected_scenes: set[str], expected_products: set[str]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line_no, row in enumerate(_read_jsonl(path), 1):
        scene, product, rank = row.get("scene_id"), row.get("product_id"), row.get("rank")
        if not isinstance(scene, str) or not scene.strip() or not isinstance(product, str) or not product.strip():
            raise RetrievalError(f"Missing scene_id/product_id in retrieval result at line {line_no}")
        scene, product = scene.strip(), product.strip()
        if scene not in expected_scenes:
            raise RetrievalError(f"Unknown retrieval scene {scene!r} at line {line_no}")
        if product not in expected_products:
            raise RetrievalError(f"Retrieved product {product!r} is absent from the catalog/index")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise RetrievalError(f"Invalid retrieval rank at line {line_no}: {rank!r}")
        score = row.get("cosine_similarity")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise RetrievalError(f"Invalid cosine_similarity at line {line_no}")
        grouped[scene].append({"product_id": product, "rank": rank, "cosine_similarity": float(score)})
    for scene in expected_scenes:
        rows = grouped.get(scene, [])
        rows.sort(key=lambda item: item["rank"])
        ranks = [item["rank"] for item in rows]
        if ranks != list(range(1, len(rows) + 1)):
            raise RetrievalError(f"Retrieval ranks for scene {scene!r} must be contiguous from 1")
        ids = [item["product_id"] for item in rows]
        if len(ids) != len(set(ids)):
            raise RetrievalError(f"Duplicate product result for scene {scene!r}")
        ordered = sorted(rows, key=lambda item: (-item["cosine_similarity"], item["product_id"]))
        if [item["product_id"] for item in rows] != [item["product_id"] for item in ordered]:
            raise RetrievalError(f"Retrieval scores are not deterministically descending for scene {scene!r}")
    return dict(grouped), max((len(rows) for rows in grouped.values()), default=0)


def _ndcg(relevant: set[str], retrieved: list[dict[str, Any]], k: int) -> float:
    ranks = {row["product_id"]: row["rank"] for row in retrieved if row["rank"] <= k}
    dcg = sum(1.0 / math.log2(rank + 1) for product, rank in ranks.items() if product in relevant)
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0


def evaluate(pairs: Iterable[tuple[str, str]], catalog_ids: set[str], index_ids: set[str],
             retrieval: dict[str, list[dict[str, Any]]], ks: tuple[int, ...] = DEFAULT_KS) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in ks) or len(set(ks)) != len(ks):
        raise RetrievalError("ks must contain distinct positive integers")
    pairs = list(pairs)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for scene, product in pairs:
        counts[(scene, product)] += 1
    unique_pairs = sorted(counts)
    scene_targets: dict[str, set[str]] = defaultdict(set)
    for scene, product in unique_pairs:
        scene_targets[scene].add(product)
    all_scenes = sorted(scene_targets)
    raw_product_ids = {product for _, product in unique_pairs}
    target_missing_catalog = raw_product_ids - catalog_ids
    target_missing_index = raw_product_ids - index_ids
    target_eligible = raw_product_ids & catalog_ids & index_ids
    per_scene: list[dict[str, Any]] = []
    for scene in all_scenes:
        targets = scene_targets[scene]
        eligible = targets & catalog_ids & index_ids
        missing_catalog = targets - catalog_ids
        missing_index = targets - index_ids
        results = retrieval.get(scene, [])
        rank_by_id = {row["product_id"]: row["rank"] for row in results}
        relevant_found = sorted(
            ({"product_id": product, "rank": rank_by_id[product]} for product in eligible if product in rank_by_id),
            key=lambda row: (row["rank"], row["product_id"]),
        )
        missing_rank = sorted(eligible - set(rank_by_id))
        metrics: dict[str, Any] = {}
        if eligible:
            for k in ks:
                found = sum(1 for product in eligible if rank_by_id.get(product, k + 1) <= k)
                metrics[f"recall_at_{k}"] = found / len(eligible)
                metrics[f"hit_rate_at_{k}"] = float(found > 0)
                metrics[f"ndcg_at_{k}"] = _ndcg(eligible, results, k)
            first_rank = min((row["rank"] for row in relevant_found if row["rank"] <= max(ks)), default=None)
            metrics[f"mrr_at_{max(ks)}"] = 1.0 / first_rank if first_rank is not None else 0.0
        else:
            for k in ks:
                metrics[f"recall_at_{k}"] = None
                metrics[f"hit_rate_at_{k}"] = None
                metrics[f"ndcg_at_{k}"] = None
            metrics[f"mrr_at_{max(ks)}"] = None
        per_scene.append({
            "scene_id": scene,
            "ground_truth_unique_count": len(targets),
            "eligible_ground_truth_count": len(eligible),
            "missing_catalog_count": len(missing_catalog),
            "missing_index_count": len(missing_index),
            "duplicate_pair_rows": sum(counts[(scene, product)] - 1 for product in targets),
            "ground_truth_product_ids": sorted(targets),
            "eligible_product_ids": sorted(eligible),
            "missing_catalog_product_ids": sorted(missing_catalog),
            "missing_index_product_ids": sorted(missing_index),
            "retrieved_relevant": relevant_found,
            "missed_eligible_product_ids": missing_rank,
            "retrieved_relevant_count_at_k": {str(k): sum(row["rank"] <= k for row in relevant_found) for k in ks},
            "metrics": metrics,
        })

    eligible_scenes = [row for row in per_scene if row["eligible_ground_truth_count"] > 0]
    metric_summary: dict[str, Any] = {}
    for k in ks:
        assoc_hits = sum(row["retrieved_relevant_count_at_k"][str(k)] for row in per_scene)
        eligible_assoc_denominator = sum(row["eligible_ground_truth_count"] for row in per_scene)
        eligible_scene_denominator = len(eligible_scenes)
        mean_recall_sum = sum(row["metrics"][f"recall_at_{k}"] for row in eligible_scenes)
        hits = sum(row["metrics"][f"hit_rate_at_{k}"] for row in eligible_scenes)
        ndcg_sum = sum(row["metrics"][f"ndcg_at_{k}"] for row in eligible_scenes)
        metric_summary[f"recall_at_{k}"] = {
            "mean_per_eligible_scene": mean_recall_sum / eligible_scene_denominator if eligible_scene_denominator else None,
            "mean_per_scene_numerator": mean_recall_sum,
            "mean_per_scene_denominator": eligible_scene_denominator,
            "association_level": assoc_hits / eligible_assoc_denominator if eligible_assoc_denominator else None,
            "association_numerator": assoc_hits,
            "association_denominator_eligible": eligible_assoc_denominator,
            "all_association_adjusted": assoc_hits / len(unique_pairs) if unique_pairs else None,
            "all_association_numerator": assoc_hits,
            "all_association_denominator_including_unavailable": len(unique_pairs),
        }
        metric_summary[f"hit_rate_at_{k}"] = {
            "value": hits / eligible_scene_denominator if eligible_scene_denominator else None,
            "numerator_scenes_with_hit": int(hits), "denominator_eligible_scenes": eligible_scene_denominator,
        }
        metric_summary[f"ndcg_at_{k}"] = {
            "value": ndcg_sum / eligible_scene_denominator if eligible_scene_denominator else None,
            "sum_per_scene_ndcg": ndcg_sum, "denominator_eligible_scenes": eligible_scene_denominator,
        }
    reciprocal_sum = sum(row["metrics"][f"mrr_at_{max(ks)}"] for row in eligible_scenes)
    metric_summary[f"mrr_at_{max(ks)}"] = {
        "value": reciprocal_sum / len(eligible_scenes) if eligible_scenes else None,
        "sum_reciprocal_ranks": reciprocal_sum, "denominator_eligible_scenes": len(eligible_scenes),
    }
    all_product_count = len(raw_product_ids)
    summary = {
        "validation": {
            "row_count": len(pairs), "unique_association_count": len(unique_pairs),
            "duplicate_pair_rows": len(pairs) - len(unique_pairs), "unique_scene_count": len(all_scenes),
            "unique_target_product_count": all_product_count,
            "association_rows_for_missing_catalog_products": sum(counts[(s, p)] for s, p in unique_pairs if p not in catalog_ids),
            "unique_pairs_for_missing_catalog_products": sum(1 for _, p in unique_pairs if p not in catalog_ids),
            "unique_missing_catalog_product_count": len(target_missing_catalog),
            "association_rows_for_missing_index_products": sum(counts[(s, p)] for s, p in unique_pairs if p not in index_ids),
            "unique_pairs_for_missing_index_products": sum(1 for _, p in unique_pairs if p not in index_ids),
            "unique_missing_index_product_count": len(target_missing_index),
        },
        "coverage": {
            "catalog_product_count": len(catalog_ids), "index_product_count": len(index_ids),
            "catalog_index_id_sets_equal": catalog_ids == index_ids,
            "eligible_unique_target_product_count": len(target_eligible),
            "target_product_catalog_and_index_coverage": len(target_eligible) / all_product_count if all_product_count else None,
            "eligible_unique_association_count": sum(1 for _, p in unique_pairs if p in catalog_ids and p in index_ids),
            "unique_association_coverage": sum(1 for _, p in unique_pairs if p in catalog_ids and p in index_ids) / len(unique_pairs) if unique_pairs else None,
            "eligible_scene_count": len(eligible_scenes),
            "zero_eligible_ground_truth_scene_count": len(all_scenes) - len(eligible_scenes),
        },
        "metrics": metric_summary,
        "metric_semantics": {
            "conditional_metrics": "Recall, hit rate, MRR and NDCG denominators include only validation products present in both canonical catalog and retrieval index; scores are conditional on catalog/index coverage.",
            "all_association_adjusted_recall": "Additional association-level recall uses all unique validation pairs as denominator. Missing targets remain unavailable and are explicitly counted in coverage.",
            "duplicate_pairs": "Duplicate validation scene-product rows are counted and reported, but each unique pair contributes once to metrics.",
            "relevance": "Binary relevance from an explicit validation pair only; unlabeled products are not assumed irrelevant or exhaustively labeled.",
            "ndcg": "Binary relevance with log2(rank+1) discount and ideal ranking over eligible products, truncated at K.",
        },
    }
    breakdown: dict[str, dict[str, Any]] = {}
    for count in sorted({row["eligible_ground_truth_count"] for row in per_scene}):
        group = [row for row in per_scene if row["eligible_ground_truth_count"] == count]
        breakdown[str(count)] = {"scene_count": len(group), "eligible_scene_count": sum(bool(r["eligible_ground_truth_count"]) for r in group)}
        for k in ks:
            valid = [r for r in group if r["metrics"][f"recall_at_{k}"] is not None]
            breakdown[str(count)][f"mean_recall_at_{k}"] = sum(r["metrics"][f"recall_at_{k}"] for r in valid) / len(valid) if valid else None
            breakdown[str(count)][f"hit_rate_at_{k}"] = sum(r["metrics"][f"hit_rate_at_{k}"] for r in valid) / len(valid) if valid else None
    summary["breakdown_by_eligible_product_count"] = breakdown
    diagnostics = {
        "zero_eligible_scenes": [r["scene_id"] for r in per_scene if r["eligible_ground_truth_count"] == 0],
        "no_hit_at_10_scenes": [r["scene_id"] for r in per_scene if r["eligible_ground_truth_count"] and not r["retrieved_relevant_count_at_k"].get("10", 0)],
        "coverage_limited_scenes": [
            {"scene_id": r["scene_id"], "missing_catalog_product_ids": r["missing_catalog_product_ids"],
             "missing_index_product_ids": r["missing_index_product_ids"]}
            for r in per_scene if r["missing_catalog_count"] or r["missing_index_count"]
        ],
        "best_cases_by_recall_at_10": sorted(
            (r for r in eligible_scenes), key=lambda r: (-r["metrics"]["recall_at_10"], -r["metrics"]["hit_rate_at_10"], r["scene_id"])
        )[:10],
        "worst_cases_by_recall_at_10": sorted(
            eligible_scenes, key=lambda r: (r["metrics"]["recall_at_10"], -r["eligible_ground_truth_count"], r["scene_id"])
        )[:10],
        "high_value_failure_cases": sorted(
            (r for r in eligible_scenes if r["missed_eligible_product_ids"]),
            key=lambda r: (-len(r["missed_eligible_product_ids"]), -r["eligible_ground_truth_count"],
                           r["metrics"]["recall_at_10"], r["scene_id"]),
        )[:10],
    }
    return summary, per_scene, diagnostics


def _markdown(report: dict[str, Any], diagnostics: dict[str, Any], fingerprint: str) -> str:
    v, cov, metrics = report["validation"], report["coverage"], report["metrics"]
    lines = ["# FashionSigLIP + FAISS retrieval evaluation", "", f"Evaluation fingerprint: `{fingerprint}`", "",
             "## Validation and coverage", "",
             f"- Raw validation rows: {v['row_count']}; unique scene-product pairs: {v['unique_association_count']}; duplicate rows: {v['duplicate_pair_rows']}.",
             f"- Scenes: {v['unique_scene_count']}; unique target products: {v['unique_target_product_count']}.",
             f"- Eligible unique pairs in catalog and index: {cov['eligible_unique_association_count']}/{v['unique_association_count']} ({cov['unique_association_coverage']:.1%}).",
             f"- Missing-catalog pairs: {v['unique_pairs_for_missing_catalog_products']} involving {v['unique_missing_catalog_product_count']} distinct products; missing-index pairs: {v['unique_pairs_for_missing_index_products']} involving {v['unique_missing_index_product_count']} distinct products.",
             f"- Eligible scenes: {cov['eligible_scene_count']}; zero-eligible scenes: {cov['zero_eligible_ground_truth_scene_count']}; eligible scenes with no top-10 hit: {len(diagnostics['no_hit_at_10_scenes'])}.", "",
             "## Metrics", "", "Conditional metrics use eligible products (present in both catalog and index). Aggregate-adjusted recall additionally counts unavailable unique pairs in its denominator.", "",
             "| Metric | Value | Numerator / denominator |", "|---|---:|---:|"]
    for k in report["configuration"]["ks"]:
        m = metrics[f"recall_at_{k}"]
        lines.append(f"| Mean per-scene Recall@{k} | {m['mean_per_eligible_scene']:.4f} | sum scene recall {m['mean_per_scene_numerator']:.4f} / {m['mean_per_scene_denominator']} eligible scenes |" if m['mean_per_eligible_scene'] is not None else f"| Mean per-scene Recall@{k} | n/a | 0 eligible scenes |")
        lines.append(f"| Association Recall@{k} (conditional) | {m['association_level']:.4f} | {m['association_numerator']} / {m['association_denominator_eligible']} eligible pairs |" if m['association_level'] is not None else f"| Association Recall@{k} (conditional) | n/a | 0 eligible pairs |")
        lines.append(f"| Association Recall@{k} (all-pair adjusted) | {m['all_association_adjusted']:.4f} | {m['all_association_numerator']} / {m['all_association_denominator_including_unavailable']} unique pairs (includes unavailable targets) |" if m['all_association_adjusted'] is not None else f"| Association Recall@{k} (all-pair adjusted) | n/a | no pairs |")
        h = metrics[f"hit_rate_at_{k}"]
        lines.append(f"| Hit Rate@{k} | {h['value']:.4f} | {h['numerator_scenes_with_hit']} / {h['denominator_eligible_scenes']} eligible scenes |" if h['value'] is not None else f"| Hit Rate@{k} | n/a | 0 eligible scenes |")
        n = metrics[f"ndcg_at_{k}"]
        lines.append(f"| NDCG@{k} | {n['value']:.4f} | {n['denominator_eligible_scenes']} eligible scenes |" if n['value'] is not None else f"| NDCG@{k} | n/a | 0 eligible scenes |")
    mrr_k = max(report["configuration"]["ks"])
    mrr = metrics[f"mrr_at_{mrr_k}"]
    lines.extend([f"| MRR@{mrr_k} | {mrr['value']:.4f} | {mrr['denominator_eligible_scenes']} eligible scenes |" if mrr['value'] is not None else f"| MRR@{mrr_k} | n/a | 0 eligible scenes |", "",
                  "## Failure cases for manual review", "", "Scenes below have eligible validation products absent from the top 10. Ordering: missed eligible products descending, then ground-truth count descending, Recall@10 ascending, scene ID ascending.", ""])
    for row in diagnostics["high_value_failure_cases"]:
        ranks = ", ".join(f"{item['product_id']} (rank {item['rank']})" for item in row["retrieved_relevant"] if item["rank"] <= 10) or "none in top 10"
        lines.append(f"- `{row['scene_id']}` — eligible {row['eligible_ground_truth_count']}, missing from top 10: {', '.join(row['missed_eligible_product_ids'])}; found relevant: {ranks}.")
    lines.extend(["", "## Method and caveats", "", report["metric_semantics"]["relevance"],
                  report["metric_semantics"]["duplicate_pairs"], report["metric_semantics"]["conditional_metrics"],
                  "Validation associations are not assumed exhaustive: unlabelled retrieved products are not treated as false positives. The metrics assess exact labeled product retrieval and do not prove visual correctness for unlabeled catalog items.", ""])
    return "\n".join(lines)


def run_evaluation(validation_path: Path = DEFAULT_VALIDATION, catalog_path: Path = DEFAULT_CATALOG,
                   index_root: Path = DEFAULT_OUTPUT, retrieval_path: Path | None = None,
                   output_root: Path = DEFAULT_EVALUATION_ROOT, ks: tuple[int, ...] = DEFAULT_KS) -> Path:
    active = json.loads((index_root / "active_index.json").read_text(encoding="utf-8"))
    index_dir = Path(active["index_metadata_path"]).parent
    metadata_path = index_dir / "index_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("index_fingerprint") != active.get("index_fingerprint"):
        raise RetrievalError("Active index pointer fingerprint disagrees with its metadata")
    retrieval_path = retrieval_path or index_dir / f"all_scenes_top_{max(ks)}.jsonl"
    pairs, validation_info = read_validation(validation_path)
    catalog_ids = read_catalog_ids(catalog_path)
    index_ids = read_id_set(Path(metadata["product_id_mapping_path"]), "product_id")
    if len(index_ids) != metadata.get("ntotal"):
        raise RetrievalError("Index product ID mapping count disagrees with index metadata")
    mapping_sha256 = sha256_file(Path(metadata["product_id_mapping_path"]))
    if mapping_sha256 != metadata.get("product_id_mapping_sha256"):
        raise RetrievalError("Index product ID mapping checksum differs from index metadata")
    if sha256_file(Path(metadata["index_path"])) != metadata.get("index_sha256"):
        raise RetrievalError("Saved index checksum differs from index metadata")
    expected_scenes = {scene for scene, _ in pairs}
    retrieval, max_retrieved_k = read_retrieval_results(retrieval_path, expected_scenes, index_ids)
    absent_result_scenes = expected_scenes - set(retrieval)
    if absent_result_scenes:
        raise RetrievalError(f"No saved retrieval results for {len(absent_result_scenes)} validation scenes")
    summary, per_scene, diagnostics = evaluate(pairs, catalog_ids, index_ids, retrieval, ks)
    summary["validation"].update(validation_info)
    input_fingerprints = {
        "validation_sha256": sha256_file(validation_path), "catalog_sha256": sha256_file(catalog_path),
        "retrieval_results_sha256": sha256_file(retrieval_path), "index_metadata_sha256": sha256_file(metadata_path),
        "index_sha256": metadata["index_sha256"], "product_id_mapping_sha256": mapping_sha256,
        "product_embedding_summary_sha256": sha256_file(Path(metadata["product_embedding_summary_path"])),
        "scene_embedding_summary_sha256": sha256_file(Path(metadata["scene_embedding_summary_path"])),
    }
    config = {"ks": list(ks), "relevance": "explicit_unique_scene_product_pair", "duplicate_pair_policy": "count_and_deduplicate_for_metrics", "index_fingerprint": metadata["index_fingerprint"]}
    fingerprint = hashlib.sha256(json.dumps({"inputs": input_fingerprints, "config": config}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    out = output_root / fingerprint
    report = {"schema_version": 1, "evaluation_fingerprint": fingerprint, "configuration": config,
              "input_fingerprints": input_fingerprints, "inputs": {"validation_path": str(validation_path.resolve()),
              "catalog_path": str(catalog_path.resolve()), "retrieval_results_path": str(retrieval_path.resolve()),
              "index_metadata_path": str(metadata_path.resolve())}, "retrieval_index": {
              "backend": metadata.get("backend"), "index_type": metadata.get("index_type"), "index_fingerprint": metadata.get("index_fingerprint"),
        "index_catalog_fingerprint_matches_input": metadata.get("catalog_fingerprint") == sha256_file(catalog_path),
              "model_id": metadata.get("model_id"), "model_revision": metadata.get("model_revision"),
              "product_count": metadata.get("ntotal"), "dimension": metadata.get("dimension"),
              "top_k_available": max_retrieved_k}, **summary}
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "evaluation_report.json", report)
    _atomic_jsonl(out / "per_scene_metrics.jsonl", per_scene)
    _atomic_json(out / "retrieval_diagnostics.json", diagnostics)
    markdown = _markdown(report, diagnostics, fingerprint)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=out, prefix=".summary.", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            stream.write(markdown)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, out / "evaluation_summary.md")
    finally:
        if temp is not None and temp.exists():
            temp.unlink()
    return out
