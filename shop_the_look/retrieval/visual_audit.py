"""Offline, provenance-preserving visual inspection report for retrieval samples."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import mimetypes
import os
import random
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .evaluation import DEFAULT_EVALUATION_ROOT, DEFAULT_VALIDATION, read_catalog_ids, read_validation
from .faiss_index import DEFAULT_CATALOG, DEFAULT_OUTPUT, RetrievalError, _atomic_json, _atomic_jsonl, _read_jsonl, sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = ROOT / "image_cache"
FAILURE_SCENES = (
    "0aba9fb610db58dc142745fe27a64642", "465cb4f0c40dfe986f97a24dcd695692",
    "7eb435fcdee652715770f3696604789c", "c2c43ce039457308c491b026223f1ae7",
    "f86017bc2e570936f6850e92267e15f3", "fb6d5c9b6deef57a13820306273f35f9",
)


def select_scenes(per_scene: list[dict[str, Any]], failure_scenes: tuple[str, ...] = FAILURE_SCENES,
                  positive_count: int = 3, partial_missing_count: int = 2,
                  all_missing_count: int = 2, ordinary_count: int = 3,
                  seed: str = "shop-the-look-step11-v1") -> list[dict[str, Any]]:
    """Choose documented strata in stable order, using a fixed hash sample for ordinary cases."""
    by_id = {row["scene_id"]: row for row in per_scene}
    if len(by_id) != len(per_scene):
        raise RetrievalError("Duplicate scene in per-scene evaluation input")
    selected: dict[str, dict[str, Any]] = {}

    def add(scene_id: str, reason: str) -> None:
        item = selected.setdefault(scene_id, {"scene_id": scene_id, "selection_reasons": []})
        if reason not in item["selection_reasons"]:
            item["selection_reasons"].append(reason)

    for scene_id in failure_scenes:
        row = by_id.get(scene_id)
        if row is None:
            raise RetrievalError(f"Required audited failure scene is absent: {scene_id}")
        if row["eligible_ground_truth_count"] != 2 or row["retrieved_relevant_count_at_k"].get("10") != 0:
            raise RetrievalError(f"Audited failure scene no longer has two eligible targets and no top-10 hit: {scene_id}")
        add(scene_id, "previously_audited_two_target_top10_failure")

    positive = sorted(
        (r for r in per_scene if r["eligible_ground_truth_count"] > 0 and r["retrieved_relevant_count_at_k"].get("10", 0) > 0),
        key=lambda r: (min(x["rank"] for x in r["retrieved_relevant"]), r["scene_id"]),
    )
    for row in positive[:positive_count]:
        add(row["scene_id"], "eligible_ground_truth_retrieved_in_top10")

    partial = sorted(
        (r for r in per_scene if r["eligible_ground_truth_count"] > 0 and r["missing_catalog_count"] > 0),
        key=lambda r: r["scene_id"],
    )
    for row in partial[:partial_missing_count]:
        add(row["scene_id"], "eligible_and_catalog_missing_ground_truth")

    all_missing = sorted(
        (r for r in per_scene if r["eligible_ground_truth_count"] == 0 and r["missing_catalog_count"] > 0),
        key=lambda r: r["scene_id"],
    )
    for row in all_missing[:all_missing_count]:
        add(row["scene_id"], "all_ground_truth_targets_catalog_missing")

    ordinary = [r for r in per_scene if r["eligible_ground_truth_count"] == 1 and r["missing_catalog_count"] == 0]
    ordinary.sort(key=lambda r: (hashlib.sha256(f"{seed}:{r['scene_id']}".encode()).hexdigest(), r["scene_id"]))
    added = 0
    for row in ordinary:
        if row["scene_id"] in selected:
            continue
        add(row["scene_id"], "deterministic_ordinary_case_sample")
        added += 1
        if added >= ordinary_count:
            break
    return [selected[key] for key in selected]


def _safe_cache_path(record: dict[str, Any], cache_root: Path, workspace_root: Path = ROOT) -> Path | None:
    raw = record.get("cache_path")
    if not isinstance(raw, str) or not raw:
        return None
    candidate = Path(raw)
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace_root / candidate).resolve()
    root = cache_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RetrievalError(f"Image cache path escapes approved cache directory: {raw}") from exc
    return resolved


def _image_asset(entity_type: str, entity_id: str, manifest_records: dict[tuple[str, str], dict[str, Any]],
                 cache_root: Path, workspace_root: Path = ROOT) -> dict[str, Any]:
    record = manifest_records.get((entity_type, entity_id))
    common = {"entity_type": entity_type, "entity_id": entity_id}
    if record is None:
        return {**common, "available": False, "image_state": "no_manifest_record", "cache_path": None,
                "source_url": None, "url_strategy": None, "url_provenance": None}
    path = _safe_cache_path(record, cache_root, workspace_root)
    item = {**common, "available": False, "image_state": "missing_file", "cache_path": str(path) if path else None,
            "source_url": record.get("source_url"), "url_strategy": record.get("url_strategy"),
            "url_provenance": record.get("url_provenance"), "download_status": record.get("download_status"),
            "http_status": record.get("http_status"), "validation_outcome": record.get("validation_outcome"),
            "manifest_sha256": record.get("sha256"), "manifest_image_format": record.get("image_format"),
            "manifest_width": record.get("width"), "manifest_height": record.get("height"),
            "cache_reused": record.get("cache_reused", record.get("download_status") == "reused_cache")}
    if path is None or not path.is_file():
        return item
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            fmt, size = image.format, image.size
        mime = Image.MIME.get(fmt or "") or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        raw = path.read_bytes()
        item.update({"available": True, "image_state": "valid_decoded_image", "actual_sha256": hashlib.sha256(raw).hexdigest(),
                     "actual_image_format": fmt, "actual_width": size[0], "actual_height": size[1],
                     "_data_uri": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"})
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        item.update({"image_state": "corrupt_or_undecodable", "image_error": str(exc)[:250]})
    return item


def _card(title: str, asset: dict[str, Any], sub: str = "") -> str:
    title_safe = html.escape(title)
    sub_safe = html.escape(sub)
    if asset.get("available"):
        image = f'<img loading="lazy" src="{asset["_data_uri"]}" alt="{title_safe}">'
    else:
        image = f'<div class="placeholder">Image unavailable<br><small>{html.escape(asset.get("image_state", "unknown"))}</small></div>'
    provenance = "<br>".join(html.escape(f"{k}: {asset.get(k)}") for k in ("url_provenance", "url_strategy", "download_status") if asset.get(k))
    url = asset.get("source_url")
    if url:
        provenance += f'<br>source URL: <code>{html.escape(url)}</code>'
    path = asset.get("cache_path")
    if path:
        provenance += f'<br>cache: <code>{html.escape(path)}</code>'
    state = html.escape(asset.get("image_state", "unknown"))
    return f'<article class="card">{image}<div class="caption"><strong>{title_safe}</strong><br>{sub_safe}<br><span class="state">{state}</span><details><summary>image provenance</summary><p>{provenance}</p></details></div></article>'


def render_html(scenes: list[dict[str, Any]], selection: list[dict[str, Any]], cache_root: Path,
                workspace_root: Path = ROOT) -> str:
    if len(scenes) != len(selection) or any(a.get("scene_id") != b.get("scene_id") for a, b in zip(scenes, selection)):
        raise RetrievalError("Scene payloads and selection records must have identical order and length")
    cards: list[str] = []
    for choice, scene in zip(selection, scenes):
        scene_id = scene["scene_id"]
        section = [f'<section class="scene" id="scene-{html.escape(scene_id)}">',
                   f'<h2>Scene <code>{html.escape(scene_id)}</code></h2>',
                   f'<p>Selection: {html.escape(", ".join(choice["selection_reasons"]))}</p>',
                   f'<p>Explicit targets: {scene["ground_truth_unique_count"]}; eligible: {scene["eligible_ground_truth_count"]}; catalog-missing: {scene["missing_catalog_count"]}; index-missing: {scene["missing_index_count"]}.</p>',
                   '<h3>Scene image</h3><div class="grid">',
                   _card(f"Scene {scene_id}", scene["scene_image"]), '</div>', '<h3>Explicit ground-truth products</h3><div class="grid">']
        for asset in scene["ground_truth_images"]:
            product = asset["entity_id"]
            rank = scene["target_ranks"].get(product)
            label = "eligible in catalog/index" if asset.get("catalog_present") and asset.get("index_present") else "target absent from catalog/index"
            section.append(_card(f"Ground truth · {product}", asset, f"{label}; top-10 rank: {rank if rank is not None else 'not retrieved'}"))
        section.extend(['</div>', '<h3>Top-10 retrievals</h3><div class="grid">'])
        for asset in scene["retrieved_images"]:
            section.append(_card(f"Rank {asset['rank']} · {asset['entity_id']}", asset, f"cosine similarity: {asset['cosine_similarity']:.6f}"))
        section.extend(['</div>', '</section>'])
        cards.append("\n".join(section))
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SHOP-THE-LOOK visual retrieval audit</title>
<style>
body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f5f6f8;color:#18202a}header{padding:24px 4vw;background:#17212b;color:#fff}main{max-width:1500px;margin:auto;padding:20px 3vw}.scene{background:white;border:1px solid #d8dee6;border-radius:10px;padding:20px;margin:24px 0}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}.card{border:1px solid #d8dee6;border-radius:8px;overflow:hidden;background:#fff}.card img{width:100%;height:210px;object-fit:contain;background:#eef1f4}.placeholder{height:210px;display:grid;place-content:center;text-align:center;color:#8a3030;background:#f8eaea}.caption{padding:10px;font-size:12px;overflow-wrap:anywhere}.caption strong{font-size:13px}.state{color:#456}.scene h3{margin-top:24px}.scene code,code{overflow-wrap:anywhere}.scene p{color:#465260}details{margin-top:8px}summary{cursor:pointer}.notice{background:#fff4d6;padding:12px;border-radius:6px}
</style></head><body><header><h1>SHOP-THE-LOOK visual retrieval audit</h1><p>Offline report built only from existing local cache files. Ground truth means explicit validation association, not exhaustive annotation. Unlabelled retrieved products are not presumed wrong.</p></header><main>
<p class="notice">Product image URLs are labeled as an empirical URL hypothesis in provenance; scene URL construction follows the assignment rule. Every displayed image was checked for local existence and decoded before embedding in this report.</p>
""" + "\n".join(cards) + "\n</main></body></html>"


def _find_evaluation_dir(evaluation_root: Path, validation_path: Path, index_fingerprint: str) -> Path:
    candidates = []
    validation_digest = sha256_file(validation_path)
    for path in evaluation_root.glob("*/evaluation_report.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (value.get("retrieval_index", {}).get("index_fingerprint") == index_fingerprint and
                    value.get("input_fingerprints", {}).get("validation_sha256") == validation_digest):
                candidates.append(path.parent)
        except (OSError, json.JSONDecodeError):
            continue
    if not candidates:
        raise RetrievalError("No evaluation bundle matches the active index and validation file")
    return sorted(candidates, key=lambda p: p.name)[-1]


def build_visual_audit(validation_path: Path = DEFAULT_VALIDATION, catalog_path: Path = DEFAULT_CATALOG,
                       evaluation_dir: Path | None = None,
                       index_root: Path = DEFAULT_OUTPUT, cache_root: Path = DEFAULT_CACHE_ROOT,
                       output_root: Path = ROOT / "artifacts" / "visual_audit") -> Path:
    active = json.loads((index_root / "active_index.json").read_text(encoding="utf-8"))
    index_dir = Path(active["index_metadata_path"]).parent
    if evaluation_dir is None:
        evaluation_dir = _find_evaluation_dir(DEFAULT_EVALUATION_ROOT, validation_path, active["index_fingerprint"])
    report = json.loads((evaluation_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    if report.get("retrieval_index", {}).get("index_fingerprint") != active.get("index_fingerprint"):
        raise RetrievalError("Evaluation bundle does not match the active retrieval index")
    per_scene = _read_jsonl(evaluation_dir / "per_scene_metrics.jsonl")
    selection = select_scenes(per_scene)
    pairs, validation_info = read_validation(validation_path)
    gt_by_scene: dict[str, set[str]] = {}
    for scene_id, product_id in pairs:
        gt_by_scene.setdefault(scene_id, set()).add(product_id)
    if set(gt_by_scene) != {r["scene_id"] for r in per_scene}:
        raise RetrievalError("Validation scenes disagree with evaluation per-scene results")
    catalog_ids = read_catalog_ids(catalog_path)
    index_metadata_path = index_dir / "index_metadata.json"
    index_metadata = json.loads(index_metadata_path.read_text(encoding="utf-8"))
    index_ids = {row["product_id"].strip() for row in _read_jsonl(Path(index_metadata["product_id_mapping_path"]))}
    retrieval: dict[str, list[dict[str, Any]]] = {}
    retrieval_path = index_dir / "all_scenes_top_10.jsonl"
    for row in _read_jsonl(retrieval_path):
        retrieval.setdefault(row["scene_id"], []).append(row)
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ingestion_report_path = cache_root / "ingestion_report.json"
    ingestion_report = json.loads(ingestion_report_path.read_text(encoding="utf-8"))
    manifest_keys = {(r.get("entity_type"), r.get("entity_id")) for r in manifest.get("records", [])}
    report_keys = {(r.get("entity_type"), r.get("entity_id")) for r in ingestion_report.get("records", [])}
    if manifest_keys != report_keys:
        raise RetrievalError("Image manifest and ingestion report entity inventories disagree")
    manifest_records: dict[tuple[str, str], dict[str, Any]] = {}
    for record in manifest.get("records", []):
        manifest_records[(record.get("entity_type"), record.get("entity_id"))] = record

    scene_rows = {row["scene_id"]: row for row in per_scene}
    payload_scenes = []
    worksheet = []
    for choice in selection:
        scene_id = choice["scene_id"]
        eval_row = scene_rows[scene_id]
        gt_ids = sorted(gt_by_scene[scene_id])
        top_rows = sorted(retrieval.get(scene_id, []), key=lambda row: row["rank"])
        target_ranks = {row["product_id"]: row["rank"] for row in top_rows if row["product_id"] in gt_ids}
        scene_asset = _image_asset("scene", scene_id, manifest_records, cache_root)
        ground_assets = []
        retrieved_assets = []
        for product_id in gt_ids:
            asset = _image_asset("product", product_id, manifest_records, cache_root)
            asset["catalog_present"] = product_id in catalog_ids
            asset["index_present"] = product_id in index_ids
            ground_assets.append(asset)
            worksheet.append({"scene_id": scene_id, "role": "ground_truth", "product_id": product_id,
                              "rank": target_ranks.get(product_id), "judgment": "", "reviewer_notes": "",
                              "review_status": "awaiting_visual_review",
                              "judgment_options": ["apparent correct visual match", "plausible match but uncertain",
                                                   "apparent mismatch", "image unavailable or corrupted",
                                                   "suspected ID/label inconsistency", "insufficient evidence"],
                              "image_state": asset["image_state"], "selection_reasons": choice["selection_reasons"]})
        for row in top_rows:
            asset = _image_asset("product", row["product_id"], manifest_records, cache_root)
            asset["rank"] = row["rank"]
            asset["cosine_similarity"] = float(row["cosine_similarity"])
            asset["is_explicit_ground_truth"] = row["product_id"] in gt_ids
            retrieved_assets.append(asset)
            worksheet.append({"scene_id": scene_id, "role": "retrieved", "product_id": row["product_id"],
                              "rank": row["rank"], "cosine_similarity": float(row["cosine_similarity"]),
                              "is_explicit_ground_truth": asset["is_explicit_ground_truth"], "judgment": "",
                              "review_status": "awaiting_visual_review",
                              "judgment_options": ["apparent correct visual match", "plausible match but uncertain",
                                                   "apparent mismatch", "image unavailable or corrupted",
                                                   "suspected ID/label inconsistency", "insufficient evidence"],
                              "reviewer_notes": "", "image_state": asset["image_state"],
                              "selection_reasons": choice["selection_reasons"]})
        payload_scenes.append({"scene_id": scene_id, "selection_reasons": choice["selection_reasons"],
                               "ground_truth_unique_count": len(gt_ids),
                               "eligible_ground_truth_count": sum(pid in catalog_ids and pid in index_ids for pid in gt_ids),
                               "missing_catalog_count": sum(pid not in catalog_ids for pid in gt_ids),
                               "missing_index_count": sum(pid not in index_ids for pid in gt_ids),
                               "ground_truth_product_ids": gt_ids, "target_ranks": target_ranks,
                               "retrieved_top10": [{"product_id": r["product_id"], "rank": r["rank"], "cosine_similarity": float(r["cosine_similarity"])} for r in top_rows],
                               "scene_image": scene_asset, "ground_truth_images": ground_assets,
                               "retrieved_images": retrieved_assets})
    clean_scenes = []
    for scene in payload_scenes:
        clean = dict(scene)
        for key in ("scene_image",):
            clean[key] = {k: v for k, v in clean[key].items() if not k.startswith("_")}
        for key in ("ground_truth_images", "retrieved_images"):
            clean[key] = [{k: v for k, v in asset.items() if not k.startswith("_")} for asset in clean[key]]
        clean_scenes.append(clean)

    inputs = {"validation_sha256": sha256_file(validation_path), "catalog_sha256": sha256_file(catalog_path),
              "evaluation_report_sha256": sha256_file(evaluation_dir / "evaluation_report.json"),
              "retrieval_results_sha256": sha256_file(retrieval_path), "index_metadata_sha256": sha256_file(index_metadata_path),
        "image_manifest_sha256": sha256_file(manifest_path),
        "ingestion_report_sha256": sha256_file(ingestion_report_path),
        "index_fingerprint": report["retrieval_index"]["index_fingerprint"]}
    config = {"failure_scenes": list(FAILURE_SCENES), "positive_count": 3, "partial_missing_count": 2,
              "all_missing_count": 2, "ordinary_count": 3, "ordinary_seed": "shop-the-look-step11-v1",
              "top_k": 10, "image_source": "existing_cache_only", "html_mode": "embedded_data_uri"}
    fingerprint = hashlib.sha256(json.dumps({"inputs": inputs, "config": config}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    out = output_root / fingerprint
    out.mkdir(parents=True, exist_ok=True)
    html_text = render_html(payload_scenes, selection, cache_root)
    _atomic_json(out / "inspection_manifest.json", {"schema_version": 1, "inspection_fingerprint": fingerprint,
                 "selection_config": config, "input_fingerprints": inputs, "validation_counts": validation_info,
                 "ingestion_summary": ingestion_report.get("summary", {}),
                 "image_provenance_rules": {"scene": "assignment_specified", "product": "inferred_empirical_hypothesis"},
                 "selected_scenes": clean_scenes})
    _atomic_jsonl(out / "inspection_worksheet.jsonl", worksheet)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=out, prefix=".report.", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            stream.write(html_text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, out / "visual_inspection.html")
    finally:
        if temp is not None and temp.exists():
            temp.unlink()
    return out
