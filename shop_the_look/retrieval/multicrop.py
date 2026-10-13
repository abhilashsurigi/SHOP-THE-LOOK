"""Controlled multi-crop FashionSigLIP retrieval experiment.

This module leaves the whole-scene baseline untouched. It uses the existing
exact IndexFlatIP product index and aggregates each product's best cosine
similarity over a deterministic, label-independent set of scene windows.
"""
from __future__ import annotations

import hashlib
import base64
import html
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from shop_the_look.embedding_pipeline import DEFAULT_CACHE, DEFAULT_CATALOG, NORM_ATOL, preprocessing_config, sha256_file
from shop_the_look.models.fashionsiglip import FashionSigLIP, MODEL_ID, MODEL_REVISION, validate_and_normalize
from shop_the_look.retrieval.evaluation import DEFAULT_VALIDATION, evaluate, read_catalog_ids, read_id_set, read_validation
from shop_the_look.retrieval.faiss_index import DEFAULT_OUTPUT, RetrievalEngine

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTS = ROOT / "artifacts" / "experiments"
DEFAULT_FAILURE_SCENES = (
    "0aba9fb610db58dc142745fe27a64642", "465cb4f0c40dfe986f97a24dcd695692",
    "7eb435fcdee652715770f3696604789c", "c2c43ce039457308c491b026223f1ae7",
    "f86017bc2e570936f6850e92267e15f3", "fb6d5c9b6deef57a13820306273f35f9",
)
DEFAULT_SCALES = (0.75, 0.5)
DEFAULT_OVERLAP = 0.25
EXPECTED_DIMENSION = 768


@dataclass(frozen=True)
class Crop:
    crop_id: str
    scene_id: str
    image_path: str
    scale: float
    x0: int
    y0: int
    x1: int
    y1: int
    width: int
    height: int

    def manifest_row(self) -> dict[str, Any]:
        return {"crop_id": self.crop_id, "scene_id": self.scene_id, "original_image_path": self.image_path,
                "original_dimensions": {"width": self.width, "height": self.height},
                "scale": self.scale, "coordinates_xyxy": [self.x0, self.y0, self.x1, self.y1]}


def _positions(length: int, window: int, overlap: float) -> list[int]:
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if window <= 0 or window > length:
        raise ValueError("crop window must be positive and no larger than the image")
    stride = max(1, int(round(window * (1 - overlap))))
    positions = list(range(0, length - window + 1, stride))
    end = length - window
    if not positions or positions[-1] != end:
        positions.append(end)
    return sorted(set(positions))


def generate_crops(scene_id: str, image_path: str | Path, width: int, height: int,
                   scales: Sequence[float] = DEFAULT_SCALES,
                   overlap: float = DEFAULT_OVERLAP) -> list[Crop]:
    """Full image, then row-major spatial windows for each scale in order."""
    if not scene_id or width <= 0 or height <= 0:
        raise ValueError("scene_id and positive source dimensions are required")
    if len(set(scales)) != len(scales) or any(not 0 < s < 1 for s in scales):
        raise ValueError("scales must be distinct values strictly between zero and one")
    path = str(Path(image_path).resolve())
    crops: list[Crop] = []
    seen: set[tuple[int, int, int, int]] = set()

    def add(scale: float, x0: int, y0: int, x1: int, y1: int) -> None:
        coords = (x0, y0, x1, y1)
        if coords in seen:
            return
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"invalid crop bounds {coords} for {width}x{height}")
        seen.add(coords)
        tag = "full" if scale == 1.0 else f"scale-{scale:.4f}"
        crop_id = f"{scene_id}:{tag}:{x0},{y0},{x1},{y1}"
        crops.append(Crop(crop_id, scene_id, path, scale, x0, y0, x1, y1, width, height))

    add(1.0, 0, 0, width, height)
    for scale in scales:
        crop_width = max(1, min(width, int(round(width * scale))))
        crop_height = max(1, min(height, int(round(height * scale))))
        for y0 in _positions(height, crop_height, overlap):
            for x0 in _positions(width, crop_width, overlap):
                add(float(scale), x0, y0, x0 + crop_width, y0 + crop_height)
    if len({crop.crop_id for crop in crops}) != len(crops):
        raise ValueError("crop identifier collision")
    return crops


def validate_crop_embeddings(embeddings: Any, expected_count: int,
                             dimension: int = EXPECTED_DIMENSION) -> np.ndarray:
    vectors = validate_and_normalize(embeddings, expected_count, dimension)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=NORM_ATOL, rtol=0):
        raise ValueError("crop embedding vectors are not L2-normalized")
    return vectors


def aggregate_max_scores(score_matrix: Any, product_ids: Sequence[str], crop_ids: Sequence[str]
                         ) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Max similarity per product; ties sort lexically by product ID.

    Returns ranked products, aggregate scores by input product row, and the
    best crop row per product (first crop wins exact within-product ties).
    """
    scores = np.asarray(score_matrix, dtype=np.float32)
    if scores.ndim != 2 or scores.shape != (len(crop_ids), len(product_ids)):
        raise ValueError("score matrix shape must be (crop_count, product_count)")
    if not np.isfinite(scores).all() or not product_ids or not crop_ids:
        raise ValueError("scores, product IDs, and crop IDs must be nonempty and finite")
    if len(set(product_ids)) != len(product_ids) or len(set(crop_ids)) != len(crop_ids):
        raise ValueError("product IDs and crop IDs must be unique")
    best_crop_rows = np.argmax(scores, axis=0)
    aggregate = scores[best_crop_rows, np.arange(len(product_ids))]
    order = sorted(range(len(product_ids)), key=lambda idx: (-float(aggregate[idx]), product_ids[idx]))
    ranked = [{"product_id": product_ids[row], "rank": rank,
               "cosine_similarity": float(aggregate[row]),
               "best_crop_id": crop_ids[int(best_crop_rows[row])]}
              for rank, row in enumerate(order, 1)]
    return ranked, aggregate, best_crop_rows


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def experiment_fingerprint(payload: Any) -> str:
    """Stable fingerprint for the model, crop, index, label, and image inputs."""
    return _canonical_sha(payload)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            tmp = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            tmp = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def _atomic_npy(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            tmp = Path(stream.name)
            np.save(stream, matrix, allow_pickle=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def load_embedding_cache(embeddings_path: Path, cache_meta_path: Path, crops_path: Path,
                        fingerprint: str, crop_rows: list[dict[str, Any]],
                        dimension: int = EXPECTED_DIMENSION) -> np.ndarray | None:
    """Return a verified matching cache or None; never reuse mismatched artifacts."""
    if not (embeddings_path.is_file() and cache_meta_path.is_file() and crops_path.is_file()):
        return None
    try:
        metadata = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        stored_rows = [json.loads(line) for line in crops_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        matrix = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
        if (metadata.get("experiment_fingerprint") != fingerprint or
                metadata.get("crop_manifest_sha256") != _canonical_sha(crop_rows) or
                metadata.get("embeddings_sha256") != sha256_file(embeddings_path) or
                stored_rows != crop_rows or matrix.shape != (len(crop_rows), dimension) or
                matrix.dtype != np.float32 or not np.isfinite(matrix).all() or
                not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=NORM_ATOL, rtol=0)):
            return None
        return matrix
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _inline_asset(source: str | Path) -> str | None:
    path = Path(source).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        return None
    if not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            mime = Image.MIME.get(image.format)
        if not mime:
            return None
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except (OSError, ValueError):
        return None


def unchanged_file_hashes(expected: dict[str, str]) -> bool:
    """Verify an immutable baseline file inventory against its pre-run SHA-256 values."""
    return all(Path(path).is_file() and sha256_file(Path(path)) == digest for path, digest in expected.items())


def _load_scene_inputs(cache_dir: Path, scene_ids: Sequence[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records") if isinstance(manifest, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"invalid image cache manifest: {manifest_path}")
    selected = {}
    for record in records:
        if isinstance(record, dict) and record.get("entity_type") == "scene" and record.get("entity_id") in scene_ids:
            selected[record["entity_id"]] = record
    if set(selected) != set(scene_ids):
        raise ValueError(f"scene cache manifest is missing {len(set(scene_ids) - set(selected))} validation scenes")
    inputs = {}
    for scene_id in sorted(scene_ids):
        record = selected[scene_id]
        if record.get("download_status") not in {"success", "reused_cache"} or record.get("validation_outcome") != "valid_image":
            raise ValueError(f"scene {scene_id} is not marked as a valid cached image")
        raw_path = Path(record["cache_path"])
        path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"cached validation scene is missing or empty: {path}")
        digest = sha256_file(path)
        if digest != record.get("sha256"):
            raise ValueError(f"cached validation scene digest disagrees with manifest: {path}")
        with Image.open(path) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise ValueError(f"invalid image dimensions for {path}")
            inputs[scene_id] = {"path": path, "sha256": digest, "width": image.width, "height": image.height,
                                "format": image.format, "source_url": record.get("source_url"),
                                "url_strategy": record.get("url_strategy"), "url_provenance": record.get("url_provenance")}
    return inputs, {"manifest_sha256": sha256_file(manifest_path), "scene_inputs": [
        {"scene_id": sid, "sha256": inputs[sid]["sha256"], "source_url": inputs[sid]["source_url"]}
        for sid in sorted(inputs)]}


def _build_crop_embeddings(crops_by_scene: dict[str, list[Crop]], scene_inputs: dict[str, dict[str, Any]],
                           encoder: Any, batch_size: int, output: Path) -> tuple[np.ndarray, list[Crop], float]:
    crops = [crop for scene_id in sorted(crops_by_scene) for crop in crops_by_scene[scene_id]]
    matrix = np.empty((len(crops), EXPECTED_DIMENSION), dtype=np.float32)
    position = {crop.crop_id: row for row, crop in enumerate(crops)}
    pending_images: list[Image.Image] = []
    pending_crops: list[Crop] = []
    inference_seconds = 0.0

    def flush() -> None:
        nonlocal inference_seconds
        if not pending_images:
            return
        started = time.perf_counter()
        vectors = encoder.embed_images(pending_images, batch_size=min(batch_size, len(pending_images)))
        inference_seconds += time.perf_counter() - started
        vectors = validate_crop_embeddings(vectors, len(pending_images))
        for crop, vector in zip(pending_crops, vectors):
            matrix[position[crop.crop_id]] = vector
        pending_images.clear(); pending_crops.clear()

    for scene_id in sorted(crops_by_scene):
        src = scene_inputs[scene_id]
        with Image.open(src["path"]) as image:
            image.load()
            rgb = image.convert("RGB")
            for crop in crops_by_scene[scene_id]:
                pending_images.append(rgb.crop((crop.x0, crop.y0, crop.x1, crop.y1)))
                pending_crops.append(crop)
                if len(pending_images) >= batch_size:
                    flush()
            rgb.close()
    flush()
    if matrix.shape != (len(crops), EXPECTED_DIMENSION) or not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=NORM_ATOL, rtol=0):
        raise ValueError("generated crop embedding cache failed validation")
    _atomic_npy(output, matrix)
    return matrix, crops, inference_seconds


def _rank_full_scores(score_matrix: np.ndarray, product_ids: Sequence[str], crop_ids: Sequence[str]
                      ) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    return aggregate_max_scores(score_matrix, product_ids, crop_ids)


def _html_report(path: Path, scenes: list[dict[str, Any]], product_paths: dict[str, str | None]) -> None:
    chunks = ["<!doctype html><html><head><meta charset='utf-8'><title>Multi-crop failure scenes</title>",
              "<style>body{font:15px system-ui;margin:2rem;max-width:1300px}section{border-top:2px solid #777;padding:1rem 0}img{max-width:120px;max-height:150px;object-fit:contain}figure{display:inline-flex;flex-direction:column;width:135px;vertical-align:top;margin:.4rem;padding:.3rem;border:1px solid #ddd}figure.scene{width:220px}figure.scene img{max-width:210px;max-height:380px}small{overflow-wrap:anywhere}.missing{color:#a00}</style></head><body>",
              "<h1>Multi-crop retrieval: original failure scenes</h1><p>For manual review. Retrieval relevance is label-based; no visual correctness judgments are assigned.</p>"]
    for row in scenes:
        chunks.append(f"<section><h2>{html.escape(row['scene_id'])}</h2><p>Eligible explicit targets: {len(row['eligible_ground_truth_product_ids'])}; missing from catalog/index: {len(row['missing_catalog_or_index_product_ids'])}</p>")
        scene_src = _inline_asset(row["scene_image_path"])
        if scene_src:
            chunks.append(f"<figure class='scene'><img src='{scene_src}' alt='scene'><figcaption>Scene image<br><small>{html.escape(row['scene_image_path'])}</small></figcaption></figure>")
        else:
            chunks.append("<figure class='scene'><div class='missing'>Scene image unavailable</div><figcaption>Scene image</figcaption></figure>")
        chunks.append("<h3>Explicit ground truth</h3>")
        for target in row["ground_truth"]:
            source = product_paths.get(target["product_id"])
            rel = _inline_asset(source) if source else None
            label = f"{target['product_id']} — before rank {target['baseline_rank']}, after rank {target['multicrop_rank']}"
            if rel:
                chunks.append(f"<figure><img src='{rel}' alt='ground truth product'><figcaption>Ground truth<br>{html.escape(label)}</figcaption></figure>")
            else:
                chunks.append(f"<figure><div class='missing'>Image unavailable</div><figcaption>Ground truth<br>{html.escape(label)}</figcaption></figure>")
        chunks.append("<h3>Multi-crop top 10</h3>")
        for item in row["top_10"]:
            source = product_paths.get(item["product_id"])
            rel = _inline_asset(source) if source else None
            caption = f"#{item['rank']} {html.escape(item['product_id'])}<br>cosine {item['cosine_similarity']:.5f}<br>crop {html.escape(str(item['best_crop_coordinates_xyxy']))}"
            if rel:
                chunks.append(f"<figure><img src='{rel}' alt='retrieved product'><figcaption>{caption}</figcaption></figure>")
            else:
                chunks.append(f"<figure><div class='missing'>Image unavailable</div><figcaption>{caption}</figcaption></figure>")
        chunks.append("</section>")
    chunks.append("</body></html>")
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("\n".join(chunks), encoding="utf-8")
    os.replace(temp, path)


def run_multicrop_experiment(*, output_root: Path = DEFAULT_EXPERIMENTS,
                             cache_dir: Path = DEFAULT_CACHE,
                             catalog_path: Path = DEFAULT_CATALOG,
                             validation_path: Path = DEFAULT_VALIDATION,
                             index_root: Path = DEFAULT_OUTPUT,
                             scales: Sequence[float] = DEFAULT_SCALES,
                             overlap: float = DEFAULT_OVERLAP,
                             batch_size: int = 32,
                             top_k: int = 10,
                             device: str = "cpu",
                             reuse_cache: bool = True) -> Path:
    if batch_size <= 0 or top_k <= 0:
        raise ValueError("batch_size and top_k must be positive")
    pairs, validation_info = read_validation(validation_path)
    expected_scenes = sorted({scene for scene, _ in pairs})
    scene_inputs, manifest_fingerprint = _load_scene_inputs(cache_dir, expected_scenes)
    crops_by_scene = {
        sid: generate_crops(sid, scene_inputs[sid]["path"], scene_inputs[sid]["width"], scene_inputs[sid]["height"], scales, overlap)
        for sid in expected_scenes
    }
    crops = [crop for sid in expected_scenes for crop in crops_by_scene[sid]]

    # Keep the model choice and preprocessing identical to the existing pipeline.
    model_load_started = time.perf_counter()
    encoder = FashionSigLIP(device=device, revision=MODEL_REVISION)
    model_load_seconds = time.perf_counter() - model_load_started
    preprocess = preprocessing_config(encoder)
    active = json.loads((index_root / "active_index.json").read_text(encoding="utf-8"))
    metadata_path = Path(active["index_metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("index_fingerprint") != active.get("index_fingerprint") or metadata.get("index_type") != "IndexFlatIP":
        raise ValueError("active product index pointer is inconsistent or not IndexFlatIP")
    catalog_ids = read_catalog_ids(catalog_path)
    mapping_path = Path(metadata["product_id_mapping_path"])
    if sha256_file(mapping_path) != metadata.get("product_id_mapping_sha256"):
        raise ValueError("saved product mapping checksum does not match metadata")
    if sha256_file(Path(metadata["index_path"])) != metadata.get("index_sha256"):
        raise ValueError("saved product index checksum does not match metadata")
    index_ids = read_id_set(mapping_path, "product_id")
    if len(index_ids) != int(metadata.get("ntotal", -1)):
        raise ValueError("product index mapping count disagrees with index metadata")
    if metadata.get("model_revision") != MODEL_REVISION or metadata.get("backend") != "faiss":
        raise ValueError("multi-crop run requires the existing pinned FashionSigLIP FAISS product index")
    if catalog_ids != index_ids:
        raise ValueError("canonical catalog and saved index product IDs disagree")
    model_matrix_fingerprint = metadata.get("product_matrix_sha256")
    config = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "device": device,
              "preprocessing": preprocess, "crop_scales": list(scales), "overlap": overlap,
              "crop_order": "scene_id_ascending; full-scene first; then scale order; row-major y,x",
              "crop_window_rounding": "round(original_dimension * scale), clamped to [1, dimension]",
              "aggregation": "maximum cosine similarity over all scene crops; score-descending then product_id-ascending",
              "embedding_dimension": EXPECTED_DIMENSION, "normalization_tolerance_atol": NORM_ATOL,
              "batch_size": batch_size, "top_k": top_k}
    fingerprint_payload = {"config": config,
        "scene_cache_fingerprint": manifest_fingerprint,
        "validation_sha256": sha256_file(validation_path), "catalog_sha256": sha256_file(catalog_path),
        "index_fingerprint": metadata["index_fingerprint"], "index_sha256": metadata["index_sha256"],
        "product_matrix_sha256": model_matrix_fingerprint,
        "scene_files": [{"scene_id": sid, "sha256": scene_inputs[sid]["sha256"]} for sid in expected_scenes]}
    fingerprint = experiment_fingerprint(fingerprint_payload)
    run_dir = output_root / f"multicrop_{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = run_dir / "crop_embeddings.npy"
    cache_meta_path = run_dir / "crop_embedding_cache.json"
    crops_path = run_dir / "crop_manifest.jsonl"
    crop_rows = [crop.manifest_row() for crop in crops]
    input_crop_manifest_sha = _canonical_sha(crop_rows)
    available_matching_cache = load_embedding_cache(embeddings_path, cache_meta_path, crops_path, fingerprint, crop_rows)
    crop_embeddings = available_matching_cache if reuse_cache else None
    reused = crop_embeddings is not None
    inference_seconds = 0.0
    if crop_embeddings is None:
        crop_embeddings, crops, inference_seconds = _build_crop_embeddings(crops_by_scene, scene_inputs, encoder, batch_size, embeddings_path)
        _atomic_jsonl(crops_path, crop_rows)
        _atomic_json(cache_meta_path, {"schema_version": 1, "experiment_fingerprint": fingerprint,
            "crop_manifest_sha256": input_crop_manifest_sha, "embeddings_sha256": sha256_file(embeddings_path),
            "embedding_shape": list(crop_embeddings.shape), "embedding_dimension": EXPECTED_DIMENSION,
            "dtype": "float32", "all_finite": True, "all_l2_normalized": True,
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "preprocessing": preprocess,
            "crop_config": {"scales": list(scales), "overlap": overlap},
            "inference_seconds": inference_seconds,
            "embedding_process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.uname().sysname == "Darwin" else 1024))})
    cache_metadata = json.loads(cache_meta_path.read_text(encoding="utf-8"))

    context = {"config": config, "fingerprint_payload": fingerprint_payload, "fingerprint": fingerprint,
        "cache_dir": str(cache_dir), "catalog_path": str(catalog_path), "validation_path": str(validation_path),
        "index_root": str(index_root), "scales": list(scales), "overlap": overlap,
        "model_load_seconds": model_load_seconds, "inference_seconds": inference_seconds,
        "cached_generation_inference_seconds": cache_metadata.get("inference_seconds", inference_seconds),
        "cached_generation_peak_rss_bytes": cache_metadata.get("embedding_process_peak_rss_bytes"),
        "reused": reused,
        "matching_cache_available_before_run": available_matching_cache is not None,
        "embedding_stage_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.uname().sysname == "Darwin" else 1024)),
        "validation_info": validation_info}
    _atomic_json(run_dir / "run_context.json", context)
    script_path = ROOT / "scripts" / "run_multicrop_experiment.py"
    completed = subprocess.run([sys.executable, str(script_path), "--complete-run", str(run_dir)],
                               cwd=ROOT, check=False)
    if completed.returncode:
        raise RuntimeError(f"isolated FAISS retrieval stage failed with exit code {completed.returncode}; crop embeddings remain cached at {embeddings_path}")
    return run_dir


def complete_multicrop_experiment(run_dir: Path) -> Path:
    """FAISS-only process stage, isolated from PyTorch's macOS OpenMP runtime."""
    run_dir = run_dir.resolve()
    context = json.loads((run_dir / "run_context.json").read_text(encoding="utf-8"))
    config = context["config"]
    fingerprint = context["fingerprint"]
    fingerprint_payload = context["fingerprint_payload"]
    cache_dir = Path(context["cache_dir"])
    catalog_path = Path(context["catalog_path"])
    validation_path = Path(context["validation_path"])
    index_root = Path(context["index_root"])
    pairs, validation_info = read_validation(validation_path)
    expected_scenes = sorted({scene for scene, _ in pairs})
    scene_inputs, _ = _load_scene_inputs(cache_dir, expected_scenes)
    crops_by_scene = {sid: generate_crops(sid, scene_inputs[sid]["path"], scene_inputs[sid]["width"],
                    scene_inputs[sid]["height"], config["crop_scales"], config["overlap"]) for sid in expected_scenes}
    crops = [crop for sid in expected_scenes for crop in crops_by_scene[sid]]
    embeddings_path = run_dir / "crop_embeddings.npy"
    crop_embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    cache_meta = json.loads((run_dir / "crop_embedding_cache.json").read_text(encoding="utf-8"))
    crop_rows = [crop.manifest_row() for crop in crops]
    if (crop_embeddings.shape != (len(crops), EXPECTED_DIMENSION) or crop_embeddings.dtype != np.float32 or
            not np.isfinite(crop_embeddings).all() or
            not np.allclose(np.linalg.norm(crop_embeddings, axis=1), 1.0, atol=NORM_ATOL, rtol=0) or
            cache_meta.get("experiment_fingerprint") != fingerprint or
            cache_meta.get("embeddings_sha256") != sha256_file(embeddings_path) or
            cache_meta.get("crop_manifest_sha256") != _canonical_sha(crop_rows)):
        raise ValueError("cached crop embedding dimensions or dtype disagree with crop manifest")
    stored_crop_rows = _read_jsonl(run_dir / "crop_manifest.jsonl")
    if stored_crop_rows != crop_rows:
        raise ValueError("saved crop manifest does not match regenerated deterministic crop coordinates")
    active = json.loads((index_root / "active_index.json").read_text(encoding="utf-8"))
    metadata_path = Path(active["index_metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    engine = RetrievalEngine.load(index_root, catalog_path=catalog_path, metadata_path=metadata_path)
    if engine.metadata["index_fingerprint"] != fingerprint_payload["index_fingerprint"]:
        raise ValueError("active product index changed after crop embeddings were generated")
    catalog_ids = read_catalog_ids(catalog_path)
    index_ids = set(engine.index.product_ids)
    top_k = min(int(config["top_k"]), engine.index.ntotal)

    baseline_eval_dir = _find_baseline_evaluation(ROOT / "artifacts" / "evaluation", metadata["index_fingerprint"])
    baseline_rows = _read_jsonl(baseline_eval_dir / "per_scene_metrics.jsonl")
    saved_baseline_report = json.loads((baseline_eval_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    baseline_top_path = Path(metadata["index_path"]).parent / "all_scenes_top_10.jsonl"
    baseline_top_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(baseline_top_path):
        baseline_top_by_scene.setdefault(row["scene_id"], []).append(row)
    baseline_files = [baseline_top_path, metadata_path, Path(metadata["index_path"]), Path(metadata["product_id_mapping_path"]),
        baseline_eval_dir / "evaluation_report.json", baseline_eval_dir / "per_scene_metrics.jsonl",
        baseline_eval_dir / "retrieval_diagnostics.json", Path(metadata["product_matrix_path"]),
        Path(metadata["scene_embedding_summary_path"]), Path(metadata["product_embedding_summary_path"])]
    baseline_hashes = {str(path): sha256_file(path) for path in baseline_files}

    multi_retrieval: dict[str, list[dict[str, Any]]] = {}
    multi_top_rows: list[dict[str, Any]] = []
    baseline_rank_by_scene: dict[str, dict[str, int]] = {}
    multi_rank_by_scene: dict[str, dict[str, int]] = {}
    target_set_by_scene: dict[str, set[str]] = {}
    for scene, product in pairs:
        target_set_by_scene.setdefault(scene, set()).add(product)
    failure_score_details: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    query_seconds = 0.0
    scene_to_crop_rows: dict[str, list[int]] = {}
    for crop_row, crop in enumerate(crops):
        scene_to_crop_rows.setdefault(crop.scene_id, []).append(crop_row)
    for scene_id in expected_scenes:
        scene_row = engine.scene_rows[scene_id]
        query = np.ascontiguousarray(engine.scene_vectors[scene_row:scene_row + 1], dtype=np.float32)
        started = time.perf_counter()
        b_scores, b_ids = engine.index._index.search(query, engine.index.ntotal)
        query_seconds += time.perf_counter() - started
        order = sorted(range(engine.index.ntotal), key=lambda i: (-float(b_scores[0][i]), engine.index.product_ids[int(b_ids[0][i])]))
        baseline_rank_by_scene[scene_id] = {engine.index.product_ids[int(b_ids[0][i])]: rank for rank, i in enumerate(order, 1)}
        rows = scene_to_crop_rows[scene_id]
        vectors = np.ascontiguousarray(crop_embeddings[rows], dtype=np.float32)
        started = time.perf_counter()
        crop_scores, crop_indices = engine.index._index.search(vectors, engine.index.ntotal)
        query_seconds += time.perf_counter() - started
        scores = np.empty((len(rows), engine.index.ntotal), dtype=np.float32)
        for crop_num in range(len(rows)):
            scores[crop_num, crop_indices[crop_num]] = crop_scores[crop_num]
        crop_ids = [crops[row].crop_id for row in rows]
        ranked, aggregate, best_rows = aggregate_max_scores(scores, engine.index.product_ids, crop_ids)
        multi_rank_by_scene[scene_id] = {entry["product_id"]: entry["rank"] for entry in ranked}
        multi_retrieval[scene_id] = [{key: entry[key] for key in ("product_id", "rank", "cosine_similarity")} for entry in ranked[:top_k]]
        crop_by_id = {crop.crop_id: crop for crop in (crops[row] for row in rows)}
        for entry in ranked[:top_k]:
            crop = crop_by_id[entry["best_crop_id"]]
            multi_top_rows.append({"scene_id": scene_id, "product_id": entry["product_id"], "rank": entry["rank"],
                "cosine_similarity": entry["cosine_similarity"], "best_crop_id": crop.crop_id,
                "best_crop_coordinates_xyxy": [crop.x0, crop.y0, crop.x1, crop.y1], "best_crop_scale": crop.scale})
        if scene_id in DEFAULT_FAILURE_SCENES:
            failure_score_details[scene_id] = (scores, aggregate, best_rows)

    multi_summary, multi_per_scene, _ = evaluate(pairs, catalog_ids, index_ids, multi_retrieval, (1, 5, 10))
    baseline_retrieval = {sid: [{"product_id": row["product_id"], "rank": row["rank"], "cosine_similarity": row["cosine_similarity"]}
                                for row in baseline_top_by_scene.get(sid, [])] for sid in expected_scenes}
    baseline_summary, baseline_per_scene, _ = evaluate(pairs, catalog_ids, index_ids, baseline_retrieval, (1, 5, 10))
    if len(baseline_per_scene) != len(baseline_rows):
        raise ValueError("saved baseline evaluation does not cover the same validation scenes")
    if baseline_summary["metrics"] != saved_baseline_report["metrics"]:
        raise ValueError("saved baseline metrics do not match the same evaluator applied to its saved retrievals")
    baseline_scene_metrics = {row["scene_id"]: row for row in baseline_per_scene}
    multi_scene_metrics = {row["scene_id"]: row for row in multi_per_scene}
    scene_deltas = []
    for sid in expected_scenes:
        before = baseline_scene_metrics[sid]["metrics"]["recall_at_10"]
        after = multi_scene_metrics[sid]["metrics"]["recall_at_10"]
        if before is None or after is None:
            delta, change = None, "not_evaluable"
        else:
            delta = after - before
            change = "improved" if after > before else "regressed" if after < before else "unchanged"
        scene_deltas.append({"scene_id": sid, "baseline_recall_at_10": before, "multicrop_recall_at_10": after,
            "delta": delta, "change": change})

    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    product_path_by_id: dict[str, str | None] = {}
    for record in manifest.get("records", []):
        if isinstance(record, dict) and record.get("entity_type") == "product" and record.get("entity_id") in index_ids:
            raw = Path(record["cache_path"])
            resolved = raw if raw.is_absolute() else ROOT / raw
            product_path_by_id[record["entity_id"]] = str(resolved) if resolved.is_file() else None
    failure_details = []
    for scene_id in DEFAULT_FAILURE_SCENES:
        if scene_id not in expected_scenes:
            continue
        eligible = target_set_by_scene.get(scene_id, set()) & catalog_ids & index_ids
        missing = target_set_by_scene.get(scene_id, set()) - catalog_ids.intersection(index_ids)
        score_matrix, aggregate, best_rows = failure_score_details[scene_id]
        local_rows = scene_to_crop_rows[scene_id]
        scene_detail = {"scene_id": scene_id, "scene_image_path": str(scene_inputs[scene_id]["path"]),
            "ground_truth_product_ids": sorted(target_set_by_scene.get(scene_id, set())),
            "eligible_ground_truth_product_ids": sorted(eligible), "missing_catalog_or_index_product_ids": sorted(missing),
            "ground_truth": [], "new_top10_contains_relevant": False, "top_10": []}
        for product_id in sorted(target_set_by_scene.get(scene_id, set())):
            base_rank = baseline_rank_by_scene[scene_id].get(product_id) if product_id in index_ids else None
            new_rank = multi_rank_by_scene[scene_id].get(product_id) if product_id in index_ids else None
            detail = {"product_id": product_id, "eligible": product_id in eligible,
                "missing_from_catalog": product_id not in catalog_ids, "missing_from_index": product_id not in index_ids,
                "baseline_rank": base_rank, "multicrop_rank": new_rank,
                "product_image_path": product_path_by_id.get(product_id)}
            if product_id in index_ids:
                product_row = engine.index.product_ids.index(product_id)
                crop_local_row = int(best_rows[product_row])
                crop = crops[local_rows[crop_local_row]]
                detail["best_crop"] = {"crop_id": crop.crop_id, "coordinates_xyxy": [crop.x0, crop.y0, crop.x1, crop.y1],
                    "scale": crop.scale, "cosine_similarity": float(aggregate[product_row])}
            scene_detail["ground_truth"].append(detail)
        top = [item for item in multi_top_rows if item["scene_id"] == scene_id]
        scene_detail["top_10"] = [{**item, "product_image_path": product_path_by_id.get(item["product_id"])} for item in top]
        scene_detail["new_top10_contains_relevant"] = any(item["product_id"] in eligible for item in scene_detail["top_10"])
        failure_details.append(scene_detail)

    _atomic_jsonl(run_dir / "multicrop_top10.jsonl", multi_top_rows)
    _atomic_jsonl(run_dir / "per_scene_metric_deltas.jsonl", scene_deltas)
    _atomic_json(run_dir / "six_failure_scenes.json", {"method": "exact full-index max similarity over crop embeddings", "scenes": failure_details})
    _html_report(run_dir / "six_failure_scenes.html", failure_details, product_path_by_id)
    changes = {name: sum(row["change"] == name for row in scene_deltas)
               for name in ("improved", "unchanged", "regressed", "not_evaluable")}
    if not unchanged_file_hashes(baseline_hashes):
        raise RuntimeError("an existing baseline artifact changed during the experiment")
    cached_generation_peak = context.get("cached_generation_peak_rss_bytes")
    stage_peak = int(cached_generation_peak) if cached_generation_peak is not None else None
    search_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.uname().sysname == "Darwin" else 1024))
    model_load_seconds = context.get("model_load_seconds")
    report = {"schema_version": 1, "experiment_fingerprint": fingerprint, "configuration": config,
        "input_fingerprints": fingerprint_payload,
        "counts": {"scene_count": len(expected_scenes), "crop_count": len(crops), "product_count": engine.index.ntotal,
            "validation_rows": validation_info["row_count"], "unique_association_count": validation_info["unique_pair_count"]},
        "coverage": {key: value for key, value in multi_summary.items() if key in {"validation", "coverage"}},
        "baseline_metrics": baseline_summary["metrics"], "multicrop_metrics": multi_summary["metrics"],
        "per_scene_recall_at_10_change_counts": changes,
        "per_scene_recall_at_10_mean_delta": float(np.mean([row["delta"] for row in scene_deltas if row["delta"] is not None])) if any(row["delta"] is not None for row in scene_deltas) else None,
        "timing": {"model_load_seconds": model_load_seconds,
            "crop_embedding_inference_seconds": context["cached_generation_inference_seconds"],
            "inference_seconds_this_execution": context["inference_seconds"], "index_query_seconds": query_seconds,
            "total_measured_seconds": (model_load_seconds + context["inference_seconds"] + query_seconds)
                if model_load_seconds is not None else None},
        "resources": {"embedding_process_peak_rss_bytes": stage_peak,
            "cache_validation_process_peak_rss_bytes": context["embedding_stage_peak_rss_bytes"] if context["reused"] else None,
            "retrieval_process_peak_rss_bytes": search_peak,
            "measurement": "separate process peak RSS; when cache is reused, embedding peak comes from the verified cache-generation run"},
        "cache": {"reused_crop_embeddings": context["reused"],
            "matching_cache_available_before_run": context["matching_cache_available_before_run"],
            "embedding_path": str(embeddings_path.resolve()),
            "crop_manifest_path": str((run_dir / "crop_manifest.jsonl").resolve()), "embedding_sha256": sha256_file(embeddings_path)},
        "baseline_artifacts_unchanged": True, "diagnostics": {"six_failure_scene_count": len(failure_details), "scenes": failure_details},
        "metric_semantics": "same evaluation.evaluate implementation and explicit validation associations as whole-scene baseline; unlabeled products are not treated as negatives"}
    _atomic_json(run_dir / "experiment_report.json", report)
    return run_dir


def _find_baseline_evaluation(root: Path, index_fingerprint: str) -> Path:
    candidates = []
    for report_path in root.glob("*/evaluation_report.json"):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("retrieval_index", {}).get("index_fingerprint") == index_fingerprint:
                candidates.append(report_path.parent)
        except (OSError, ValueError):
            continue
    if not candidates:
        raise ValueError(f"No saved evaluation report matches active index under {root}")
    return sorted(candidates)[-1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
