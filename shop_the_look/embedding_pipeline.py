"""Resumable, local-cache-only FashionSigLIP embedding generation.

Successful work is committed in batch NPZ shards. A shard is reusable only
when its atomic JSON sidecar, content hash, run fingerprint, image digest,
shape, finiteness, and unit norm all validate. Final NPY matrices are built
through a temporary memory map and atomically promoted after validation.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from shop_the_look.models.fashionsiglip import FashionSigLIP, MODEL_ID, MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "audit_output" / "canonical_catalog.jsonl"
DEFAULT_ASSOCIATIONS = ROOT / "audit_output" / "validation_associations.json"
DEFAULT_CACHE = ROOT / "image_cache"
DEFAULT_OUTPUT = ROOT / "artifacts" / "embeddings"
NORM_ATOL = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


@dataclass(frozen=True)
class EmbeddingItem:
    entity_type: str
    entity_id: str
    image_path: Path | None
    image_sha256: str | None
    source_url: str | None


def load_inventories(catalog_path: Path = DEFAULT_CATALOG,
                     associations_path: Path = DEFAULT_ASSOCIATIONS,
                     cache_dir: Path = DEFAULT_CACHE) -> tuple[dict[str, list[EmbeddingItem]], dict[str, Any]]:
    """Build sorted product and scene inventories solely from local artifacts."""
    issues: dict[str, Any] = {"malformed_catalog_rows": [], "malformed_association_rows": []}
    products: set[str] = set()
    with catalog_path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                issues["malformed_catalog_rows"].append({"line": line_no, "reason": "malformed_json"})
                continue
            value = row.get("product") if isinstance(row, dict) else None
            if not isinstance(value, str) or not value.strip():
                issues["malformed_catalog_rows"].append({"line": line_no, "reason": "blank_or_invalid_product_id"})
            else:
                products.add(value.strip())
    associations = json.loads(associations_path.read_text(encoding="utf-8"))
    if not isinstance(associations, list):
        raise ValueError("Validation associations must be a JSON array")
    scenes: set[str] = set()
    for row_no, row in enumerate(associations, 1):
        value = row.get("scene") if isinstance(row, dict) else None
        if not isinstance(value, str) or not value.strip():
            issues["malformed_association_rows"].append({"row": row_no, "reason": "blank_or_invalid_scene_id"})
        else:
            scenes.add(value.strip())
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records") if isinstance(manifest, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"Cache manifest does not contain a records list: {manifest_path}")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("entity_type"), str) and isinstance(record.get("entity_id"), str):
            indexed[(record["entity_type"], record["entity_id"])] = record
    inventories: dict[str, list[EmbeddingItem]] = {}
    for kind, values in (("product", products), ("scene", scenes)):
        items = []
        for entity_id in sorted(values):
            record = indexed.get((kind, entity_id), {})
            raw_path = record.get("cache_path")
            image_path = Path(raw_path) if isinstance(raw_path, str) else None
            if image_path is not None and not image_path.is_absolute():
                image_path = ROOT / image_path
            digest = record.get("sha256") if isinstance(record.get("sha256"), str) else None
            items.append(EmbeddingItem(kind, entity_id, image_path, digest, record.get("source_url")))
        inventories[kind] = items
    issues["catalog_fingerprint"] = sha256_file(catalog_path)
    issues["association_fingerprint"] = sha256_file(associations_path)
    # Stable and relevant manifest fingerprint: only resolved identity, file, URL and content hash.
    relevant = [{"entity_type": kind, "entity_id": item.entity_id,
                 "cache_path": str(item.image_path) if item.image_path else None,
                 "source_url": item.source_url, "sha256": item.image_sha256}
                for kind in ("product", "scene") for item in inventories[kind]]
    issues["image_manifest_fingerprint"] = _canonical_hash(relevant)
    return inventories, issues


def preprocessing_config(encoder: Any) -> dict[str, Any]:
    """Fingerprint processor configuration actually loaded by the wrapper."""
    snapshot = Path(encoder.snapshot_path)
    files = {}
    for name in ("preprocessor_config.json", "processor_config.json", "config.json", "open_clip_config.json"):
        path = snapshot / name
        if path.is_file():
            files[name] = sha256_file(path)
    return {"input_color_mode": "RGB", "processor_files_sha256": files,
            "embedding_normalized": True, "normalization_tolerance_atol": NORM_ATOL,
            "dtype": "float32"}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _load_shards(run_dir: Path, fingerprint: str, dimension: int) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], int]:
    vectors: dict[str, np.ndarray] = {}
    item_meta: dict[str, dict[str, Any]] = {}
    valid_shards = 0
    batch_dir = run_dir / "batches"
    if not batch_dir.exists():
        return vectors, item_meta, valid_shards
    for sidecar_path in sorted(batch_dir.glob("batch-*.json")):
        data_path = sidecar_path.with_suffix(".npz")
        try:
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if metadata.get("run_fingerprint") != fingerprint or not data_path.is_file():
                continue
            if sha256_file(data_path) != metadata.get("shard_sha256"):
                continue
            with np.load(data_path, allow_pickle=False) as archive:
                ids = archive["entity_id"].astype(str).tolist()
                batch_vectors = archive["embeddings"]
            rows = metadata.get("items", [])
            if batch_vectors.dtype != np.float32 or batch_vectors.shape != (len(ids), dimension) or len(rows) != len(ids):
                continue
            if [r.get("entity_id") for r in rows] != ids:
                continue
            if not np.isfinite(batch_vectors).all() or not np.allclose(np.linalg.norm(batch_vectors, axis=1), 1.0, atol=NORM_ATOL):
                continue
            valid_shards += 1
            for index, row in enumerate(rows):
                entity_id = row["entity_id"]
                if entity_id not in vectors:
                    vectors[entity_id] = batch_vectors[index].copy()
                    item_meta[entity_id] = row
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return vectors, item_meta, valid_shards


def _atomic_shard(batch_dir: Path, sequence: int, fingerprint: str, ids: list[str],
                  vectors: np.ndarray, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    stem = f"batch-{sequence:06d}"
    npz_path, json_path = batch_dir / f"{stem}.npz", batch_dir / f"{stem}.json"
    with tempfile.NamedTemporaryFile("wb", dir=batch_dir, prefix=f".{stem}.", suffix=".npz.tmp", delete=False) as stream:
        temp_npz = Path(stream.name)
        np.savez_compressed(stream, entity_id=np.asarray(ids, dtype="U"), embeddings=np.asarray(vectors, dtype=np.float32))
        stream.flush()
        os.fsync(stream.fileno())
    digest = sha256_file(temp_npz)
    os.replace(temp_npz, npz_path)
    sidecar = {"schema_version": 1, "run_fingerprint": fingerprint, "batch_sequence": sequence,
               "row_count": len(ids), "embedding_dimension": int(vectors.shape[1]),
               "dtype": "float32", "shard_sha256": digest, "items": rows,
               "committed_at_utc": datetime.now(timezone.utc).isoformat()}
    _atomic_json(json_path, sidecar)
    return npz_path, json_path


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def _compact(run_dir: Path, items: list[EmbeddingItem], vectors: dict[str, np.ndarray],
             metadata_by_id: dict[str, dict[str, Any]], dimension: int) -> None:
    successful = [item for item in items if item.entity_id in vectors]
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / "embeddings.npy"
    temp = run_dir / ".embeddings.tmp.npy"
    if temp.exists():
        temp.unlink()
    if successful:
        matrix = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=(len(successful), dimension))
        try:
            for start in range(0, len(successful), 256):
                batch = successful[start:start + 256]
                matrix[start:start + len(batch)] = np.stack([vectors[item.entity_id] for item in batch])
            matrix.flush()
        finally:
            del matrix
    else:
        with temp.open("wb") as stream:
            np.save(stream, np.empty((0, dimension), dtype=np.float32), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
    os.replace(temp, final)
    _write_jsonl_atomic(run_dir / "ids.jsonl", (
        {"row": index, "entity_type": item.entity_type, "entity_id": item.entity_id,
         "image_sha256": metadata_by_id[item.entity_id]["image_sha256"]}
        for index, item in enumerate(successful)
    ))


def run_entity(items: list[EmbeddingItem], encoder: Any, output_root: Path, fingerprints: dict[str, Any],
               batch_size: int = 8, limit: int | None = None,
               model_load_seconds: float | None = None,
               progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Generate/resume one entity inventory using atomic batch checkpoint shards."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    full_inventory_ids = [item.entity_id for item in items]
    if limit is not None:
        items = items[:limit]
    if len({item.entity_id for item in items}) != len(items):
        raise ValueError("Inventory contains duplicate entity IDs")
    kind = items[0].entity_type if items else "entities"
    dimension = int(encoder.expected_embedding_dimension)
    preprocess = preprocessing_config(encoder)
    configuration = {"entity_type": kind, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                     "embedding_dimension": dimension, "preprocessing": preprocess,
                     "catalog_fingerprint": fingerprints["catalog_fingerprint"],
                     "association_fingerprint": fingerprints["association_fingerprint"],
                     "image_manifest_fingerprint": fingerprints["image_manifest_fingerprint"],
                     "inventory_ids_sha256": _canonical_hash(full_inventory_ids)}
    fingerprint = _canonical_hash(configuration)
    run_dir = output_root / ("products" if kind == "product" else "scenes") / fingerprint
    vectors, reused_meta, valid_shards = _load_shards(run_dir, fingerprint, dimension)
    # Current failures are retried on each run. Missing/bad cache input is recorded rather than dropped.
    failures: list[dict[str, Any]] = []
    pending: list[EmbeddingItem] = []
    for item in items:
        if item.entity_id in vectors:
            if reused_meta[item.entity_id].get("image_sha256") == item.image_sha256:
                continue
            # Do not let an old row survive if its source image has changed.
            vectors.pop(item.entity_id, None)
            reused_meta.pop(item.entity_id, None)
        if not item.image_path or not item.image_path.is_file():
            failures.append({"entity_type": kind, "entity_id": item.entity_id, "error_category": "missing_image_file",
                             "reason": "manifest path is absent or file does not exist", "image_path": str(item.image_path) if item.image_path else None,
                             "source_url": item.source_url})
            continue
        if not item.image_sha256:
            failures.append({"entity_type": kind, "entity_id": item.entity_id, "error_category": "missing_image_digest",
                             "reason": "cache manifest has no SHA-256", "image_path": str(item.image_path), "source_url": item.source_url})
            continue
        actual_digest = sha256_file(item.image_path)
        if actual_digest != item.image_sha256:
            failures.append({"entity_type": kind, "entity_id": item.entity_id, "error_category": "image_identity_mismatch",
                             "reason": "cached file SHA-256 differs from manifest", "expected_sha256": item.image_sha256,
                             "actual_sha256": actual_digest, "image_path": str(item.image_path), "source_url": item.source_url})
            continue
        try:
            with Image.open(item.image_path) as image:
                image.load()
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("nonpositive image dimensions")
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            failures.append({"entity_type": kind, "entity_id": item.entity_id, "error_category": "image_decode_failure",
                             "reason": str(exc)[:400], "image_path": str(item.image_path), "source_url": item.source_url})
            continue
        pending.append(item)

    generated = 0
    inference_seconds = 0.0
    start_all = time.perf_counter()
    next_sequence = max((int(p.stem.removeprefix("batch-")) for p in (run_dir / "batches").glob("batch-*.json")), default=-1) + 1 if (run_dir / "batches").exists() else 0
    first_sequence = next_sequence

    def infer_commit(part: list[EmbeddingItem]) -> None:
        nonlocal generated, inference_seconds, next_sequence
        t0 = time.perf_counter()
        try:
            part_vectors = np.asarray(encoder.embed_images([item.image_path for item in part], batch_size=len(part)), dtype=np.float32)
            if part_vectors.shape != (len(part), dimension) or not np.isfinite(part_vectors).all():
                raise ValueError(f"invalid batch embedding shape or nonfinite values: {part_vectors.shape}")
            norms = np.linalg.norm(part_vectors, axis=1)
            if np.any(norms <= 0) or not np.allclose(norms, 1.0, atol=NORM_ATOL):
                raise ValueError(f"batch vectors are not L2 normalized: norms={norms.tolist()}")
            rows = [{"entity_type": kind, "entity_id": item.entity_id, "image_sha256": item.image_sha256}
                    for item in part]
            _atomic_shard(run_dir / "batches", next_sequence, fingerprint,
                          [item.entity_id for item in part], part_vectors, rows)
            for index, item in enumerate(part):
                vectors[item.entity_id] = part_vectors[index].copy()
                reused_meta[item.entity_id] = rows[index]
            generated += len(part)
            inference_seconds += time.perf_counter() - t0
            next_sequence += 1
        except Exception as exc:
            if len(part) > 1:
                midpoint = len(part) // 2
                infer_commit(part[:midpoint])
                infer_commit(part[midpoint:])
            else:
                item = part[0]
                failures.append({"entity_type": kind, "entity_id": item.entity_id,
                                 "error_category": "inference_failure", "reason": str(exc)[:500],
                                 "image_path": str(item.image_path), "source_url": item.source_url})

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        # Failed batch contents are recursively subdivided; only irrecoverable
        # single-item failures are recorded and no incomplete batch is committed.
        infer_commit(batch)
        if progress:
            progress({"entity_type": kind, "attempted_pending": min(offset + len(batch), len(pending)),
                      "pending_total": len(pending), "successful_cached": len(vectors),
                      "newly_generated": generated, "failures": len(failures),
                      "elapsed_seconds": time.perf_counter() - start_all})

    # A no-op repeat reuses shards and does not report them as fresh inference.
    _compact(run_dir, items, vectors, reused_meta, dimension)
    _write_jsonl_atomic(run_dir / "failures.jsonl", sorted(failures, key=lambda row: row["entity_id"]))
    successful = len(vectors)
    failed_ids = {row["entity_id"] for row in failures}
    unresolved = [item.entity_id for item in items if item.entity_id not in vectors and item.entity_id not in failed_ids]
    # Failures from prior attempts are represented in the final inventory report too.
    final_matrix = np.load(run_dir / "embeddings.npy", mmap_mode="r" if successful else None)
    final_norms = np.linalg.norm(final_matrix, axis=1) if successful else np.asarray([], dtype=np.float32)
    summary = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "entity_type": kind, "requested": len(items), "successful_embeddings": successful,
        "reused_embeddings": max(0, successful - generated), "newly_generated_embeddings": generated,
        "failed_embeddings": len(failed_ids), "unresolved_embeddings": len(unresolved),
        "failure_categories": {}, "valid_batch_shards": valid_shards,
        "new_batches_committed": next_sequence - first_sequence,
        "embedding_dimension": dimension, "dtype": "float32", "normalization_tolerance_atol": NORM_ATOL,
        "all_finite": successful == 0 or bool(np.isfinite(final_matrix).all()),
        "all_l2_normalized": bool(np.allclose(final_norms, 1.0, atol=NORM_ATOL)),
        "inference_seconds": inference_seconds, "total_entity_seconds": time.perf_counter() - start_all,
        "model_load_seconds": model_load_seconds,
        "throughput_images_per_second": generated / inference_seconds if inference_seconds else 0.0,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "device": encoder.device, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "Pillow": _package_version("Pillow"), "torch": _package_version("torch"),
                     "transformers": _package_version("transformers"),
                     "open_clip_torch": _package_version("open_clip_torch"),
                     "huggingface_hub": _package_version("huggingface-hub")},
        "model_configuration_fingerprint": fingerprint, "preprocessing": preprocess,
        "artifact_sha256": {"embeddings.npy": sha256_file(run_dir / "embeddings.npy"),
                            "ids.jsonl": sha256_file(run_dir / "ids.jsonl"),
                            "failures.jsonl": sha256_file(run_dir / "failures.jsonl")},
        "source_fingerprints": fingerprints, "artifacts": {
            "embeddings": str(run_dir / "embeddings.npy"), "ids": str(run_dir / "ids.jsonl"),
            "failures": str(run_dir / "failures.jsonl"), "batches": str(run_dir / "batches"),
            "run_history": str(run_dir / "run_history.jsonl")},
        "validation": {"unique_successful_ids": successful == len(set(vectors)),
                       "inventory_accounted_for": successful + len(failed_ids) + len(unresolved) == len(items),
                       "unresolved_ids": unresolved},
    }
    for failure in failures:
        category = failure["error_category"]
        summary["failure_categories"][category] = summary["failure_categories"].get(category, 0) + 1
    history_path = run_dir / "run_history.jsonl"
    history = []
    if history_path.is_file():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    history.append({"created_at_utc": summary["created_at_utc"], "requested": len(items),
                    "newly_generated_embeddings": generated, "reused_embeddings": summary["reused_embeddings"],
                    "failed_embeddings": len(failed_ids), "inference_seconds": inference_seconds,
                    "total_entity_seconds": summary["total_entity_seconds"],
                    "model_load_seconds": model_load_seconds, "run_fingerprint": fingerprint})
    _write_jsonl_atomic(history_path, history)
    generation_events = [row for row in history if row.get("newly_generated_embeddings", 0) > 0]
    summary["generation_totals"] = {
        "generation_runs": len(generation_events),
        "newly_generated_embeddings": sum(int(row["newly_generated_embeddings"]) for row in generation_events),
        "inference_seconds": sum(float(row.get("inference_seconds", 0.0)) for row in generation_events),
        "measured_model_load_seconds": [row["model_load_seconds"] for row in generation_events
                                         if row.get("model_load_seconds") is not None],
    }
    # Summary is the final commit marker for the compacted matrix and row map.
    _atomic_json(run_dir / "summary.json", summary)
    md = [f"# {kind.title()} FashionSigLIP embeddings", "", f"- Requested: {len(items)}", f"- Successful: {successful}",
          f"- Reused: {summary['reused_embeddings']}", f"- Newly generated: {generated}",
          f"- Failed this run: {len(failed_ids)}", f"- Unresolved: {len(unresolved)}", f"- Dimension: {dimension}",
          f"- Device: {encoder.device}", f"- Model revision: `{MODEL_REVISION}`", f"- Run fingerprint: `{fingerprint}`",
          f"- Model load seconds: {model_load_seconds}", f"- Inference seconds: {inference_seconds:.3f}",
          f"- Recorded generation totals: {summary['generation_totals']['newly_generated_embeddings']} vectors in "
          f"{summary['generation_totals']['inference_seconds']:.3f} inference seconds",
          f"- Throughput: {summary['throughput_images_per_second']:.3f} images/s",
          f"- Peak process RSS bytes: {summary['peak_process_rss_bytes']}", "", "## Failure categories", ""]
    md.extend([f"- {name}: {count}" for name, count in sorted(summary["failure_categories"].items())] or ["- None"])
    md.extend(["", "## Artifacts", "", f"- Matrix: `{run_dir / 'embeddings.npy'}`", f"- IDs: `{run_dir / 'ids.jsonl'}`",
               f"- Failures: `{run_dir / 'failures.jsonl'}`", f"- Batch checkpoints: `{run_dir / 'batches'}`",
               f"- Run history: `{run_dir / 'run_history.jsonl'}`", ""])
    markdown_path = run_dir / "summary.md"
    temp_md = run_dir / ".summary.tmp.md"
    temp_md.write_text("\n".join(md), encoding="utf-8")
    os.replace(temp_md, markdown_path)
    return summary


def generate(catalog_path: Path = DEFAULT_CATALOG, associations_path: Path = DEFAULT_ASSOCIATIONS,
             cache_dir: Path = DEFAULT_CACHE, output_root: Path = DEFAULT_OUTPUT,
             entity_types: tuple[str, ...] = ("scene", "product"), batch_size: int = 8,
             limit_products: int | None = None, limit_scenes: int | None = None,
             encoder_factory: Callable[[], Any] | None = None,
             progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    inventories, fingerprints = load_inventories(catalog_path, associations_path, cache_dir)
    counts = {"scene": limit_scenes, "product": limit_products}
    # Loading happens only after inventory validation, once for a full scene+product run.
    model_load_start = time.perf_counter()
    encoder = (encoder_factory or (lambda: FashionSigLIP(device="cpu")))()
    model_load_seconds = time.perf_counter() - model_load_start
    output = {}
    for kind in ("scene", "product"):
        if kind in entity_types:
            output[kind] = run_entity(inventories[kind], encoder, output_root, fingerprints,
                                      batch_size=batch_size, limit=counts[kind],
                                      model_load_seconds=model_load_seconds, progress=progress)
    return output
