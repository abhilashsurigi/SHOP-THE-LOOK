from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from shop_the_look.models.fashionsiglip import FashionSigLIP, MODEL_ID, MODEL_REVISION

DEFAULT_CACHE = ROOT / "image_cache"
DEFAULT_OUTPUT = ROOT / "artifacts" / "smoke_test"


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def choose_rows(manifest_path: Path) -> list[dict[str, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Manifest has no records list: {manifest_path}")
    chosen: list[dict[str, str]] = []
    for kind, count in (("product", 3), ("scene", 2)):
        rows = [r for r in records if r.get("entity_type") == kind and r.get("download_status") in {"success", "reused_cache"}]
        rows.sort(key=lambda r: r["entity_id"])
        if len(rows) < count:
            raise ValueError(f"Need {count} cached {kind} images; found {len(rows)}")
        for record in rows[:count]:
            path = Path(record["cache_path"])
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"Missing or empty cached image for {kind} {record['entity_id']}: {path}")
            chosen.append({"entity_type": kind, "entity_id": record["entity_id"], "path": str(path)})
    return chosen


def run(cache_dir: Path = DEFAULT_CACHE, output_root: Path = DEFAULT_OUTPUT, device: str = "cpu") -> Path:
    rows = choose_rows(cache_dir / "manifest.json")
    image_metadata = []
    for row in rows:
        with Image.open(row["path"]) as image:
            image.load()
            image_metadata.append({**row, "image_format": image.format, "width": image.width, "height": image.height})
    init_start = time.perf_counter()
    encoder = FashionSigLIP(device=device)
    model_load_seconds = time.perf_counter() - init_start
    inference_start = time.perf_counter()
    embeddings = encoder.embed_images([row["path"] for row in rows], batch_size=2)
    inference_seconds = time.perf_counter() - inference_start
    products = [i for i, row in enumerate(rows) if row["entity_type"] == "product"]
    scenes = [i for i, row in enumerate(rows) if row["entity_type"] == "scene"]
    similarity = embeddings[scenes] @ embeddings[products].T
    norms = np.linalg.norm(embeddings, axis=1)
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if embeddings.shape != (5, embeddings.shape[1]) or embeddings.shape[1] <= 0:
        raise RuntimeError(f"Expected five vectors with nonzero dimension; got {embeddings.shape}")
    if not np.isfinite(embeddings).all() or not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError(f"Embedding validation failed: finite={bool(np.isfinite(embeddings).all())}, norms={norms.tolist()}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output_dir / "embeddings.npz",
        embeddings=embeddings,
        entity_type=np.asarray([r["entity_type"] for r in rows]),
        entity_id=np.asarray([r["entity_id"] for r in rows]),
    )
    np.save(output_dir / "scene_product_cosine_similarity.npy", similarity)
    try:
        import torch
        torch_version = torch.__version__
        mps_available = bool(torch.backends.mps.is_available())
    except Exception:
        torch_version, mps_available = None, False
    metadata = {
        "scope": "five-image model smoke test only; not retrieval evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": encoder.device,
        "mps_available": mps_available,
        "embedding_count": int(embeddings.shape[0]),
        "embedding_dimension": int(embeddings.shape[1]),
        "expected_embedding_dimension": encoder.expected_embedding_dimension,
        "normalization_tolerance_atol": 1e-5,
        "model_snapshot_path": str(encoder.snapshot_path),
        "verified_model_files": encoder.model_files,
        "peak_process_rss_bytes": int(peak_rss_bytes),
        "dtype": str(embeddings.dtype),
        "all_finite": bool(np.isfinite(embeddings).all()),
        "l2_norms": norms.tolist(),
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch_version,
            "transformers": package_version("transformers"),
            "open_clip_torch": package_version("open_clip_torch"),
            "huggingface_hub": package_version("huggingface-hub"),
            "numpy": np.__version__,
            "Pillow": package_version("Pillow"),
            "safetensors": package_version("safetensors"),
        },
        "images": image_metadata,
        "similarity_matrix": {
            "shape": list(similarity.shape),
            "rows_scene_ids": [rows[i]["entity_id"] for i in scenes],
            "columns_product_ids": [rows[i]["entity_id"] for i in products],
            "values": similarity.tolist(),
        },
        "artifacts": ["embeddings.npz", "scene_product_cosine_similarity.npy"],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "dimension": embeddings.shape[1], "device": encoder.device,
                      "model_load_seconds": model_load_seconds, "inference_seconds": inference_seconds,
                      "peak_process_rss_bytes": int(peak_rss_bytes),
                      "similarity_matrix": similarity.tolist(), "l2_norms": norms.tolist()}, indent=2))
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate five Marqo-FashionSigLIP embeddings from local cache only.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    # Isolate Hugging Face downloads from the global cache and dataset cache.
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    run(args.cache_dir, args.output_dir, args.device)
