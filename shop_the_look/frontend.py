"""Upload validation and live-image retrieval helpers for the Streamlit app."""
from __future__ import annotations

import hashlib
import io
import json
import atexit
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from shop_the_look.models.fashionsiglip import validate_and_normalize
from shop_the_look.retrieval.faiss_index import ROOT as RETRIEVAL_ROOT
from shop_the_look.retrieval.multicrop import aggregate_max_scores, generate_crops

if TYPE_CHECKING:
    from shop_the_look.retrieval.faiss_index import RetrievalEngine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "image_cache" / "manifest.json"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def decode_upload(data: bytes, filename: str, *, max_bytes: int = MAX_UPLOAD_BYTES,
                  max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
    """Decode a supported upload, enforce size limits, and return an RGB copy."""
    suffix = Path(filename or "").suffix.lower().lstrip(".")
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("Upload a JPG, JPEG, or PNG image.")
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ValueError(f"Image exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            expected_format = "JPEG" if suffix in {"jpg", "jpeg"} else "PNG"
            if opened.format != expected_format:
                raise ValueError("The file extension does not match its JPG or PNG contents.")
            width, height = opened.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ValueError("Image dimensions are invalid or exceed the 40 megapixel limit.")
            opened.load()
            return opened.convert("RGB").copy()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Could not decode the uploaded image: {exc}") from exc


def load_product_image_map(manifest_path: Path = DEFAULT_MANIFEST, root: Path = ROOT) -> dict[str, Path]:
    """Map catalog product IDs to existing, valid cached image files only."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records") if isinstance(manifest, dict) else None
    if not isinstance(records, list):
        raise ValueError("Product image manifest has no records list.")
    images: dict[str, Path] = {}
    for row in records:
        if not isinstance(row, dict) or row.get("entity_type") != "product":
            continue
        if row.get("download_status") not in {"success", "reused_cache"} or row.get("validation_outcome") != "valid_image":
            continue
        product_id, raw = row.get("entity_id"), row.get("cache_path")
        if not isinstance(product_id, str) or not isinstance(raw, str):
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root.resolve())
            if resolved.is_file() and resolved.stat().st_size:
                with Image.open(resolved) as image:
                    image.verify()
                images[product_id] = resolved
        except (OSError, ValueError, Image.DecompressionBombError):
            continue
    return images


def retrieve_upload(image: Image.Image, mode: str, model: Any, engine: RetrievalEngine,
                    top_k: int = 10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Embed a new in-memory image and search the saved product index."""
    if mode not in {"Whole-scene", "Multi-crop"}:
        raise ValueError("mode must be Whole-scene or Multi-crop")
    if top_k not in {5, 10, 20}:
        raise ValueError("top_k must be 5, 10, or 20")
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError("Image dimensions are invalid or exceed the 40 megapixel limit.")
    if mode == "Whole-scene":
        vectors = validate_and_normalize(model.embed_images([rgb]), 1, expected_dimension=768)
        result = engine.search_embedding(vectors[0], top_k=top_k)
        return result, {"query_source": "uploaded_image", "embedding_count": 1, "dimension": 768,
                        "norms": np.linalg.norm(vectors, axis=1).tolist()}

    # The upload digest identifies this crop set; no validation scene cache is consulted.
    upload_digest = hashlib.sha256(rgb.tobytes() + f"{width}x{height}".encode()).hexdigest()
    crops = generate_crops(f"upload-{upload_digest[:16]}", "<uploaded-image>", width, height)
    crop_images = [rgb.crop((crop.x0, crop.y0, crop.x1, crop.y1)) for crop in crops]
    vectors = validate_and_normalize(model.embed_images(crop_images), len(crop_images), expected_dimension=768)
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("Crop embeddings are not L2-normalized.")
    if hasattr(engine, "search_multi_embeddings"):
        ranked = engine.search_multi_embeddings(vectors, [crop.crop_id for crop in crops], top_k)
    else:
        product_ids = engine.index.product_ids
        scores = np.empty((len(crops), len(product_ids)), dtype=np.float32)
        for row, vector in enumerate(vectors):
            full_ranking = engine.search_embedding(vector, top_k=engine.index.ntotal)
            score_by_id = {item["product_id"]: item["cosine_similarity"] for item in full_ranking}
            scores[row] = [score_by_id[product_id] for product_id in product_ids]
        ranked, _, _ = aggregate_max_scores(scores, product_ids, [crop.crop_id for crop in crops])
        ranked = ranked[:top_k]
    return ranked, {"query_source": "uploaded_image", "embedding_count": len(crops),
                            "dimension": 768, "norms": np.linalg.norm(vectors, axis=1).tolist(),
                            "crop_count": len(crops)}


class LiveIndexSearchWorker:
    """Hold the FAISS index in a process separate from PyTorch/OpenMP inference."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [sys.executable, "-u", "-m", "shop_the_look.retrieval.live_index_worker"],
            cwd=RETRIEVAL_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        ready_line = self._process.stdout.readline() if self._process.stdout else ""
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            self.close()
            raise RuntimeError("The saved product-index search worker did not start correctly.") from exc
        if not ready.get("ready"):
            detail = ready.get("error", "unknown worker startup error")
            self.close()
            raise RuntimeError(detail.replace(str(RETRIEVAL_ROOT), "<project>"))
        self.index_count = int(ready["index_count"])
        self.dimension = int(ready["dimension"])
        atexit.register(self.close)

    def _request(self, request: dict) -> list[dict]:
        with self._lock:
            if self._process.poll() is not None or not self._process.stdin or not self._process.stdout:
                raise RuntimeError("The local product-index search worker is no longer running.")
            self._process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("The local product-index search worker returned an invalid response.") from exc
        if "error" in response:
            raise RuntimeError(response["error"].replace(str(RETRIEVAL_ROOT), "<project>"))
        return response["results"]

    def search_embedding(self, query: np.ndarray, top_k: int) -> list[dict]:
        vector = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.dimension or not np.isfinite(vector).all():
            raise ValueError("The query embedding does not match the saved index.")
        return self._request({"mode": "whole", "queries": vector.tolist(), "top_k": top_k})

    def search_multi_embeddings(self, vectors: np.ndarray, crop_ids: list[str], top_k: int) -> list[dict]:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension or not np.isfinite(matrix).all():
            raise ValueError("Crop embeddings do not match the saved product index.")
        return self._request({"mode": "multi", "queries": matrix.tolist(),
                              "crop_ids": crop_ids, "top_k": top_k})

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
