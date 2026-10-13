"""Exact inner-product product search with explicit FAISS/NumPy backends.

The production backend is ``faiss.IndexFlatIP``. NumPy is an explicitly
selected exact fallback for environments where FAISS is unavailable; metadata
always records which backend actually produced an index.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDINGS = ROOT / "artifacts" / "embeddings"
DEFAULT_CATALOG = ROOT / "audit_output" / "canonical_catalog.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "retrieval"
EXPECTED_DIMENSION = 768
NORM_ATOL = 1e-5


class RetrievalError(ValueError):
    """Invalid query, artifacts, or index metadata."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = []
        with path.open(encoding="utf-8") as stream:
            for row_no, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RetrievalError(f"Malformed JSONL at {path}:{row_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise RetrievalError(f"Expected JSON object at {path}:{row_no}")
                rows.append(row)
        return rows
    except OSError as exc:
        raise RetrievalError(f"Cannot read {path}: {exc}") from exc


def discover_embedding_bundle(embedding_root: Path, entity_type: str) -> tuple[Path, dict[str, Any]]:
    """Resolve one complete, valid embedding summary without hardcoded run IDs."""
    if entity_type not in {"product", "scene"}:
        raise ValueError("entity_type must be 'product' or 'scene'")
    kind_dir = embedding_root / ("products" if entity_type == "product" else "scenes")
    candidates = []
    for summary_path in kind_dir.rglob("summary.json") if kind_dir.is_dir() else []:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("entity_type") != entity_type or summary.get("successful_embeddings") != summary.get("requested"):
                continue
            artifacts = summary.get("artifacts", {})
            matrix = Path(artifacts.get("embeddings", ""))
            mapping = Path(artifacts.get("ids", ""))
            if not matrix.is_absolute():
                matrix = summary_path.parent / matrix
            if not mapping.is_absolute():
                mapping = summary_path.parent / mapping
            if matrix.is_file() and mapping.is_file():
                candidates.append((summary_path.stat().st_mtime_ns, summary_path, matrix, mapping, summary))
        except (OSError, ValueError, TypeError):
            continue
    if not candidates:
        raise RetrievalError(f"No complete {entity_type} embedding bundle found under {kind_dir}")
    candidates.sort(key=lambda row: row[0], reverse=True)
    _, summary_path, matrix, mapping, summary = candidates[0]
    summary = dict(summary)
    summary["_summary_path"] = str(summary_path.resolve())
    summary["_matrix_path"] = str(matrix.resolve())
    summary["_mapping_path"] = str(mapping.resolve())
    return summary_path.parent, summary


def _catalog_ids(catalog_path: Path) -> tuple[list[str], str]:
    values = []
    with catalog_path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RetrievalError(f"Malformed canonical catalog at line {line_no}: {exc}") from exc
            value = row.get("product") if isinstance(row, dict) else None
            if not isinstance(value, str) or not value.strip():
                raise RetrievalError(f"Blank or invalid product ID in canonical catalog at line {line_no}")
            values.append(value.strip())
    if len(values) != len(set(values)):
        raise RetrievalError("Canonical catalog contains duplicate product IDs")
    return values, sha256_file(catalog_path)


def _load_bundle(summary: dict[str, Any], entity_type: str,
                 catalog_ids: set[str] | None = None) -> tuple[np.ndarray, list[str]]:
    matrix_path, mapping_path = Path(summary["_matrix_path"]), Path(summary["_mapping_path"])
    try:
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise RetrievalError(f"Cannot load {entity_type} matrix {matrix_path}: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape[1] != int(summary.get("embedding_dimension", -1)):
        raise RetrievalError(f"{entity_type} matrix shape {matrix.shape} disagrees with summary dimension")
    if matrix.shape[1] != EXPECTED_DIMENSION:
        raise RetrievalError(f"Expected {EXPECTED_DIMENSION} dimensions; {entity_type} matrix has {matrix.shape[1]}")
    if matrix.dtype != np.float32:
        raise RetrievalError(f"Expected float32 {entity_type} matrix, got {matrix.dtype}")
    if not np.isfinite(matrix).all():
        raise RetrievalError(f"{entity_type} matrix contains non-finite values")
    expected_matrix_digest = summary.get("artifact_sha256", {}).get("embeddings.npy")
    if expected_matrix_digest and sha256_file(matrix_path) != expected_matrix_digest:
        raise RetrievalError(f"{entity_type} matrix checksum differs from its embedding summary")
    expected_mapping_digest = summary.get("artifact_sha256", {}).get("ids.jsonl")
    if expected_mapping_digest and sha256_file(mapping_path) != expected_mapping_digest:
        raise RetrievalError(f"{entity_type} ID map checksum differs from its embedding summary")
    norms = np.linalg.norm(matrix, axis=1)
    if len(norms) != len(matrix) or not np.allclose(norms, 1.0, atol=NORM_ATOL):
        raise RetrievalError(f"{entity_type} matrix has non-normalized rows")
    rows = _read_jsonl(mapping_path)
    if len(rows) != len(matrix) or len(rows) != int(summary.get("successful_embeddings", -1)):
        raise RetrievalError(f"{entity_type} ID map count {len(rows)} does not match matrix rows {len(matrix)}")
    ids = []
    for expected_row, row in enumerate(rows):
        value = row.get("product_id") if entity_type == "product" else row.get("scene_id")
        value = value if isinstance(value, str) else row.get("entity_id")
        if not isinstance(value, str) or not value.strip():
            raise RetrievalError(f"Missing {entity_type} ID at mapped row {expected_row}")
        if row.get("row") != expected_row:
            raise RetrievalError(f"Non-contiguous {entity_type} ID map: expected row {expected_row}, got {row.get('row')}")
        ids.append(value)
    if len(ids) != len(set(ids)):
        raise RetrievalError(f"{entity_type} ID map contains duplicate IDs")
    if entity_type == "product" and catalog_ids is not None:
        mapped = set(ids)
        absent, extra = catalog_ids - mapped, mapped - catalog_ids
        if absent or extra:
            raise RetrievalError(f"Product ID map/catalog mismatch: {len(absent)} catalog IDs absent, {len(extra)} extras")
    return matrix, ids


def validate_vectors(vectors: Any, expected_dimension: int = EXPECTED_DIMENSION,
                     *, normalize: bool = False) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 1 or array.shape[0] != expected_dimension:
        raise RetrievalError(f"Query must be a 1D vector of dimension {expected_dimension}; got {array.shape}")
    if not np.isfinite(array).all():
        raise RetrievalError("Query vector contains NaN or infinite values")
    norm = float(np.linalg.norm(array))
    if not np.isfinite(norm) or norm <= 0:
        raise RetrievalError("Query vector must have a finite, nonzero L2 norm")
    if normalize:
        array = array / norm
    elif not np.isclose(norm, 1.0, atol=NORM_ATOL):
        raise RetrievalError(f"Product vectors must be L2-normalized; got query/index norm {norm}")
    return np.ascontiguousarray(array, dtype=np.float32)


class FlatProductIndex:
    """IndexFlatIP wrapper or explicit NumPy exact-search fallback."""

    def __init__(self, vectors: np.ndarray, product_ids: list[str], backend: str = "faiss") -> None:
        matrix = np.asarray(vectors)
        if matrix.ndim != 2 or matrix.shape[1] <= 0:
            raise RetrievalError(f"Product matrix must be 2D with nonzero dimension; got {matrix.shape}")
        if matrix.shape[1] != EXPECTED_DIMENSION:
            raise RetrievalError(f"Expected {EXPECTED_DIMENSION} product dimensions; got {matrix.shape[1]}")
        if matrix.dtype != np.float32:
            raise RetrievalError(f"Product matrix must be float32; got {matrix.dtype}")
        if matrix.shape[0] != len(product_ids):
            raise RetrievalError(f"Product ID count {len(product_ids)} does not match vector rows {matrix.shape[0]}")
        if any(not isinstance(value, str) or not value.strip() for value in product_ids):
            raise RetrievalError("Product index cannot contain missing IDs")
        if len(product_ids) != len(set(product_ids)):
            raise RetrievalError("Product ID map contains duplicate IDs")
        if not np.isfinite(matrix).all():
            raise RetrievalError("Product embeddings contain non-finite values")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=NORM_ATOL):
            raise RetrievalError("Product embeddings must be L2-normalized")
        if backend not in {"faiss", "numpy"}:
            raise ValueError("backend must be 'faiss' or 'numpy'")
        self.dimension = int(matrix.shape[1])
        self.product_ids = list(product_ids)
        self.backend = backend
        self._vectors: np.ndarray | None = matrix if backend == "numpy" else None
        self._index = None
        if backend == "faiss":
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError("FAISS is not installed in this Python environment; install faiss-cpu or explicitly use --backend numpy") from exc
            self._index = faiss.IndexFlatIP(self.dimension)
            self._index.add(np.ascontiguousarray(matrix, dtype=np.float32))
            self._faiss = faiss

    @property
    def ntotal(self) -> int:
        return len(self.product_ids) if self.backend == "numpy" else int(self._index.ntotal)

    def search(self, query: Any, top_k: int) -> list[dict[str, Any]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise RetrievalError("top_k must be a positive integer")
        if self.ntotal == 0:
            raise RetrievalError("Cannot search an empty product index")
        vector = validate_vectors(query, self.dimension, normalize=True)
        if self.backend == "numpy":
            scores = np.asarray(self._vectors @ vector, dtype=np.float32)
        else:
            found_scores, found_rows = self._index.search(vector.reshape(1, -1), self.ntotal)
            scores = np.empty(self.ntotal, dtype=np.float32)
            scores[found_rows[0]] = found_scores[0]
        order = sorted(range(self.ntotal), key=lambda row: (-float(scores[row]), self.product_ids[row]))
        return [{"product_id": self.product_ids[row], "rank": rank,
                 "cosine_similarity": float(scores[row])}
                for rank, row in enumerate(order[:min(top_k, self.ntotal)], 1)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
                temp = Path(stream.name)
                if self.backend == "numpy":
                    np.save(stream, np.asarray(self._vectors), allow_pickle=False)
                else:
                    stream.close()
                    self._faiss.write_index(self._index, str(temp))
                    with temp.open("rb") as written:
                        os.fsync(written.fileno())
                if not stream.closed:
                    stream.flush()
                    os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            if temp is not None and temp.exists():
                temp.unlink()

    @classmethod
    def load(cls, path: Path, backend: str, product_ids: list[str], dimension: int) -> "FlatProductIndex":
        if backend == "numpy":
            try:
                vectors = np.load(path, mmap_mode="r", allow_pickle=False)
            except Exception as exc:
                raise RetrievalError(f"Cannot load NumPy exact index {path}: {exc}") from exc
            instance = cls(vectors, product_ids, backend="numpy")
        elif backend == "faiss":
            try:
                import faiss
                index = faiss.read_index(str(path))
            except Exception as exc:
                raise RetrievalError(f"Cannot load FAISS index {path}: {exc}") from exc
            if int(index.d) != dimension or int(index.ntotal) != len(product_ids):
                raise RetrievalError("FAISS index dimension/row count does not match metadata")
            if int(index.metric_type) != int(faiss.METRIC_INNER_PRODUCT):
                raise RetrievalError("Expected a FAISS inner-product index")
            instance = cls.__new__(cls)
            instance.dimension, instance.product_ids, instance.backend = dimension, product_ids, backend
            instance._index, instance._faiss, instance._vectors = index, faiss, None
        else:
            raise RetrievalError(f"Unknown index backend {backend!r}")
        if instance.dimension != dimension or instance.ntotal != len(product_ids):
            raise RetrievalError("Loaded index shape does not match metadata")
        return instance


def build_product_index(embedding_root: Path = DEFAULT_EMBEDDINGS,
                        catalog_path: Path = DEFAULT_CATALOG,
                        output_root: Path = DEFAULT_OUTPUT,
                        backend: str = "faiss") -> dict[str, Any]:
    """Validate source artifacts and atomically build an exact product index."""
    started = time.perf_counter()
    bundle_dir, product_summary = discover_embedding_bundle(embedding_root, "product")
    catalog_ids, catalog_sha256 = _catalog_ids(catalog_path)
    if product_summary.get("source_fingerprints", {}).get("catalog_fingerprint") != catalog_sha256:
        raise RetrievalError("Product embedding catalog fingerprint differs from the current canonical catalog")
    product_matrix, product_ids = _load_bundle(product_summary, "product", set(catalog_ids))
    expected = (len(product_ids), EXPECTED_DIMENSION)
    if product_matrix.shape != expected:
        raise RetrievalError(f"Expected product matrix {expected}; got {product_matrix.shape}")
    t0 = time.perf_counter()
    index = FlatProductIndex(product_matrix, product_ids, backend=backend)
    build_seconds = time.perf_counter() - t0
    config = {"backend": backend, "index_type": "IndexFlatIP" if backend == "faiss" else "NumPyExactIP",
              "metric": "inner_product", "dimension": index.dimension, "product_count": index.ntotal,
              "model_revision": product_summary.get("model_revision"),
              "embedding_fingerprint": product_summary.get("model_configuration_fingerprint"),
              "catalog_fingerprint": catalog_sha256,
              "product_matrix_sha256": sha256_file(Path(product_summary["_matrix_path"])),
              "product_mapping_sha256": sha256_file(Path(product_summary["_mapping_path"]))}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    run_dir = output_root / fingerprint
    index_path = run_dir / ("products.faiss" if backend == "faiss" else "products.npy")
    index.save(index_path)
    mapping_path = run_dir / "product_ids.jsonl"
    source_rows = _read_jsonl(Path(product_summary["_mapping_path"]))
    mapped_rows = [{"row": row, "product_id": product_id} for row, product_id in enumerate(product_ids)]
    _atomic_jsonl(mapping_path, mapped_rows)
    metadata = {
        "schema_version": 1, "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "backend": backend, "index_type": config["index_type"], "metric": "inner_product",
        "dimension": index.dimension, "ntotal": index.ntotal, "top_k_similarity": "cosine_similarity_for_l2_normalized_vectors",
        "model_id": product_summary.get("model_id"), "model_revision": product_summary.get("model_revision"),
        "embedding_fingerprint": product_summary.get("model_configuration_fingerprint"),
        "catalog_fingerprint": catalog_sha256, "image_manifest_fingerprint": product_summary.get("source_fingerprints", {}).get("image_manifest_fingerprint"),
        "product_matrix_path": product_summary["_matrix_path"], "product_embedding_summary_path": product_summary["_summary_path"],
        "product_embedding_fingerprint": product_summary.get("model_configuration_fingerprint"),
        "scene_embedding_summary_path": discover_embedding_bundle(embedding_root, "scene")[1]["_summary_path"],
        "index_fingerprint": fingerprint, "index_path": str(index_path.resolve()),
        "product_id_mapping_path": str(mapping_path.resolve()),
        "product_matrix_sha256": config["product_matrix_sha256"], "source_mapping_sha256": config["product_mapping_sha256"],
        "index_sha256": sha256_file(index_path), "product_id_mapping_sha256": sha256_file(mapping_path),
        "build_seconds": build_seconds, "total_seconds": time.perf_counter() - started,
        "canonical_catalog_path": str(catalog_path.resolve()), "canonical_catalog_count": len(catalog_ids),
        "row_mapping_validated": len(product_ids) == len(catalog_ids) and set(product_ids) == set(catalog_ids),
    }
    metadata_path = run_dir / "index_metadata.json"
    _atomic_json(metadata_path, metadata)
    _atomic_json(output_root / "active_index.json", {"index_metadata_path": str(metadata_path.resolve()),
                                                        "index_fingerprint": fingerprint, "backend": backend})
    return {**metadata, "metadata_path": str(metadata_path.resolve()), "product_mapping_source_rows": len(source_rows)}


class RetrievalEngine:
    """Search a persisted index using saved validation-scene vectors or vectors."""

    def __init__(self, index: FlatProductIndex, metadata: dict[str, Any],
                 scene_vectors: np.ndarray, scene_ids: list[str], catalog_ids: set[str]) -> None:
        self.index, self.metadata = index, metadata
        self.scene_vectors, self.scene_ids = scene_vectors, scene_ids
        self.catalog_ids = catalog_ids
        self.scene_rows = {scene_id: row for row, scene_id in enumerate(scene_ids)}
        if len(self.scene_rows) != len(scene_ids):
            raise RetrievalError("Scene embedding ID map contains duplicates")
        if self.index.ntotal != len(self.index.product_ids):
            raise RetrievalError("Index size and product ID mapping disagree")
        if not set(self.index.product_ids).issubset(catalog_ids):
            raise RetrievalError("Index contains a product ID absent from canonical catalog")

    @classmethod
    def load(cls, output_root: Path = DEFAULT_OUTPUT,
             embedding_root: Path = DEFAULT_EMBEDDINGS,
             catalog_path: Path = DEFAULT_CATALOG,
             metadata_path: Path | None = None) -> "RetrievalEngine":
        if metadata_path is None:
            pointer = output_root / "active_index.json"
            try:
                metadata_path = Path(json.loads(pointer.read_text(encoding="utf-8"))["index_metadata_path"])
            except (OSError, KeyError, json.JSONDecodeError) as exc:
                raise RetrievalError(f"Cannot resolve active retrieval index at {pointer}: {exc}") from exc
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RetrievalError(f"Cannot read index metadata {metadata_path}: {exc}") from exc
        index_path = Path(metadata["index_path"])
        mapping_path = Path(metadata["product_id_mapping_path"])
        if sha256_file(index_path) != metadata.get("index_sha256"):
            raise RetrievalError("Saved index checksum does not match metadata")
        if sha256_file(mapping_path) != metadata.get("product_id_mapping_sha256"):
            raise RetrievalError("Saved product ID mapping checksum does not match metadata")
        mappings = _read_jsonl(mapping_path)
        ids = []
        for row_no, row in enumerate(mappings):
            if row.get("row") != row_no or not isinstance(row.get("product_id"), str) or not row["product_id"].strip():
                raise RetrievalError(f"Invalid product row mapping at row {row_no}")
            ids.append(row["product_id"])
        if len(ids) != len(set(ids)) or len(ids) != int(metadata.get("ntotal", -1)):
            raise RetrievalError("Saved product mapping has duplicate IDs or disagrees with index size")
        index = FlatProductIndex.load(index_path, metadata["backend"], ids, int(metadata["dimension"]))
        catalog_ids, catalog_sha256 = _catalog_ids(catalog_path)
        if catalog_sha256 != metadata.get("catalog_fingerprint"):
            raise RetrievalError("Canonical catalog changed since this index was built")
        if set(ids) != set(catalog_ids):
            raise RetrievalError("Saved product mapping does not exactly match the canonical catalog")
        _, scene_summary = discover_embedding_bundle(embedding_root, "scene")
        if scene_summary.get("model_revision") != metadata.get("model_revision"):
            raise RetrievalError("Scene and product embeddings use different model revisions")
        scene_vectors, scene_ids = _load_bundle(scene_summary, "scene")
        if scene_vectors.shape[1] != int(metadata["dimension"]):
            raise RetrievalError("Scene query dimension does not match product index")
        return cls(index, metadata, scene_vectors, scene_ids, set(catalog_ids))

    def search_embedding(self, query: Any, top_k: int = 10) -> list[dict[str, Any]]:
        results = self.index.search(query, top_k)
        for result in results:
            if result["product_id"] not in self.catalog_ids:
                raise RetrievalError(f"Index returned product absent from canonical catalog: {result['product_id']}")
        return results

    def search_scene(self, scene_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        if scene_id not in self.scene_rows:
            raise RetrievalError(f"Unknown validation scene ID {scene_id!r}")
        row = self.scene_rows[scene_id]
        results = self.search_embedding(self.scene_vectors[row], top_k)
        return [{"scene_id": scene_id, **result} for result in results]

    def search_all_scenes(self, top_k: int = 10) -> list[dict[str, Any]]:
        return [{"scene_id": scene_id, "results": self.search_scene(scene_id, top_k)} for scene_id in self.scene_ids]


def write_retrieval_report(engine: RetrievalEngine, scene_ids: list[str], top_k: int,
                           output_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    rows = []
    for scene_id in scene_ids:
        rows.extend(engine.search_scene(scene_id, top_k))
    elapsed = time.perf_counter() - started
    _atomic_jsonl(output_path, rows)
    summary = {"query_count": len(scene_ids), "top_k": top_k, "result_count": len(rows),
               "search_seconds": elapsed, "average_query_seconds": elapsed / len(scene_ids) if scene_ids else 0.0,
               "backend": engine.metadata["backend"], "index_type": engine.metadata["index_type"],
               "index_vector_count": engine.index.ntotal, "output_path": str(output_path.resolve()),
               "unique_products_per_query": all(len({r["product_id"] for r in rows if r["scene_id"] == sid}) == min(top_k, engine.index.ntotal) for sid in scene_ids),
               "all_products_in_catalog": all(row["product_id"] in engine.catalog_ids for row in rows),
               "scores_descending_per_scene": all(
                   [r["cosine_similarity"] for r in rows if r["scene_id"] == sid] ==
                   sorted([r["cosine_similarity"] for r in rows if r["scene_id"] == sid], reverse=True)
                   for sid in scene_ids)}
    _atomic_json(output_path.with_name(output_path.stem + "_summary.json"), summary)
    return summary
