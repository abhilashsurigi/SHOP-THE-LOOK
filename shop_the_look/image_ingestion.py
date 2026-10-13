"""Resumable, validated image ingestion for Shop-the-Look IDs.

Uses urllib from the standard library because HTTPX is not a project dependency.
TLS certificate verification stays enabled. Product URLs are an empirical
hypothesis; scene URLs follow the assignment's documented rule.
"""

from __future__ import annotations

import argparse
import certifi
import hashlib
import json
import os
import socket
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError


PRODUCT_URL_STRATEGIES: dict[str, Callable[[str], str]] = {
    "pinterest_cdn_hypothesis": lambda entity_id: _pinterest_url(entity_id, "https"),
}


def _pinterest_url(entity_id: str, scheme: str) -> str:
    return (
        f"{scheme}://i.pinimg.com/400x/{entity_id[:2]}/{entity_id[2:4]}/"
        f"{entity_id[4:6]}/{entity_id}.jpg"
    )


def resolve_image_url(entity_type: str, entity_id: str, product_strategy: str = "pinterest_cdn_hypothesis") -> tuple[str, str, str]:
    """Return (URL, strategy name, provenance label)."""
    if not isinstance(entity_id, str) or not entity_id.strip():
        raise ValueError("entity ID must be a nonempty string")
    entity_id = entity_id.strip()
    if entity_type == "scene":
        return _pinterest_url(entity_id, "http"), "assignment_scene_rule", "assignment_specified"
    if entity_type != "product":
        raise ValueError(f"unsupported entity type: {entity_type}")
    try:
        url = PRODUCT_URL_STRATEGIES[product_strategy](entity_id)
    except KeyError as exc:
        raise ValueError(f"unknown product URL strategy: {product_strategy}") from exc
    return url, product_strategy, "inferred_empirical_hypothesis"


def _atomic_write(path: Path, write_fn: Callable[[Any], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_path = stream.name
            write_fn(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except OSError:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_write(path, lambda stream: (json.dump(payload, stream, ensure_ascii=False, indent=2), stream.write("\n")))


def load_inventory(catalog_path: str | Path, associations_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load unique products/scenes and count malformed or missing input rows."""
    issues = {"catalog_invalid_rows": [], "association_invalid_rows": []}
    products: dict[str, None] = {}
    try:
        with Path(catalog_path).open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    issues["catalog_invalid_rows"].append({"line": line_no, "reason": "malformed_json"})
                    continue
                product = row.get("product") if isinstance(row, dict) else None
                if not isinstance(product, str) or not product.strip():
                    issues["catalog_invalid_rows"].append({"line": line_no, "reason": "missing_or_invalid_product"})
                    continue
                products.setdefault(product.strip(), None)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Unable to read canonical catalog '{catalog_path}': {exc}") from exc

    scenes: dict[str, set[str]] = {}
    try:
        raw = json.loads(Path(associations_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read validation associations '{associations_path}': {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"Validation associations must be a JSON array: '{associations_path}'")
    for row_no, row in enumerate(raw, 1):
        scene = row.get("scene") if isinstance(row, dict) else None
        product = row.get("product") if isinstance(row, dict) else None
        if not isinstance(scene, str) or not scene.strip():
            issues["association_invalid_rows"].append({"row": row_no, "reason": "missing_or_invalid_scene"})
            continue
        scene_id = scene.strip()
        scenes.setdefault(scene_id, set())
        if isinstance(product, str) and product.strip():
            scenes[scene_id].add(product.strip())
    return {
        "product": list(products),
        "scene": list(scenes),
        "scene_products": {scene: sorted(product_ids) for scene, product_ids in scenes.items()},
    }, issues


@dataclass
class DownloadConfig:
    timeout: float = 15.0
    max_retries: int = 2
    workers: int = 8
    backoff_base: float = 0.5
    backoff_max: float = 4.0
    min_dimension: int = 1
    max_bytes: int = 20 * 1024 * 1024
    user_agent: str = "ShopTheLookImageIngestion/1.0"
    manifest_checkpoint_every: int = 100


class ImageIngestor:
    def __init__(self, output_dir: str | Path, config: DownloadConfig | None = None,
                 product_url_strategy: str = "pinterest_cdn_hypothesis") -> None:
        self.output_dir = Path(output_dir)
        self.config = config or DownloadConfig()
        self.product_url_strategy = product_url_strategy
        self.manifest_path = self.output_dir / "manifest.json"
        self._old_manifest = self._read_manifest()

    def _read_manifest(self) -> dict[str, Any]:
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) and isinstance(value.get("records"), list) else {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}

    def _cache_path(self, entity_type: str, entity_id: str) -> Path:
        # IDs are expected to be opaque, but encode unsafe path characters to
        # avoid traversal while keeping ordinary IDs stable and readable.
        safe = "".join(ch if ch.isalnum() or ch in "-_" else f"_{ord(ch):02x}" for ch in entity_id)
        return self.output_dir / "images" / ("products" if entity_type == "product" else "scenes") / f"{safe}.img"

    @staticmethod
    def _validate_image(path: Path, min_dimension: int = 1) -> dict[str, Any]:
        try:
            if path.stat().st_size == 0:
                return {"ok": False, "error_category": "empty_image", "error_message": "Downloaded file is empty"}
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    fmt = image.format
            if width < min_dimension or height < min_dimension:
                return {"ok": False, "error_category": "below_minimum_dimension",
                        "error_message": f"Image dimensions {width}x{height} below minimum {min_dimension}",
                        "width": width, "height": height, "format": fmt}
            return {"ok": True, "width": width, "height": height, "format": fmt}
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            return {"ok": False, "error_category": "corrupt_image", "error_message": str(exc)[:300]}

    def _cached_record(self, entity_type: str, entity_id: str, url: str, strategy: str, provenance: str, path: Path) -> dict[str, Any] | None:
        old = next((r for r in self._old_manifest.get("records", [])
                    if r.get("entity_type") == entity_type and r.get("entity_id") == entity_id), None)
        cache_metadata = old
        if old and old.get("download_status") not in {"success", "reused_cache"}:
            cache_metadata = old.get("preserved_cache")
        if not path.is_file() or not isinstance(cache_metadata, dict):
            return None
        try:
            size = path.stat().st_size
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None
        if cache_metadata.get("source_url") != url or cache_metadata.get("sha256") != digest or cache_metadata.get("downloaded_bytes") != size:
            return None
        validation = self._validate_image(path, self.config.min_dimension)
        if not validation["ok"]:
            return None
        return {
            "entity_type": entity_type, "entity_id": entity_id,
            "source_url": url, "url_strategy": strategy, "url_provenance": provenance,
            "cache_path": str(path), "download_status": "reused_cache",
            "http_status": cache_metadata.get("http_status"), "content_type": cache_metadata.get("content_type"),
            "downloaded_bytes": size, "sha256": digest,
            "image_format": validation.get("format"), "width": validation.get("width"),
            "height": validation.get("height"), "retry_count": 0,
            "error_category": None, "error_message": None,
            "validation_outcome": "valid_image", "cache_reused": True,
        }

    def _download_one(self, entity_type: str, entity_id: str, force: bool) -> dict[str, Any]:
        url, strategy, provenance = resolve_image_url(entity_type, entity_id, self.product_url_strategy)
        path = self._cache_path(entity_type, entity_id)
        if not force:
            cached = self._cached_record(entity_type, entity_id, url, strategy, provenance, path)
            if cached:
                return cached
        path.parent.mkdir(parents=True, exist_ok=True)
        # Leftovers from interrupted attempts are never valid cache entries.
        for stale_part in path.parent.glob(f".{path.name}.*.part"):
            try:
                stale_part.unlink()
            except OSError:
                pass
        attempts = 0
        last: dict[str, Any] = {}
        retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
        while attempts <= self.config.max_retries:
            attempts += 1
            temp_name: str | None = None
            retry_this_attempt = True
            try:
                request = urllib.request.Request(url, headers={"User-Agent": self.config.user_agent}, method="GET")
                # Match the existing data_audit.py TLS setup while retaining
                # certificate verification through Certifi's CA bundle.
                context = ssl.create_default_context(cafile=certifi.where())
                with urllib.request.urlopen(request, timeout=self.config.timeout, context=context) as response:
                    status = getattr(response, "status", None)
                    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                    if status is None or not (200 <= status < 300):
                        last = {"http_status": status, "content_type": content_type,
                                "error_category": "http_failure", "error_message": f"Unexpected HTTP status {status}"}
                    elif not content_type.startswith("image/"):
                        last = {"http_status": status, "content_type": content_type,
                                "error_category": "invalid_content_type", "error_message": f"Expected image/*, received {content_type or '(missing)'}"}
                        retry_this_attempt = False
                    else:
                        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".part", delete=False) as temp:
                            temp_name = temp.name
                            total = 0
                            digest = hashlib.sha256()
                            while True:
                                block = response.read(64 * 1024)
                                if not block:
                                    break
                                total += len(block)
                                if total > self.config.max_bytes:
                                    raise ValueError("Image exceeds configured maximum byte size")
                                digest.update(block)
                                temp.write(block)
                            temp.flush()
                            os.fsync(temp.fileno())
                        if total == 0:
                            last = {"http_status": status, "content_type": content_type, "downloaded_bytes": 0,
                                    "error_category": "empty_image", "error_message": "Response body is empty"}
                            retry_this_attempt = False
                        else:
                            validation = self._validate_image(Path(temp_name), self.config.min_dimension)
                            if not validation["ok"]:
                                last = {"http_status": status, "content_type": content_type, "downloaded_bytes": total,
                                        **validation}
                                retry_this_attempt = False
                            else:
                                os.replace(temp_name, path)
                                temp_name = None
                                return {
                                    "entity_type": entity_type, "entity_id": entity_id,
                                    "source_url": url, "url_strategy": strategy, "url_provenance": provenance,
                                    "cache_path": str(path), "download_status": "success",
                                    "http_status": status, "content_type": content_type,
                                    "downloaded_bytes": total, "sha256": digest.hexdigest(),
                                    "image_format": validation.get("format"), "width": validation.get("width"),
                                    "height": validation.get("height"), "retry_count": attempts - 1,
                                    "error_category": None, "error_message": None,
                                    "validation_outcome": "valid_image", "cache_reused": False,
                                }
            except urllib.error.HTTPError as exc:
                last = {"http_status": exc.code, "content_type": exc.headers.get("Content-Type", "").split(";", 1)[0].lower() if exc.headers else None,
                        "error_category": "http_failure", "error_message": str(exc.reason)[:300]}
                if exc.code not in retryable_statuses:
                    break
            except urllib.error.URLError as exc:
                reason = exc.reason
                if isinstance(reason, ssl.SSLCertVerificationError):
                    category = "tls_certificate_failure"
                    retry_this_attempt = False
                elif isinstance(reason, ssl.SSLError):
                    category = "tls_failure"
                    retry_this_attempt = False
                elif isinstance(reason, socket.gaierror):
                    category = "dns_failure"
                elif isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower():
                    category = "timeout"
                else:
                    category = "connection_failure"
                last = {"http_status": None, "error_category": category,
                        "error_message": str(reason)[:300]}
            except ssl.SSLCertVerificationError as exc:
                last = {"http_status": None, "error_category": "tls_certificate_failure",
                        "error_message": str(exc)[:300]}
                retry_this_attempt = False
            except ssl.SSLError as exc:
                last = {"http_status": None, "error_category": "tls_failure",
                        "error_message": str(exc)[:300]}
                retry_this_attempt = False
            except (TimeoutError, ConnectionError, OSError) as exc:
                last = {"http_status": None,
                        "error_category": "dns_failure" if isinstance(exc, socket.gaierror) else (
                            "timeout" if isinstance(exc, (TimeoutError, socket.timeout)) or "timed out" in str(exc).lower()
                            else "connection_failure"),
                        "error_message": str(exc)[:300]}
            except ValueError as exc:
                last = {"http_status": None, "error_category": "oversized_response", "error_message": str(exc)[:300]}
                break
            finally:
                if temp_name:
                    try:
                        os.unlink(temp_name)
                    except OSError:
                        pass
            if not retry_this_attempt:
                break
            if attempts <= self.config.max_retries:
                delay = min(self.config.backoff_base * (2 ** (attempts - 1)), self.config.backoff_max)
                time.sleep(delay)

        preserved_cache = None
        if path.is_file():
            prior = next((r for r in self._old_manifest.get("records", [])
                          if r.get("entity_type") == entity_type and r.get("entity_id") == entity_id), None)
            if prior and prior.get("source_url") == url and prior.get("sha256"):
                preserved_cache = {
                    "source_url": url,
                    "http_status": prior.get("http_status"),
                    "content_type": prior.get("content_type"),
                    "downloaded_bytes": prior.get("downloaded_bytes"),
                    "sha256": prior.get("sha256"),
                }
            elif prior and isinstance(prior.get("preserved_cache"), dict):
                preserved_cache = prior["preserved_cache"]
        return {
            "entity_type": entity_type, "entity_id": entity_id,
            "source_url": url, "url_strategy": strategy, "url_provenance": provenance,
            "cache_path": str(path), "download_status": "failed",
            "http_status": last.get("http_status"), "content_type": last.get("content_type"),
            "downloaded_bytes": last.get("downloaded_bytes", 0),
            "image_format": last.get("format"), "width": last.get("width"), "height": last.get("height"),
            "retry_count": attempts - 1, "error_category": last.get("error_category", "unknown_error"),
            "error_message": last.get("error_message"),
            "validation_outcome": last.get("error_category", "unknown_error"), "cache_reused": False,
            "existing_cache_preserved": path.is_file(),
            "preserved_cache": preserved_cache,
        }

    def _write_manifest_checkpoint(self, records: list[dict[str, Any]]) -> None:
        by_key = {(r.get("entity_type"), r.get("entity_id")): r
                  for r in self._old_manifest.get("records", [])}
        by_key.update({(r["entity_type"], r["entity_id"]): r for r in records})
        manifest = {
            "schema_version": 1,
            "records": sorted(by_key.values(), key=lambda r: (r.get("entity_type", ""), r.get("entity_id", ""))),
        }
        _atomic_json(self.manifest_path, manifest)
        self._old_manifest = manifest

    def ingest(self, inventory: dict[str, Any], entity_type: str = "both", limit: int | None = None,
               force: bool = False, input_issues: dict[str, Any] | None = None) -> dict[str, Any]:
        selected = ["product", "scene"] if entity_type == "both" else (["product"] if entity_type == "products" else ["scene"])
        requested: list[tuple[str, str]] = []
        for kind in selected:
            ids = list(dict.fromkeys(inventory.get(kind, [])))
            if limit is not None:
                ids = ids[:max(0, limit)]
            requested.extend((kind, value) for value in ids)
        scene_products = inventory.get("scene_products", {})
        results: list[dict[str, Any] | None] = [None] * len(requested)
        completed_records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, self.config.workers)) as executor:
            futures = {
                executor.submit(self._download_one, kind, entity_id, force): index
                for index, (kind, entity_id) in enumerate(requested)
            }
            since_checkpoint = 0
            for future in as_completed(futures):
                index = futures[future]
                record = future.result()
                if record["entity_type"] == "scene":
                    record["associated_product_ids"] = list(scene_products.get(record["entity_id"], []))
                results[index] = record
                completed_records.append(record)
                since_checkpoint += 1
                if since_checkpoint >= self.config.manifest_checkpoint_every:
                    self._write_manifest_checkpoint(completed_records)
                    since_checkpoint = 0
        records = [record for record in results if record is not None]
        if completed_records:
            self._write_manifest_checkpoint(completed_records)
        summary: dict[str, Any] = {
            "total_requested_images": len(records),
            "successful_downloads": sum(r["download_status"] == "success" for r in records),
            "successful_image_decodes": sum(r["validation_outcome"] == "valid_image" for r in records),
            "reused_cached_images": sum(r["cache_reused"] for r in records),
            "http_failures": sum(r["error_category"] == "http_failure" for r in records),
            "connection_failures": sum(r["error_category"] == "connection_failure" for r in records),
            "timeouts": sum(r["error_category"] == "timeout" for r in records),
            "invalid_content_types": sum(r["error_category"] == "invalid_content_type" for r in records),
            "empty_images": sum(r["error_category"] == "empty_image" for r in records),
            "corrupt_images": sum(r["error_category"] == "corrupt_image" for r in records),
            "retry_count": sum(r["retry_count"] for r in records),
            "failures_by_error_category": {},
            "by_entity_type": {},
        }
        for record in records:
            category = record.get("error_category")
            if category:
                summary["failures_by_error_category"][category] = summary["failures_by_error_category"].get(category, 0) + 1
        for kind in selected:
            subset = [r for r in records if r["entity_type"] == kind]
            summary["by_entity_type"][kind] = {
                "requested": len(subset),
                "successful_downloads": sum(r["download_status"] == "success" for r in subset),
                "reused_cached_images": sum(r["cache_reused"] for r in subset),
                "failed": sum(r["download_status"] == "failed" for r in subset),
            }
        report = {
            "execution_status": "completed_with_failures" if any(r["download_status"] == "failed" for r in records) else "success",
            "sample_run": limit is not None,
            "limit_per_entity_type": limit,
            "entity_type": entity_type,
            "url_strategies": {
                "scene": {"strategy": "assignment_scene_rule", "provenance": "assignment_specified", "url_pattern": "http://i.pinimg.com/400x/{id[0:2]}/{id[2:4]}/{id[4:6]}/{id}.jpg"},
                "product": {"strategy": self.product_url_strategy, "provenance": "inferred_empirical_hypothesis"},
            },
            "summary": summary,
            "input_issues": input_issues or {},
            "records": records,
        }
        # Manifest carries the latest outcome for requested IDs while retaining
        # prior entries for records not selected by a limited run.
        try:
            _atomic_json(self.output_dir / "ingestion_report.json", report)
            _atomic_json(self.output_dir / "ingestion_summary.json", {k: v for k, v in report.items() if k != "records"})
        except OSError as exc:
            raise ValueError(f"Unable to write image ingestion reports under '{self.output_dir}': {exc}") from exc
        return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and validate product and scene images.")
    parser.add_argument("--entity-type", choices=("products", "scenes", "both"), default="both")
    parser.add_argument("--limit", type=int, default=None, help="Maximum entities per selected type; marks report as a sample")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=15.0, help="Per-request socket timeout in seconds (urllib backend)")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Atomic manifest checkpoint interval")
    parser.add_argument("--resume", action="store_true", help="Explicitly select resume mode; valid cache reuse is the default unless --force is used")
    parser.add_argument("--force", action="store_true", help="Redownload even if a valid cached image exists")
    parser.add_argument("--output-dir", default="image_cache", help="Cache and report directory")
    parser.add_argument("--catalog", default="audit_output/canonical_catalog.jsonl")
    parser.add_argument("--associations", default="audit_output/validation_associations.json")
    parser.add_argument("--product-url-strategy", choices=tuple(PRODUCT_URL_STRATEGIES), default="pinterest_cdn_hypothesis")
    parser.add_argument("--min-dimension", type=int, default=1)
    parser.add_argument("--user-agent", default="ShopTheLookImageIngestion/1.0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0 or args.max_retries < 0 or args.workers < 1 or args.min_dimension < 1 or args.checkpoint_every < 1:
        print("ERROR: timeout/workers/min-dimension/checkpoint-every must be positive and max-retries nonnegative", file=sys.stderr)
        return 2
    try:
        inventory, issues = load_inventory(args.catalog, args.associations)
        ingestor = ImageIngestor(args.output_dir, DownloadConfig(
            timeout=args.timeout, max_retries=args.max_retries, workers=args.workers,
            min_dimension=args.min_dimension, user_agent=args.user_agent,
            manifest_checkpoint_every=args.checkpoint_every,
        ), args.product_url_strategy)
        report = ingestor.ingest(inventory, args.entity_type, args.limit, args.force, issues)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("Image ingestion complete: " + ", ".join(
        f"{key}={value}" for key, value in report["summary"].items() if not isinstance(value, dict)
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
