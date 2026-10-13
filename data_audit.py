#!/usr/bin/env python3
"""
Shop-the-Look dataset audit.

Uses only the Python standard library for JSONL parsing and HTTP checks.
Pillow is optional; when installed, it is used to verify that downloaded
responses are decodable images (not merely responses with image/* headers).

Examples:
  python data_audit.py --zip dataset.zip
  python data_audit.py --zip dataset.zip --image-check sample --sample-size 100
  python data_audit.py --zip dataset.zip --image-check all --workers 8
  python data_audit.py --zip dataset.zip --image-check none

The assignment documents this Pinterest URL pattern for scene images:
http://i.pinimg.com/400x/{id[0:2]}/{id[2:4]}/{id[4:6]}/{id}.jpg

Product IDs are tested with the same pattern as a hypothesis; the assignment
does not explicitly guarantee product-image URLs use it.
"""

from __future__ import annotations
import ssl
import certifi

import argparse
import concurrent.futures
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from PIL import Image, UnidentifiedImageError
    from io import BytesIO
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def image_url(identifier: str, scheme: str = "https") -> str:
    """Build the Pinterest CDN URL from a 32-character identifier."""
    return (
        f"{scheme}://i.pinimg.com/400x/"
        f"{identifier[0:2]}/{identifier[2:4]}/{identifier[4:6]}/"
        f"{identifier}.jpg"
    )


def read_jsonl_from_zip(zf: zipfile.ZipFile, member: str) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    with zf.open(member) as raw:
        for line_no, raw_line in enumerate(raw, 1):
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
                else:
                    malformed += 1
                    print(f"WARNING: {member}:{line_no}: JSON value is not an object", file=sys.stderr)
            except json.JSONDecodeError as exc:
                malformed += 1
                print(f"WARNING: {member}:{line_no}: malformed JSON ({exc})", file=sys.stderr)
    return rows, malformed


def find_member(zf: zipfile.ZipFile, requested: str) -> str:
    matches = [name for name in zf.namelist() if name == requested or name.endswith("/" + requested)]
    if not matches:
        raise FileNotFoundError(f"Could not find {requested!r} in ZIP. Members: {zf.namelist()}")
    if len(matches) > 1:
        raise ValueError(f"Multiple ZIP members match {requested!r}: {matches}")
    return matches[0]


def image_check(task: tuple[str, str, float]) -> dict[str, Any]:
    identifier, kind, timeout = task
    url = image_url(identifier)
    result: dict[str, Any] = {
        "kind": kind, "id": identifier, "url": url, "status": None,
        "content_type": None, "bytes": 0, "result": None, "detail": None,
    }
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ShopTheLookDatasetAudit/1.0)"},
        method="GET",
    )
    try:
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        with urllib.request.urlopen(
            request,
            timeout=timeout,
            context=ssl_context
        ) as response:
            body = response.read(8 * 1024 * 1024 + 1)
            result["status"] = getattr(response, "status", None)
            result["content_type"] = response.headers.get("Content-Type", "").split(";")[0].lower()
            result["bytes"] = len(body)
            if len(body) > 8 * 1024 * 1024:
                result["result"] = "oversized_response"
                result["detail"] = "Response exceeded 8 MiB audit safety limit"
                return result
            if not result["content_type"].startswith("image/"):
                result["result"] = "non_image_response"
                result["detail"] = f"Content-Type was {result['content_type'] or '(missing)'}"
                return result
            if not body:
                result["result"] = "empty_response"
                return result
            if PIL_AVAILABLE:
                try:
                    with Image.open(BytesIO(body)) as img:
                        img.verify()
                    result["result"] = "success_valid_image"
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    result["result"] = "corrupt_image"
                    result["detail"] = str(exc)[:250]
            else:
                # Without Pillow, report the HTTP/content-type check separately.
                result["result"] = "success_header_only"
                result["detail"] = "Install Pillow to verify image decoding"
    except urllib.error.HTTPError as exc:
        result["status"] = exc.code
        result["result"] = "http_failure"
        result["detail"] = str(exc.reason)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result["result"] = "connection_failure"
        result["detail"] = str(getattr(exc, "reason", exc))[:250]
    except Exception as exc:
        result["result"] = "unexpected_failure"
        result["detail"] = repr(exc)[:250]
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Shop-the-Look JSONL data and image URLs.")
    parser.add_argument("--zip", dest="zip_path", default="dataset.zip", help="Path to supplied dataset ZIP")
    parser.add_argument("--catalog-member", default="product_catelog.jsonl")
    parser.add_argument("--validation-member", default="validation.jsonl")
    parser.add_argument("--out-dir", default="audit_output", help="Directory for reports")
    parser.add_argument("--image-check", choices=("sample", "all", "none"), default="sample")
    parser.add_argument("--sample-size", type=int, default=100, help="Unique IDs checked per type in sample mode")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheme", choices=("https", "http"), default="https")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"ERROR: ZIP not found: {zip_path}", file=sys.stderr)
        return 2

    with zipfile.ZipFile(zip_path) as zf:
        catalog_member = find_member(zf, args.catalog_member)
        validation_member = find_member(zf, args.validation_member)
        catalog_rows, catalog_malformed = read_jsonl_from_zip(zf, catalog_member)
        validation_rows, validation_malformed = read_jsonl_from_zip(zf, validation_member)

    catalog_ids = [str(r.get("product", "")).strip() for r in catalog_rows]
    validation_products = [str(r.get("product", "")).strip() for r in validation_rows]
    validation_scenes = [str(r.get("scene", "")).strip() for r in validation_rows]

    blank_catalog_ids = sum(not x for x in catalog_ids)
    blank_validation_products = sum(not x for x in validation_products)
    blank_validation_scenes = sum(not x for x in validation_scenes)

    catalog_counts = Counter(x for x in catalog_ids if x)
    validation_product_counts = Counter(x for x in validation_products if x)
    scene_counts = Counter(x for x in validation_scenes if x)
    pair_counts = Counter(
        (str(r.get("scene", "")).strip(), str(r.get("product", "")).strip())
        for r in validation_rows
        if str(r.get("scene", "")).strip() and str(r.get("product", "")).strip()
    )

    catalog_unique = set(catalog_counts)
    validation_unique_products = set(validation_product_counts)
    validation_unique_scenes = set(scene_counts)
    validation_products_in_catalog = validation_unique_products & catalog_unique
    validation_products_missing_from_catalog = validation_unique_products - catalog_unique
    pair_products_missing_from_catalog = {
        product for _, product in pair_counts if product and product not in catalog_unique
    }
    duplicate_catalog_rows = sum(count - 1 for count in catalog_counts.values() if count > 1)
    duplicate_validation_pairs = sum(count - 1 for count in pair_counts.values() if count > 1)

    # Detailed association records; preserve source rows and identify catalog coverage.
    associations = []
    for row in validation_rows:
        product = str(row.get("product", "")).strip()
        scene = str(row.get("scene", "")).strip()
        associations.append({
            "scene": scene or None,
            "product": product or None,
            "product_in_catalog": bool(product and product in catalog_unique),
            "scene_url": image_url(scene, args.scheme) if scene else None,
            "product_url_hypothesis": image_url(product, args.scheme) if product else None,
            "duplicate_pair_count": pair_counts.get((scene, product), 0),
        })
    write_json(out_dir / "validation_associations.json", associations)

    # IDs for image URL checks. Scene URLs are assignment-specified; product URLs are a hypothesis.
    checks: list[dict[str, Any]] = []
    check_summary: dict[str, Any] = {
        "mode": args.image_check,
        "pillow_available": PIL_AVAILABLE,
        "product_url_pattern_status": "hypothesis; not specified by assignment",
        "scene_url_pattern_status": "specified by assignment",
        "scheme": args.scheme,
    }
    if args.image_check != "none":
        rng = random.Random(args.seed)
        ids_by_type = {
            "product": sorted(catalog_unique),
            "scene": sorted(validation_unique_scenes),
        }
        tasks = []
        for kind, ids in ids_by_type.items():
            if args.image_check == "sample":
                n = min(max(args.sample_size, 0), len(ids))
                chosen = sorted(rng.sample(ids, n)) if n else []
            else:
                chosen = ids
            tasks.extend((identifier, kind, args.timeout) for identifier in chosen)

        print(f"Checking {len(tasks)} image URLs ({args.image_check} mode, {args.workers} workers)...", flush=True)
        started = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for result in executor.map(image_check, tasks):
                checks.append(result)
        elapsed = round(time.time() - started, 2)

        # Apply requested scheme in report URLs, because image_check's URL helper defaults to HTTPS.
        for result in checks:
            result["url"] = image_url(result["id"], args.scheme)

        by_type = {}
        for kind in ("product", "scene"):
            typed = [r for r in checks if r["kind"] == kind]
            by_result = Counter(r["result"] for r in typed)
            by_type[kind] = {
                "checked": len(typed),
                "success": by_result.get("success_valid_image", 0) + by_result.get("success_header_only", 0),
                "valid_decoded_image": by_result.get("success_valid_image", 0),
                "header_only_success_no_pillow": by_result.get("success_header_only", 0),
                "http_failures": by_result.get("http_failure", 0),
                "connection_failures": by_result.get("connection_failure", 0),
                "non_image_responses": by_result.get("non_image_response", 0),
                "corrupt_images": by_result.get("corrupt_image", 0),
                "empty_responses": by_result.get("empty_response", 0),
                "other_failures": sum(v for k, v in by_result.items() if k not in {
                    "success_valid_image", "success_header_only", "http_failure",
                    "connection_failure", "non_image_response", "corrupt_image", "empty_response"
                }),
                "results": dict(by_result),
            }
        check_summary["elapsed_seconds"] = elapsed
        check_summary["by_type"] = by_type
        write_json(out_dir / "image_check_details.json", checks)

    report = {
        "dataset_zip": str(zip_path),
        "catalog_member": catalog_member,
        "validation_member": validation_member,
        "counts": {
            "catalog_rows": len(catalog_rows),
            "catalog_unique_nonblank_product_ids": len(catalog_unique),
            "catalog_duplicate_rows_beyond_first_occurrence": duplicate_catalog_rows,
            "catalog_ids_with_duplicates": sum(1 for n in catalog_counts.values() if n > 1),
            "catalog_blank_product_ids": blank_catalog_ids,
            "validation_rows": len(validation_rows),
            "validation_unique_products_nonblank": len(validation_unique_products),
            "validation_unique_scenes_nonblank": len(validation_unique_scenes),
            "validation_unique_scene_product_pairs": len(pair_counts),
            "validation_duplicate_pairs_beyond_first_occurrence": duplicate_validation_pairs,
            "validation_blank_product_ids": blank_validation_products,
            "validation_blank_scene_ids": blank_validation_scenes,
            "validation_unique_products_found_in_catalog": len(validation_products_in_catalog),
            "validation_unique_products_missing_from_catalog": len(validation_products_missing_from_catalog),
            "validation_rows_with_product_in_catalog": sum(
                1 for r in validation_rows
                if str(r.get("product", "")).strip() in catalog_unique
            ),
            "validation_rows_with_product_missing_from_catalog": sum(
                1 for r in validation_rows
                if str(r.get("product", "")).strip()
                and str(r.get("product", "")).strip() not in catalog_unique
            ),
            "validation_pairs_with_product_missing_from_catalog": sum(
                count for (scene, product), count in pair_counts.items()
                if product and product not in catalog_unique
            ),
            "validation_scenes_with_multiple_products": sum(
                1 for scene in validation_unique_scenes
                if len({p for s, p in pair_counts if s == scene and p}) > 1
            ),
            "validation_products_associated_with_multiple_scenes": sum(
                1 for product in validation_unique_products
                if len({s for s, p in pair_counts if p == product and s}) > 1
            ),
            "catalog_malformed_json_lines": catalog_malformed,
            "validation_malformed_json_lines": validation_malformed,
        },
        "validation_product_coverage": {
            "unique_validation_products": len(validation_unique_products),
            "unique_validation_products_in_catalog": len(validation_products_in_catalog),
            "unique_validation_products_missing_from_catalog": len(validation_products_missing_from_catalog),
            "unique_product_coverage_fraction": (
                len(validation_products_in_catalog) / len(validation_unique_products)
                if validation_unique_products else None
            ),
            "missing_product_ids": sorted(validation_products_missing_from_catalog),
            "missing_product_ids_at_pair_level": sorted(pair_products_missing_from_catalog),
        },
        "image_checks": check_summary,
        "notes": [
            "Catalog source member is named product_catelog.jsonl in the supplied ZIP.",
            "Duplicate catalog rows are counted as occurrences beyond the first for each nonblank product ID.",
            "Validation coverage is computed both at unique-product and row/pair level.",
            "Image URL checks are sampled by default; sample counts are not full-dataset success rates.",
            "The assignment specifies the URL construction for scene IDs. Product-image URL use is an empirical hypothesis.",
            "Without Pillow, image checks verify HTTP success and image/* Content-Type only; they cannot confirm image decoding.",
        ],
    }
    write_json(out_dir / "audit_report.json", report)

    # Plain-text summary for quick terminal review.
    c = report["counts"]
    lines = [
        "SHOP-THE-LOOK DATA AUDIT",
        "=" * 28,
        f"Catalog rows: {c['catalog_rows']:,}",
        f"Unique nonblank catalog product IDs: {c['catalog_unique_nonblank_product_ids']:,}",
        f"Duplicate catalog rows beyond first occurrence: {c['catalog_duplicate_rows_beyond_first_occurrence']:,}",
        f"Catalog IDs occurring more than once: {c['catalog_ids_with_duplicates']:,}",
        f"Blank catalog product IDs: {c['catalog_blank_product_ids']:,}",
        "",
        f"Validation rows: {c['validation_rows']:,}",
        f"Unique validation scenes: {c['validation_unique_scenes_nonblank']:,}",
        f"Unique validation products: {c['validation_unique_products_nonblank']:,}",
        f"Unique scene-product pairs: {c['validation_unique_scene_product_pairs']:,}",
        f"Duplicate validation pairs beyond first occurrence: {c['validation_duplicate_pairs_beyond_first_occurrence']:,}",
        f"Validation rows whose product is in catalog: {c['validation_rows_with_product_in_catalog']:,}",
        f"Validation rows whose product is missing from catalog: {c['validation_rows_with_product_missing_from_catalog']:,}",
        f"Unique validation products missing from catalog: {c['validation_unique_products_missing_from_catalog']:,}",
        f"Scenes associated with multiple products: {c['validation_scenes_with_multiple_products']:,}",
        f"Products associated with multiple scenes: {c['validation_products_associated_with_multiple_scenes']:,}",
        f"Malformed catalog JSONL lines: {c['catalog_malformed_json_lines']:,}",
        f"Malformed validation JSONL lines: {c['validation_malformed_json_lines']:,}",
        "",
        f"Image check mode: {args.image_check}",
        f"Pillow available: {PIL_AVAILABLE}",
    ]
    if args.image_check != "none":
        lines.append(f"Image check elapsed: {check_summary.get('elapsed_seconds', 0)} seconds")
        for kind, values in check_summary.get("by_type", {}).items():
            lines.append(
                f"{kind.title()} images checked: {values['checked']:,}; "
                f"success: {values['success']:,}; HTTP failures: {values['http_failures']:,}; "
                f"connection failures: {values['connection_failures']:,}; "
                f"non-image: {values['non_image_responses']:,}; corrupt: {values['corrupt_images']:,}"
            )
    lines.extend([
        "",
        f"Detailed report: {out_dir / 'audit_report.json'}",
        f"Validation associations: {out_dir / 'validation_associations.json'}",
    ])
    if args.image_check != "none":
        lines.append(f"Image check details: {out_dir / 'image_check_details.json'}")
    summary_path = out_dir / "audit_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
