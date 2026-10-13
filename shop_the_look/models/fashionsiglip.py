"""Small, validated image-embedding wrapper for Marqo-FashionSigLIP.

The official model requires ``trust_remote_code=True``. The custom code is
loaded only from the reviewed, pinned Hugging Face snapshot. Its model config
normally asks OpenCLIP to resolve ``hf-hub:Marqo/marqo-fashionSigLIP`` without a
revision. This wrapper downloads the necessary OpenCLIP files at the same
revision and replaces that field with OpenCLIP's documented ``local-dir:``
form before constructing the model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

MODEL_ID = "Marqo/marqo-fashionSigLIP"
MODEL_REVISION = "c56244cc94f92419e8369fa71efdaf403b124ce8"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE = PROJECT_ROOT / ".hf_cache"
SNAPSHOT_FILES = (
    "config.json",
    "marqo_fashionSigLIP.py",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "spiece.model",
    "model.safetensors",
    "open_clip_config.json",
    "open_clip_model.safetensors",
)


def select_device(torch_module: Any, requested: str = "cpu") -> str:
    """Validate a configured device; CPU is the conservative default."""
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        mps = getattr(getattr(torch_module, "backends", None), "mps", None)
        if mps is None or not callable(getattr(mps, "is_available", None)) or not mps.is_available():
            raise RuntimeError("MPS was requested but this PyTorch build reports MPS unavailable; use device='cpu'.")
        return "mps"
    raise ValueError("device must be 'cpu' or 'mps'; CPU is the supported default for this smoke test")


def validate_and_normalize(embeddings: Any, expected_count: int, expected_dimension: int | None = None) -> np.ndarray:
    """Return finite float32 2D unit vectors or raise an actionable error."""
    if expected_count <= 0:
        raise ValueError("expected_count must be greater than zero")
    try:
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        array = np.asarray(embeddings)
    except Exception as exc:
        raise ValueError(f"Embedding output could not be converted to a numeric array: {exc}") from exc
    if array.ndim != 2 or array.shape[0] != expected_count or array.shape[1] == 0:
        raise ValueError(f"Expected embedding shape ({expected_count}, nonzero_dimension), got {array.shape}")
    if expected_dimension is not None and array.shape[1] != expected_dimension:
        raise ValueError(f"Expected embedding dimension {expected_dimension}, got {array.shape[1]}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Embedding output must be numeric, got dtype {array.dtype}")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Embedding output contains NaN or infinite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Embedding output contains a zero or invalid-norm vector")
    normalized = array / norms
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized embedding output contains NaN or infinite values")
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_snapshot_files(snapshot_path: Path) -> dict[str, dict[str, Any]]:
    required = ("config.json", "marqo_fashionSigLIP.py", "model.safetensors", "open_clip_config.json", "open_clip_model.safetensors")
    missing = [name for name in required if not (snapshot_path / name).is_file()]
    if missing:
        raise RuntimeError(f"Pinned model snapshot is missing required files: {', '.join(missing)}")
    inventory: dict[str, dict[str, Any]] = {}
    for name in required:
        path = snapshot_path / name
        inventory[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return inventory


class FashionSigLIP:
    """Load the reviewed Marqo checkpoint and embed Pillow images or paths."""

    def __init__(self, device: str = "cpu", revision: str = MODEL_REVISION) -> None:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoConfig, AutoModel, AutoProcessor
        except ImportError as exc:
            missing = getattr(exc, "name", None) or "model dependency"
            raise RuntimeError(
                f"Missing model dependency {missing!r}. Install the pinned model requirements "
                "inside the project virtual environment."
            ) from exc
        if revision != MODEL_REVISION:
            raise ValueError(f"Only reviewed model revision {MODEL_REVISION} is allowed; got {revision}")
        self.device = select_device(torch, device)
        try:
            snapshot_path = Path(snapshot_download(
                repo_id=MODEL_ID,
                revision=revision,
                cache_dir=str(MODEL_CACHE),
                allow_patterns=list(SNAPSHOT_FILES),
            ))
            snapshot_path = snapshot_path.resolve()
            self.snapshot_path = snapshot_path
            self.model_files = _verify_snapshot_files(snapshot_path)
            open_clip_config = json.loads((snapshot_path / "open_clip_config.json").read_text(encoding="utf-8"))
            self.expected_embedding_dimension = int(open_clip_config["model_cfg"]["embed_dim"])
            config = AutoConfig.from_pretrained(
                str(snapshot_path), trust_remote_code=True, local_files_only=True,
            )
            config.open_clip_model_name = f"local-dir:{snapshot_path}"
            self.model = AutoModel.from_pretrained(
                str(snapshot_path), config=config, trust_remote_code=True,
                use_safetensors=True, local_files_only=True, low_cpu_mem_usage=False,
            )
            self.processor = AutoProcessor.from_pretrained(
                str(snapshot_path), trust_remote_code=True, local_files_only=True,
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {MODEL_ID}@{revision} on {self.device}. Check the pinned local snapshot, "
                "installed versions, and available memory/disk space."
            ) from exc
        self._torch = torch

    @staticmethod
    def _open_image(source: Image.Image | str | Path) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        try:
            with Image.open(source) as image:
                image.load()
                return image.convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ValueError(f"Cannot read image input {source!s}: {exc}") from exc

    def embed_images(
        self,
        images: Sequence[Image.Image | str | Path] | Iterable[Image.Image | str | Path],
        batch_size: int = 8,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        sources = list(images)
        if not sources:
            raise ValueError("images must contain at least one Pillow image or image path")
        opened = [self._open_image(image) for image in sources]
        outputs: list[np.ndarray] = []
        try:
            with self._torch.inference_mode():
                for start in range(0, len(opened), batch_size):
                    batch = opened[start:start + batch_size]
                    processed = self.processor(images=batch, return_tensors="pt")
                    if "pixel_values" not in processed:
                        raise RuntimeError("Official processor returned no 'pixel_values' for image inputs")
                    pixel_values = processed["pixel_values"].to(self.device)
                    features = self.model.get_image_features(pixel_values, normalize=True)
                    vectors = validate_and_normalize(
                        features, expected_count=len(batch),
                        expected_dimension=self.expected_embedding_dimension,
                    )
                    outputs.append(vectors)
        except Exception as exc:
            if isinstance(exc, (ValueError, RuntimeError)) and str(exc).startswith(("Expected embedding", "Embedding output", "Normalized embedding", "Official processor")):
                raise
            raise RuntimeError(f"FashionSigLIP image inference failed on {self.device}: {exc}") from exc
        result = np.concatenate(outputs, axis=0)
        if result.shape[0] != len(sources):
            raise RuntimeError(f"Model returned {result.shape[0]} vectors for {len(sources)} images")
        return result
