"""Exact top-k product retrieval over saved FashionSigLIP embeddings."""

from .faiss_index import FlatProductIndex, RetrievalEngine, build_product_index, discover_embedding_bundle

__all__ = ["FlatProductIndex", "RetrievalEngine", "build_product_index", "discover_embedding_bundle"]
