from __future__ import annotations

import hashlib
import html
from pathlib import Path

import streamlit as st
from PIL import Image

from shop_the_look.frontend import (
    DEFAULT_MANIFEST,
    LiveIndexSearchWorker,
    ROOT,
    decode_upload,
    load_product_image_map,
    retrieve_upload,
)
from shop_the_look.models.fashionsiglip import FashionSigLIP


st.set_page_config(page_title="SHOP THE LOOK", page_icon="👗", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1280px; padding-top: 2.2rem;}
.hero {padding: 1.3rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#fff4f0,#f5f2ff); margin-bottom: 1.2rem;}
.hero h1 {letter-spacing: .08em; margin: 0; font-size: 2.2rem;}
.hero p {color: #5b5864; margin: .45rem 0 0; font-size: 1.05rem;}
.result-label {font-size: .8rem; color: #716d77; margin-bottom: .2rem;}
.product-id {font-family: monospace; font-size: .8rem; overflow-wrap: anywhere; color: #514d58;}
.score {font-size: .9rem; font-weight: 600;}
</style>
<div class="hero"><h1>SHOP THE LOOK</h1><p>Upload a fashion image to discover visually similar products.</p></div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the FashionSigLIP model…")
def get_model() -> FashionSigLIP:
    return FashionSigLIP(device="cpu")


@st.cache_resource(show_spinner="Loading the saved catalog index…")
def get_search_worker() -> LiveIndexSearchWorker:
    return LiveIndexSearchWorker()


@st.cache_data(show_spinner=False, max_entries=512)
def get_product_image(path_string: str) -> Image.Image | None:
    try:
        with Image.open(path_string) as image:
            image.load()
            return image.convert("RGB").copy()
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False, max_entries=4)
def get_product_image_map(manifest_path: str, root_path: str) -> dict[str, str]:
    return {product_id: str(path) for product_id, path in
            load_product_image_map(Path(manifest_path), Path(root_path)).items()}


uploaded = st.file_uploader("Choose a fashion image", type=["jpg", "jpeg", "png"],
                            help="JPG, JPEG, or PNG · maximum 15 MB and 40 megapixels")
query_image = None
if uploaded is not None:
    try:
        upload_bytes = uploaded.getvalue()
        upload_digest = hashlib.sha256(upload_bytes).hexdigest()
        if st.session_state.get("shop_the_look_upload_digest") != upload_digest:
            for key in ("shop_the_look_results", "shop_the_look_image_map",
                        "shop_the_look_query_info", "shop_the_look_mode", "shop_the_look_upload_name"):
                st.session_state.pop(key, None)
            st.session_state["shop_the_look_upload_digest"] = upload_digest
        query_image = decode_upload(upload_bytes, uploaded.name)
        left, right = st.columns([1, 1.5], gap="large")
        with left:
            st.image(query_image, caption=f"Uploaded image · {query_image.width} × {query_image.height}", width="stretch")
        with right:
            mode = st.radio("Retrieval mode", ["Whole-scene", "Multi-crop"], horizontal=True)
            top_k = st.selectbox("Number of products", [5, 10, 20], index=1)
            run = st.button("Find Similar Products", type="primary", width="stretch")
    except ValueError as exc:
        st.error(str(exc))
        run = False
else:
    for key in ("shop_the_look_results", "shop_the_look_image_map", "shop_the_look_query_info",
                "shop_the_look_mode", "shop_the_look_upload_name", "shop_the_look_upload_digest"):
        st.session_state.pop(key, None)
    st.info("Upload a JPG, JPEG, or PNG image to get started.")
    run = False

if run and query_image is not None:
    try:
        with st.spinner("Embedding your image and searching the catalog…"):
            model = get_model()
            engine = get_search_worker()
            product_images = get_product_image_map(str(DEFAULT_MANIFEST), str(ROOT))
            results, query_info = retrieve_upload(query_image, mode, model, engine, int(top_k))
        st.session_state["shop_the_look_results"] = results
        st.session_state["shop_the_look_image_map"] = {key: str(value) for key, value in product_images.items()}
        st.session_state["shop_the_look_query_info"] = query_info
        st.session_state["shop_the_look_mode"] = mode
        st.session_state["shop_the_look_upload_name"] = uploaded.name
    except Exception as exc:
        message = str(exc).replace(str(ROOT), "<project>")
        st.error(f"Could not complete image search: {message}")

if "shop_the_look_results" in st.session_state:
    results = st.session_state["shop_the_look_results"]
    product_images = st.session_state.get("shop_the_look_image_map", {})
    query_info = st.session_state.get("shop_the_look_query_info", {})
    st.subheader("Similar products")
    st.caption(f"{st.session_state.get('shop_the_look_mode', 'Whole-scene')} search · {len(results)} results")
    if query_info.get("query_source") != "uploaded_image" or query_info.get("dimension") != 768:
        st.error("Search result metadata failed validation; run the search again.")
    elif not results:
        st.info("No matching products were returned.")
    else:
        columns = st.columns(5, gap="medium")
        for position, result in enumerate(results):
            with columns[position % len(columns)]:
                st.markdown(f'<div class="result-label">RANK {result["rank"]}</div>', unsafe_allow_html=True)
                image_path = product_images.get(result["product_id"])
                product_image = get_product_image(image_path) if image_path else None
                if product_image is None:
                    st.markdown("<div style='height:220px;display:grid;place-items:center;background:#f3f3f5;border-radius:10px;color:#777'>Image unavailable</div>", unsafe_allow_html=True)
                else:
                    st.image(product_image, width="stretch")
                st.markdown(f'<div class="product-id">{html.escape(str(result["product_id"]))}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="score">Cosine similarity&nbsp; {result["cosine_similarity"]:.4f}</div>', unsafe_allow_html=True)
