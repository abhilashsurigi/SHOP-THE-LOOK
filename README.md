# SHOP THE LOOK

An image-to-product retrieval demo using the saved Marqo FashionSigLIP embeddings and exact FAISS product index.

## Run the Streamlit app

From the repository root:

```bash
.venv/bin/python -m streamlit run app.py
```

The app accepts new JPG, JPEG, and PNG uploads (up to 15 MB and 40 megapixels). It embeds the uploaded image with the pinned FashionSigLIP model, then searches the existing product index. Whole-scene and deterministic multi-crop modes are available. Product images are read from the local cache; the app does not fetch images over the network.

If Streamlit is not installed in the project environment, install the app dependency with:

```bash
.venv/bin/python -m pip install -r requirements-app.txt
```

The app loads model and index resources once per Streamlit process. The first model load can take a little time. It does not rebuild the product index or recompute product embeddings.
