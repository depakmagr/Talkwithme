import faiss # facebook AI Similarity Search
import numpy as np
import os
import pickle

INDEX_PATH = "vector_store.index"
META_PATH = "metadata.pkl"

def save_vector_store(index, metadata):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

def load_vector_store():
    if not os.path.exists(INDEX_PATH):
        return None, []
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb")as f:
        metadata = pickle.load(f)
    return index, metadata

def search(index, query_vector, k=5):
    D, I = index.search(np.array([query_vector]).astype("float32"), k)
    return I[0]