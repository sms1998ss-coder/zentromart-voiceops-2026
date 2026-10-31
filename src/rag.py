"""Semantic Knowledge Base Retriever using SentenceTransformers and FAISS.

Indexes ZentroMart's Markdown policy documents (`kb/*.md`), preserving section headers
and prioritizing active policies over superseded ones.
"""
import os
import glob
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class DocumentRetriever:
    def __init__(self, kb_path: str):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = []
        self.index = None
        self.load_documents(kb_path)

    def load_documents(self, kb_path: str):
        if not os.path.exists(kb_path):
            print(f"Warning: {kb_path} not found. Please ensure the 'kb' folder is in the project root.")
            return

        files = sorted(glob.glob(os.path.join(kb_path, "*.md")))
        for f in files:
            filename = os.path.basename(f)
            # Prioritize pol_returns_v4.md and skip superseded v3 policy per Capstone Brief
            if "superseded" in filename.lower():
                continue

            with open(f, "r", encoding="utf-8") as file:
                # Chunking by double newline keeps Markdown sections and [SECTION-ID] headers intact
                sections = file.read().split("\n\n")
                for sec in sections:
                    if sec.strip():
                        self.chunks.append(f"[Source: {filename}] {sec.strip()}")

        if self.chunks:
            embeddings = self.model.encode(self.chunks)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(np.array(embeddings, dtype=np.float32))

    def retrieve(self, query: str, top_k: int = 3):
        if not self.index or not query or not query.strip():
            return []
        query_emb = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_emb, dtype=np.float32), top_k)
        return [self.chunks[i] for i in indices[0] if 0 <= i < len(self.chunks)]


def build_index(kb_path: str) -> DocumentRetriever:
    return DocumentRetriever(kb_path)
