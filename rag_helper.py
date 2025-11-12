# rag_helper.py
import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleRAG:
    def __init__(self, csv_paths):
        self.docs = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_texts = []

        # Load and preprocess CSVs
        for path in csv_paths:
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                text = " ".join([f"{col}: {row[col]}" for col in df.columns if not pd.isna(row[col])])
                self.docs.append(text)

        if not self.docs:
            print(" No documents loaded for RAG.")
            return

        self.doc_texts = self.vectorizer.fit_transform(self.docs)
        print(f" RAG initialized with {len(self.docs)} text chunks.")

    def retrieve(self, query, top_k=3):
        """Find top_k most similar rows for a given query."""
        if not self.docs:
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_texts).flatten()
        top_indices = sims.argsort()[-top_k:][::-1]

        return [self.docs[i] for i in top_indices]

    def build_context(self, query, top_k=3):
        """Return the retrieved context block as a formatted string."""
        results = self.retrieve(query, top_k)
        if not results:
            return "No relevant housing or applicant data found."
        return "\n\n".join([f"- {r}" for r in results])
