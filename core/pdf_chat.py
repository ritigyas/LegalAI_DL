from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

chunks = []
index = None
import re

chunks = []

def split_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def build_index(text):
    global chunks
    chunks = split_text(text)

def search_pdf(query):
    query_words = set(re.findall(r'\w+', query.lower()))

    scored = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        score = len(query_words & chunk_words)

        if score > 0:
            scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    return " ".join([c for c, _ in scored[:3]])