from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Global storage
chunks = []
index = None

# 🔹 Step 1: Split text into chunks
def split_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# 🔹 Step 2: Build FAISS index
def build_index(text):
    global chunks, index

    # Split into chunks
    chunks = split_text(text)

    # Convert chunks to embeddings
    embeddings = model.encode(chunks)

    # Convert to numpy array (float32 required for FAISS)
    embeddings = np.array(embeddings).astype('float32')

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to index
    index.add(embeddings)

# 🔹 Step 3: Search function
def search_pdf(query, top_k=3):
    global index, chunks

    if index is None:
        return "Index not built. Please call build_index() first."

    # Convert query to embedding
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype('float32')

    # Search in FAISS index
    distances, indices = index.search(query_vector, top_k)

    # Fetch top results
    results = [chunks[i] for i in indices[0]]

    return " ".join(results)