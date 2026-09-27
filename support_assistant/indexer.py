"""
Zepto Vector Indexer: Embeds policy corpus using all-MiniLM-L6-v2 and stores in ChromaDB
"""
import os
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = "support_assistant/docs"
CHROMA_DIR = "support_assistant/chroma_db"
COLLECTION_NAME = "zepto_policies"

def build_index():
    print("=== Initializing ChromaDB & Embedding Model (all-MiniLM-L6-v2) ===")
    os.makedirs(CHROMA_DIR, exist_ok=True)
    
    # Initialize persistent ChromaDB client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Use sentence-transformers all-MiniLM-L6-v2 embedding function
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create or recreate collection
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
        
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Load and index all 8 documents
    doc_files = sorted([f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")])
    ids = []
    documents = []
    metadatas = []
    
    for filename in doc_files:
        doc_id = filename.replace(".txt", "")
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        ids.append(doc_id)
        documents.append(text)
        metadatas.append({"source": filename, "doc_id": doc_id})
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"Successfully embedded and indexed {len(ids)} policy documents into collection '{COLLECTION_NAME}'.")
    return collection

if __name__ == "__main__":
    build_index()
