from dotenv import load_dotenv

import json
import re
import hashlib
from pathlib import Path
import time

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Helper: compute SHA-1 hash of a file to detect changes
def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# Text Cleaning (unchanged)
def clean_text(text: str) -> str:
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    return text.replace('`', '')

# Setup paths and embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
persist_dir = Path("./db")
hash_file = persist_dir / "posts_hash.txt"
json_path = Path("posts.json")

# Load or Build Index
print("Starting index load/build phase")
start_all = time.perf_counter()
json_hash = sha1_of_file(str(json_path))
print(f"Computed SHA-1 hash for posts.json: {json_hash[:8]}...")

if persist_dir.exists() and hash_file.exists() and hash_file.read_text() == json_hash:
    print("Using existing vector database.")
    vector_database = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
else:
    print("Rebuilding vector database...")
    t0 = time.perf_counter()
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("Total posts in JSON:", len(data))
    ids = sorted(int(pid) for pid in data.keys())
    missing = [i for i in range(1, max(ids)+1) if i not in ids]
    print(f"Missing IDs: {missing[:10]} … (total gaps: {len(missing)})")


    documents = []
    print("Chunking by post...")
    for post_id, post in data.items():
        subj = post.get('subject', '').strip()
        cont = post.get('content', '').strip()
        ia = post.get('instructor_answer', '').strip()
        ea = post.get('endorsed_answer', '').strip()
        full = ' '.join(filter(None, [subj, cont, ia, ea]))
        documents.append(Document(
            page_content=full,
            metadata={'post_id': post_id, 'subject': subj, 'idx': 0}
        ))

    print(f"Embedding {len(documents)} documents...")
    vector_database = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"}
    )
    hash_file.write_text(json_hash)
    t1 = time.perf_counter()
    print(f"Rebuild completed in {t1 - t0:.2f} seconds.")

print("Index load/build phase complete.")
end_all = time.perf_counter()
print(f"Total load/build time: {end_all - start_all:.2f} seconds.")

# Query & Retrieval (Semantic Only)
print("Starting query phase.")
prep_start = time.perf_counter()
query = input("Enter query: ")
prep_end = time.perf_counter()
print(f"Query input received in {prep_end - prep_start:.2f} seconds.")

results = vector_database.similarity_search_with_score(query, k=10)

print("Retrieval complete. Top 10 posts based on semantic similarity:")
for idx, (doc, dist) in enumerate(results[:10], start=1):
    pid = doc.metadata['post_id']
    subj = doc.metadata.get('subject', '')
    sim = 1.0 - dist
    print(f"{idx}. Post #{pid} — {subj} (score: {sim:.4f})")
