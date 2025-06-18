import json
import re
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
import time  # for timing phases

from rank_bm25 import BM25Okapi

# Helper: compute SHA-1 hash of a file to detect changes
def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

# Custom embeddings class using SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts: list[str]):
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    def embed_query(self, text: str):
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

# Setup paths and embedding model
embedding_model = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
persist_dir = Path("./db")
hash_file = persist_dir / "posts_hash.txt"
json_path = Path("posts.json")

# Load or Build Index
print("Starting index load/build phase")
start_all = time.perf_counter()
json_hash = sha1_of_file(str(json_path))

if persist_dir.exists() and hash_file.exists() and hash_file.read_text() == json_hash:
    print("Using existing vector database.")
    vector_database = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space":"cosine"}
    )
else:
    print("Rebuilding vector database...")
    t0 = time.perf_counter()
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    post_texts = []
    post_ids = []
    print("Chunking by post...")
    for post_id, post in data.items():
        subj = post.get('subject', '').strip()
        cont = post.get('content', '').strip()
        ia = post.get('instructor_answer', '').strip()
        ea = post.get('endorsed_answer', '').strip()
        full = ' '.join(filter(None, [subj, cont, ia, ea]))
        post_texts.append(full)
        post_ids.append(post_id)
        # Post-level chunking: embed the entire post as one document
        documents.append(Document(
            page_content=full,
            metadata={'post_id': post_id, 'subject': subj, 'idx': 0}
        ))

    print(f"Embedding {len(documents)} chunks...")
    vector_database = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space":"cosine"}
    )
    hash_file.write_text(json_hash)
    t1 = time.perf_counter()
    print(f"Rebuild completed in {t1 - t0:.2f} seconds.")
    # BM25 indexing
    tokenized_corpus = [re.findall(r"[A-Za-z]+|\d+", txt.lower()) for txt in post_texts]
    bm25 = BM25Okapi(tokenized_corpus)

# Ensure BM25 exists if DB reused
if 'bm25' not in locals():
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    post_texts = [' '.join(filter(None, [
        p.get('subject',''), p.get('content',''),
        p.get('instructor_answer',''), p.get('endorsed_answer','')
    ])) for p in data.values()]
    post_ids = list(data.keys())
    tokenized_corpus = [re.findall(r"[A-Za-z]+|\d+", txt.lower()) for txt in post_texts]
    bm25 = BM25Okapi(tokenized_corpus)

print("Index load/build phase complete.")
end_all = time.perf_counter()
print(f"Total load/build time: {end_all - start_all:.2f} seconds.")

# Query & Retrieval (Hybrid with RRF reranking)
print("Starting query phase.")
prep_start = time.perf_counter()
query = input("Enter query: ")
prep_end = time.perf_counter()
print(f"Query input received in {prep_end - prep_start:.2f} seconds.")

# BM25 stage
tokens = re.findall(r"[A-Za-z]+|\d+", query.lower())
bm25_ids = bm25.get_top_n(tokens, post_ids, n=100)

# Semantic stage
sem_results = vector_database.similarity_search_with_score(query, k=100)
sem_ids = [d.metadata['post_id'] for d, _ in sem_results]

# RRF reranking
nrrf_scores = {}
bm25_rank = {pid: i+1 for i, pid in enumerate(bm25_ids)}
sem_rank = {d.metadata['post_id']: i+1 for i, (d, _) in enumerate(sem_results)}
k_rrf = 60
candidates = set(bm25_ids) | set(sem_ids)
for pid in candidates:
    br = bm25_rank.get(pid, len(bm25_ids) + 1)
    sr = sem_rank.get(pid, len(sem_ids) + 1)
    nrrf_scores[pid] = 1.0/(k_rrf + br) + 1.0/(k_rrf + sr)

# Sort and display top 10
sorted_posts = sorted(nrrf_scores.items(), key=lambda x: x[1], reverse=True)[:10]
print("Retrieval complete. Top 10 posts based on RRF fused ranking:")
for idx, (pid, score) in enumerate(sorted_posts, start=1):
    subj = data.get(pid, {}).get('subject', '')
    print(f"{idx}. Post #{pid} — {subj} (RRF score: {score:.6f})")
