import json
import re
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
import sys
import time  # for timing phases

# Helper: compute SHA-1 hash of a file to detect changes
def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# Embedding Model Wrapper
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()
    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_tensor=False).tolist()

# Text Cleaning and Chunking
def clean_text(text: str) -> str:
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    return text.replace('`', '')

def split_into_sentences(text: str) -> list[str]:
    sentences = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = re.split(r'(?<=[\.\?!])\s+', line)
        sentences.extend([p.strip() for p in parts if p.strip()])
    return sentences

# Sliding Window Chunking
def make_sliding_chunks(sentences: list[str]) -> list[str]:
    chunks = []
    n = len(sentences)
    for w in (2, 3):
        for i in range(n - w + 1):
            window = sentences[i:i+w]
            chunks.append(' '.join(window))
    return chunks

# Setup paths and embedding model
embedding_model = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
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
        
        # Post‐level chunking: embed the entire post as one document
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

# If DB existed, still build BM25
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

# Query & Retrieval (Hybrid)
print("Starting query phase.")
prep_start = time.perf_counter()
query = input("Enter query: ")
prep_end = time.perf_counter()
print(f"Query input received in {prep_end - prep_start:.2f} seconds.")

# BM25 stage
tokens = re.findall(r"[A-Za-z]+|\d+", query.lower())
bm25_ids = bm25.get_top_n(tokens, post_ids, n=100)  # increase BM25 candidates to 100 for broader coverage
bm25_set = set(bm25_ids)

# Semantic stage
results = vector_database.similarity_search_with_score(query, k=100)
results = [(d, dist) for d, dist in results if d.metadata['post_id'] in bm25_set]

# Score & output (max chunk similarity per post)
post_scores = {}
for d, dist in results:
    sim = 1.0 - dist
    pid = d.metadata['post_id']
    subj = d.metadata['subject']
    # track the maximum similarity across chunks
    if pid not in post_scores:
        post_scores[pid] = {'subject': subj, 'score': sim}
    else:
        post_scores[pid]['score'] = max(post_scores[pid]['score'], sim)
# sort by highest single-chunk similarity
top_posts = sorted(
    post_scores.items(),
    key=lambda x: x[1]['score'],
    reverse=True
)
print("Retrieval complete. Top 10 posts based on max chunk similarity:")
for idx, (pid, info) in enumerate(top_posts[:10], start=1):
    print(f"{idx}. Post #{pid} — {info['subject']} (score: {info['score']:.4f})")
