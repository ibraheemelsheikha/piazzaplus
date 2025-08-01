import json
import re
import hashlib
from pathlib import Path
import time
import os
import argparse

import nltk
nltk.download('punkt_tab')
from langchain.text_splitter import NLTKTextSplitter

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

import base64
import httpx
import requests
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv

# Load environment variables
env_loaded = load_dotenv()

# Argument parsing for course code
parser = argparse.ArgumentParser(description="Hybrid semantic-keyword search for Piazza courses.")
parser.add_argument('--course', type=str, default=os.getenv('PIAZZA_NETWORK_ID'),
                    help="Course network ID to process.")
args = parser.parse_args()
course_code = args.course
if course_code:
    base_dir = Path(course_code)
else:
    base_dir = Path('.')
# Ensure base directory exists
base_dir.mkdir(parents=True, exist_ok=True)

# Setup paths and embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
persist_dir = base_dir / "db"
hash_file = persist_dir / "posts_hash.txt"
json_path = base_dir / "posts.json"

# instantiate vision-capable LLM for image captioning
llm_vision = ChatOpenAI(model_name="gpt-4o-mini")

# Setup LangChain sentence splitters
splitter_2 = NLTKTextSplitter(chunk_size=2, chunk_overlap=1)
splitter_3 = NLTKTextSplitter(chunk_size=3, chunk_overlap=2)

# Helper: compute SHA-1 hash of a file to detect changes
def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# Text Cleaning
def clean_text(text: str) -> str:
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    return text.replace('`', '')

# Main indexing and retrieval logic
print(f"Starting index load/build phase for course {course_code or 'default'}")
start_all = time.perf_counter()

# Compute hash for posts.json
json_hash = sha1_of_file(str(json_path))
print(f"Computed SHA-1 hash for {json_path}: {json_hash[:8]}...")

# Build or load vector DB based on hash
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
    print("Splitting posts into sliding window chunks.")
    for post_id, post in data.items():
        subj = post.get('subject', '').strip()
        cont = post.get('content', '').strip()
        ia = post.get('instructor_answer', '').strip()
        ea = post.get('endorsed_answer', '').strip()
        full = ' '.join(filter(None, [subj, cont, ia, ea]))
        post_texts.append(full)
        post_ids.append(post_id)

        # Image captioning logic omitted for brevity...
        # Clean, chunk, and collect documents
        clean = clean_text(full)
        chunks_2 = splitter_2.split_text(clean)
        chunks_3 = splitter_3.split_text(clean)
        for idx, chunk in enumerate(chunks_2 + chunks_3):
            documents.append(Document(
                page_content=chunk,
                metadata={'post_id': post_id, 'subject': subj, 'idx': idx}
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

# If BM25 wasn't built in this run, build now
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
bm25_ids = bm25.get_top_n(tokens, post_ids, n=100)
bm25_set = set(bm25_ids)

# Semantic stage
results = vector_database.similarity_search_with_score(query, k=100)
results = [(d, dist) for d, dist in results if d.metadata['post_id'] in bm25_set]

# Score & output
post_scores = {}
for d, dist in results:
    sim = 1.0 - dist
    pid = d.metadata['post_id']
    subj = d.metadata['subject']
    post_scores.setdefault(pid, {'subject': subj, 'score': sim})
    post_scores[pid]['score'] = max(post_scores[pid]['score'], sim)

# Output top results
top_posts = sorted(post_scores.items(), key=lambda x: x[1]['score'], reverse=True)
print("Retrieval complete. Top 10 posts:")
for idx, (pid, info) in enumerate(top_posts[:10], start=1):
    print(f"{idx}. Post #{pid} — {info['subject']} (score: {info['score']:.4f})")