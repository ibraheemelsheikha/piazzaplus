import json
import re
import hashlib
from pathlib import Path
import time

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

load_dotenv()

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

# Convert Piazza image link to the redirect link by following HTTP redirect
def to_cdn_url(redirect_url: str) -> str:
    """
    Follows a Piazza redirect URL, returning the final CDN URL.
    """
    resp = requests.get(redirect_url, allow_redirects=False)
    if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
        return resp.headers.get('Location')
    resp.raise_for_status()
    return redirect_url

# Text Cleaning
def clean_text(text: str) -> str:
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    return text.replace('`', '')

# Setup LangChain sentence splitters
splitter_2 = NLTKTextSplitter(chunk_size=2, chunk_overlap=1)
splitter_3 = NLTKTextSplitter(chunk_size=3, chunk_overlap=2)

# Setup paths and embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
persist_dir = Path("./db")
hash_file = persist_dir / "posts_hash.txt"
json_path = Path("posts.json")

# instantiate vision-capable LLM for image captioning
llm_vision = ChatOpenAI(model_name="gpt-4o-mini")

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
    print("Splitting posts into sliding window chunks.")
    for post_id, post in data.items():
        subj = post.get('subject', '').strip()
        cont = post.get('content', '').strip()
        ia = post.get('instructor_answer', '').strip()
        ea = post.get('endorsed_answer', '').strip()
        full = ' '.join(filter(None, [subj, cont, ia, ea]))
        post_texts.append(full)
        post_ids.append(post_id)

        # If there are any Piazza redirect image URLs, generate captions
        image_urls = re.findall(r'https://piazza\.com/redirect/s3\?[^\s)]+', full)
        captions = []
        for redirect_url in image_urls:
            try:
                cdn_url = to_cdn_url(redirect_url)
                img_bytes = httpx.get(cdn_url, follow_redirects=True).content
                image_data = base64.b64encode(img_bytes).decode("utf-8")
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe this image to someone who is visually impaired. Please describe all drawings and transcribe any text. Use up to 350 words"},
                        {"type": "image", "source_type": "base64", "data": image_data, "mime_type": "image/png"},
                    ],
                }
                resp = llm_vision.invoke([message])
                captions.append(resp.text())
            except Exception as e:
                print(f"Image caption failed for {redirect_url}: {e}")
        if captions:
            full = full + ' ' + ' '.join(captions)

        clean = clean_text(full)

        # Create 2- and 3-sentence overlapping chunks
        chunks_2 = splitter_2.split_text(clean)
        chunks_3 = splitter_3.split_text(clean)
        chunks = chunks_2 + chunks_3

        for idx, chunk in enumerate(chunks):
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
bm25_ids = bm25.get_top_n(tokens, post_ids, n=100)
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
