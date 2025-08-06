import json
import re
import hashlib
from pathlib import Path
import time
import os

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

# load openai key
load_dotenv()

# get course nid from auth.json
with open("auth.json", "r", encoding="utf-8") as f:
    auth_map = json.load(f)
course_code = input("Enter course network ID: ").strip()
if course_code not in auth_map:
    raise KeyError(f"Course network ID {course_code} not found in auth.json")

# base data directory
data_dir = Path('data')
data_dir.mkdir(parents=True, exist_ok=True)

if course_code:
    base_dir = data_dir / course_code
else:
    base_dir = data_dir

# ensure base directory exists
base_dir.mkdir(parents=True, exist_ok=True)

# setup paths and embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
persist_dir = base_dir / "db"
hash_file = persist_dir / "posts_hash.txt"
json_path = base_dir / "posts.json"

# instantiate image captioning LLM
llm_vision = ChatOpenAI(model_name="gpt-4o-mini")

# setup langchain sentence splitters
splitter_2 = NLTKTextSplitter(chunk_size=20, chunk_overlap=12)
splitter_3 = NLTKTextSplitter(chunk_size=25, chunk_overlap=15)

# helper: compute hash of a file to detect changes
def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# convert piazza image link to the redirect link by following http redirect
def to_cdn_url(redirect_url: str) -> str:
    resp = requests.get(redirect_url, allow_redirects=False)
    if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
        return resp.headers.get('Location')
    resp.raise_for_status()
    return redirect_url

# text cleaner
def clean_text(text: str) -> str:
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    return text.replace('`', '')

# main indexing and retrieval logic
print(f"Starting index load/build phase for course {course_code or 'default'}")
start_all = time.perf_counter()

# compute hash for posts.json
json_hash = sha1_of_file(str(json_path))
print(f"Computed SHA-1 hash for {json_path}: {json_hash[:8]}...")

# build or load vector db based on hash
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

        # if there are any Piazza redirect image urls, generate captions
        image_urls = post.get("image_urls", [])
        captions = []
        for redirect_url in image_urls:
            try:
                cdn_url = to_cdn_url(redirect_url)
                img_bytes = httpx.get(cdn_url, follow_redirects=True).content
                image_data = base64.b64encode(img_bytes).decode("utf-8")
                message = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe this image to someone who is struggling with this course's content. Please describe all drawings and transcribe any text. Use up to 250 words."},
                        {"type": "image", "source_type": "base64", "data": image_data, "mime_type": "image/png"},
                    ],
                }
                resp = llm_vision.invoke([message])
                time.sleep(1)
                captions.append(resp.text())
            except Exception as e:
                print(f"Image caption failed for {redirect_url}: {e}")
        if captions:
            full = full + ' ' + ' '.join(captions)

        # clean, chunk, and collect documents
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

    # bm25 indexing
    tokenized_corpus = [re.findall(r"[A-Za-z]+|\d+", txt.lower()) for txt in post_texts]
    bm25 = BM25Okapi(tokenized_corpus)

# if bm25 wasn't built in this run, build now
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

# query & retrieval
print("Starting query phase.")
prep_start = time.perf_counter()
query = input("Enter query: ")
prep_end = time.perf_counter()
print(f"Query input received in {prep_end - prep_start:.2f} seconds.")

# bm25 stage
tokens = re.findall(r"[A-Za-z]+|\d+", query.lower())
bm25_ids = bm25.get_top_n(tokens, post_ids, n=100)
bm25_set = set(bm25_ids)

# semantic stage
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

# output top results
top_posts = sorted(post_scores.items(), key=lambda x: x[1]['score'], reverse=True)
print("Retrieval complete. Top 10 posts:")
for idx, (pid, info) in enumerate(top_posts[:10], start=1):
    print(f"{idx}. Post #{pid} — {info['subject']} (score: {info['score']:.4f})")
