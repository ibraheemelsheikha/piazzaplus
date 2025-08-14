import json
import re
from pathlib import Path
import time
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
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
json_path = base_dir / "posts.json"

# load or fail if no DB
vector_database = Chroma(
    persist_directory=str(persist_dir),
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

# bm25 indexing
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
post_texts = [' '.join(filter(None, [
    p.get('subject',''), p.get('content',''),
    p.get('instructor_answer',''), p.get('endorsed_answer','')
])) for p in data.values()]
post_ids = list(data.keys())
tokenized_corpus = [re.findall(r"[A-Za-z]+|\d+", txt.lower()) for txt in post_texts]
bm25 = BM25Okapi(tokenized_corpus)

# query & retrieval
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