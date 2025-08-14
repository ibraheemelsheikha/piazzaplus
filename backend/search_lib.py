import json, re
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()  # uses OPENAI_API_KEY

def search_top_k(course_code: str, query: str, k: int = 10):
    base_dir = Path("data") / course_code
    persist_dir = base_dir / "db"
    json_path = base_dir / "posts.json"
    if not persist_dir.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Missing vector DB or posts.json for {course_code}. "
        )

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_database = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # bm25 over whole-post text
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    post_texts = [' '.join(filter(None, [
        p.get('subject',''), p.get('content',''),
        p.get('instructor_answer',''), p.get('endorsed_answer','')
    ])) for p in data.values()]
    post_ids = list(data.keys())

    tokenized_corpus = [re.findall(r"[A-Za-z]+|\d+", txt.lower()) for txt in post_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    tokens = re.findall(r"[A-Za-z]+|\d+", query.lower())
    bm25_ids = bm25.get_top_n(tokens, post_ids, n=100)
    bm25_set = set(bm25_ids)

    # semantic stage
    results = vector_database.similarity_search_with_score(query, k=100)
    results = [(d, dist) for d, dist in results if d.metadata["post_id"] in bm25_set]

    scored = {}
    for d, dist in results:
        sim = 1.0 - dist
        pid = d.metadata["post_id"]
        subj = d.metadata["subject"]
        scored.setdefault(pid, {"post_id": pid, "subject": subj, "score": sim})
        scored[pid]["score"] = max(scored[pid]["score"], sim)

    top = sorted(scored.values(), key=lambda x: x["score"], reverse=True)[:k]
    return top
