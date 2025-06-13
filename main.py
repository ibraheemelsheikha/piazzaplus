import json
import re
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Helper: compute SHA-1 hash of a file to detect changes
# ---------------------------------------------------------------------------
def sha1_of_file(path: str) -> str:
    """Return the SHA-1 hash of the contents of the file at `path`."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------------
# Custom Embeddings Class
# Wraps SentenceTransformer for embed_documents and embed_query
# ---------------------------------------------------------------------------
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors."""
        return self.model.encode(texts, convert_to_tensor=False).tolist()
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string into a vector."""
        return self.model.encode(text, convert_to_tensor=False).tolist()

# ---------------------------------------------------------------------------
# Text Cleaning
# Remove Markdown images, convert links, strip inline code markers
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    # drop image markdown ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # convert [label](url) -> label
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # remove inline backticks
    return text.replace('`', '')

# ---------------------------------------------------------------------------
# Sentence Chunking
# Split on ., ?, or ! followed by whitespace, preserving sentences
# ---------------------------------------------------------------------------
def split_into_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # split at punctuation boundary
        parts = re.split(r'(?<=[\.\?!])\s+', line)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences

# ---------------------------------------------------------------------------
# Initialization
# Set up embedding model and paths for persistence
# ---------------------------------------------------------------------------
embedding_model = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
persist_dir = Path("./db")
hash_file = persist_dir / "posts_hash.txt"
json_path = Path("posts.json")

# ---------------------------------------------------------------------------
# Database Loading or Rebuild
# Use SHA-1 of posts.json to guard rebuilds
# ---------------------------------------------------------------------------
json_hash = sha1_of_file(str(json_path))
if persist_dir.exists() and hash_file.exists() and hash_file.read_text() == json_hash:
    # Load existing Chroma DB if JSON unchanged
    vector_database = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
else:
    # Rebuild: read all posts, clean, chunk, and index
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents: list[Document] = []
    for post_id, post in data.items():
        subject = post.get('subject', '').strip()
        content = post.get('content', '').strip()
        instr_ans = post.get('instructor_answer', '').strip()
        end_ans = post.get('endorsed_answer', '').strip()
        full_text = ' '.join(filter(None, [subject, content, instr_ans, end_ans]))

        clean = clean_text(full_text)
        sentence_chunks = split_into_sentences(clean)

        for idx, chunk in enumerate(sentence_chunks):
            meta = {
                'post_id': post_id,
                'subject': subject,
                'sentence_index': idx
            }
            documents.append(Document(page_content=chunk, metadata=meta))

    # Build Chroma with cosine via collection_metadata and persist
    vector_database = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"}
    )
    hash_file.write_text(json_hash)

# ---------------------------------------------------------------------------
# Query & Retrieval
# Prompt user, search top 100 chunks, then aggregate by post
# ---------------------------------------------------------------------------
query = input("Enter a query here: ")
results = vector_database.similarity_search_with_score(query, k=100)

# ── Aggregate chunk‐level distances as similarities per post_id ──
post_scores: dict[str, dict[str, float]] = {}
for doc, dist in results:
    pid  = doc.metadata['post_id']
    subj = doc.metadata['subject']
    sim  = 1.0 - dist                   # convert distance → similarity
    if pid not in post_scores:
        post_scores[pid] = {'subject': subj, 'score': 0.0}
    post_scores[pid]['score'] += sim    # sum up similarities

# ── Sort by total similarity (highest first) ──
sorted_posts = sorted(
    post_scores.items(),
    key=lambda x: x[1]['score'],
    reverse=True                       # now higher sim → better
)

# ── Print the top 10 most semantically similar posts ──
for rank, (pid, info) in enumerate(sorted_posts[:10], start=1):
    print(f"{rank}. Post #{pid} — {info['subject']} (score: {info['score']:.4f})")

