import json
import re
import time
import base64
import httpx
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from rank_bm25 import BM25Okapi
from utils import sha1_of_file, clean_text, to_cdn_url, splitter

# Silence overly verbose logs
logging.disable(logging.WARNING)

# Load environment and credentials
load_dotenv()
with open("auth.json", "r", encoding="utf-8") as f:
    auth_map = json.load(f)

# Constants and shared models
SCRAPE_INTERVAL = 5 * 60  # seconds between updates
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
llm_vision = ChatOpenAI(model_name="gpt-4o-mini")


def update_database():
    """
    Incrementally build or update the Chroma vector DB using globals:
      persist_dir, hash_file, json_path, vector_file,
      embedding_model, llm_vision
    """
    # ensure storage directory exists
    persist_dir.mkdir(parents=True, exist_ok=True)

    # incremental vs initial build
    if persist_dir.exists() and hash_file.exists():
        db = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )
        # compare posts.json hash
        current_hash = sha1_of_file(str(json_path))
        last_hash = hash_file.read_text()
        if current_hash == last_hash:
            print("No new posts to vectorize.")
            return

        # load previously vectorized IDs
        if vector_file.exists():
            with open(vector_file, 'r', encoding='utf-8') as vf:
                vectorized_ids = set(json.load(vf))
        else:
            vectorized_ids = set()

        print("Detected changes; updating vector database...")
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        new_ids = [pid for pid in data if pid not in vectorized_ids]
        if not new_ids:
            print("No unvectorized posts found.")
        else:
            for pid in new_ids:
                post = data[pid]
                subj = post.get('subject','').strip()
                cont = post.get('content','').strip()
                ia = post.get('instructor_answer','').strip()
                ea = post.get('endorsed_answer','').strip()
                full = ' '.join(filter(None,[subj,cont,ia,ea]))

                # caption images
                captions = []
                for url in post.get('image_urls',[]):
                    try:
                        cdn = to_cdn_url(url)
                        img = httpx.get(cdn, follow_redirects=True, timeout=10).content
                        enc = base64.b64encode(img).decode('utf-8')
                        msg = {"role":"user","content":[
                            {"type":"text","text":"Describe this image for course content, transcribe text, up to 250 words."},
                            {"type":"image","source_type":"base64","data":enc,"mime_type":"image/png"},
                        ]}
                        resp = llm_vision.invoke([msg])
                        time.sleep(1)
                        captions.append(resp.text())
                    except Exception as e:
                        print(f"#{pid}: caption failed: {e}")
                if captions:
                    full += ' ' + ' '.join(captions)

                # chunk & embed
                chunks = splitter.split_text(clean_text(full))
                docs = [Document(page_content=chunk, metadata={'post_id':pid,'subject':subj,'idx':i})
                        for i,chunk in enumerate(chunks)]
                print(f"Embedding {len(docs)} chunks for post {pid}...")
                db.add_documents(docs)
                vectorized_ids.add(pid)

            # save vectorized IDs
            Path(vector_file).write_text(json.dumps(list(vectorized_ids), indent=2), encoding='utf-8')

        # update hash
        hash_file.write_text(current_hash)
        print("Update complete.")

    else:
        # full initial build
        print("Performing initial full build...")
        start = time.perf_counter()
        data = json.loads(Path(json_path).read_text(encoding='utf-8'))

        docs, texts = [], []
        for pid, post in data.items():
            subj = post.get('subject','').strip()
            cont = post.get('content','').strip()
            ia = post.get('instructor_answer','').strip()
            ea = post.get('endorsed_answer','').strip()
            full = ' '.join(filter(None,[subj,cont,ia,ea]))
            texts.append(full)

            # caption images (same as above)
            caps=[]
            for url in post.get('image_urls',[]):
                try:
                    cdn = to_cdn_url(url)
                    img = httpx.get(cdn, follow_redirects=True, timeout=10).content
                    enc = base64.b64encode(img).decode('utf-8')
                    msg={"role":"user","content":[
                        {"type":"text","text":"Describe this image for course content, transcribe text, up to 250 words."},
                        {"type":"image","source_type":"base64","data":enc,"mime_type":"image/png"},
                    ]}
                    r=llm_vision.invoke([msg]); time.sleep(1)
                    caps.append(r.text())
                except:
                    pass
            if caps:
                full += ' ' + ' '.join(caps)

            # chunk
            for i,chunk in enumerate(splitter.split_text(clean_text(full))):
                docs.append(Document(page_content=chunk, metadata={'post_id':pid,'subject':subj,'idx':i}))

        print(f"Embedding total {len(docs)} chunks...")
        db = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=str(persist_dir),
            collection_metadata={"hnsw:space":"cosine"}
        )

        # record initial state
        current_hash = sha1_of_file(str(json_path))
        hash_file.write_text(current_hash)
        Path(vector_file).write_text(json.dumps(list(data.keys()), indent=2), encoding='utf-8')
        elapsed = time.perf_counter() - start
        print(f"Initial build done in {elapsed:.2f}s.")

        # BM25 indexing (unchanged)
        tokenized = [re.findall(r"[A-Za-z]+|\d+", t.lower()) for t in texts]
        bm25 = BM25Okapi(tokenized)


if __name__ == "__main__":
    while True:
        for course_code, creds in auth_map.items():
            # per-course paths
            data_dir = Path('data')
            data_dir.mkdir(parents=True, exist_ok=True)
            base_dir = data_dir / course_code
            base_dir.mkdir(parents=True, exist_ok=True)
            persist_dir = base_dir / "db"
            hash_file   = persist_dir / "posts_hash.txt"
            json_path   = base_dir / "posts.json"
            vector_file = persist_dir / "vectorized_ids.json"

            try:
                print(f"Starting update for {course_code}...")
                update_database()
            except Exception as e:
                print(f"[ERROR] {course_code}: {e}")

        print(f"Waiting {SCRAPE_INTERVAL} seconds until next update...")
        time.sleep(SCRAPE_INTERVAL)
