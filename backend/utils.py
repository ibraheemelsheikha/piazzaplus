import json
import re
import hashlib
from pathlib import Path
from langchain.text_splitter import NLTKTextSplitter
import requests
import nltk

# setup langchain sentence splitter
nltk.download('punkt_tab', quiet=True)
splitter = NLTKTextSplitter(chunk_size=1, chunk_overlap=0)

# compute hash of a file to detect changes
def sha1_of_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# text cleaner
def clean_text(text: str) -> str:
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    return text.replace('`', '')

# load posts from json
def load_stored_posts(path: Path) -> dict:
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# convert piazza image link to the redirect link by following http redirect
def to_cdn_url(redirect_url: str) -> str:
    resp = requests.get(redirect_url, allow_redirects=False)
    if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
        return resp.headers.get('Location')
    resp.raise_for_status()
    return redirect_url

def save_stored_posts(data: dict, path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)