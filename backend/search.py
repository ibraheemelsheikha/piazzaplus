import json
import time
from pathlib import Path
from dotenv import load_dotenv

# use the same search pipeline as the API
from search_lib import search_top_k

# load OPENAI_API_KEY from .env
load_dotenv()

# get course nid from auth.json
with open("auth.json", "r", encoding="utf-8") as f:
    auth_map = json.load(f)

course_code = input("Enter course network ID: ").strip()
if course_code not in auth_map:
    raise KeyError(f"Course network ID {course_code} not found in auth.json")

# get query
prep_start = time.perf_counter()
query = input("Enter query: ").strip()
prep_end = time.perf_counter()
print(f"Query input received in {prep_end - prep_start:.2f} seconds.")

results = search_top_k(course_code, query, k=10)

# print results in the same style as before
print("Retrieval complete. Top 10 posts:")
for idx, item in enumerate(results[:10], start=1):
    print(f"{idx}. Post #{item['post_id']} — {item['subject']} (score: {item['score']:.4f})")
