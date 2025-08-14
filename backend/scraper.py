import os
import json
import time
from pathlib import Path
import warnings
from piazza_api import Piazza
from bs4 import MarkupResemblesLocatorWarning
from utils import save_stored_posts, load_stored_posts
from post import create_post_from_api

# suppress html parsing warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# --- CONFIGURATION ---
AUTH_PATH = Path("auth.json")
PIAZZA_DOMAIN = "https://piazza.com"
RATE_LIMIT = 2.0        # seconds between API calls
SCRAPE_INTERVAL = 5 * 60  # seconds between runs

# load credentials map
if not AUTH_PATH.exists():
    print("auth.json not found.")
    exit(1)
auth_map = json.loads(AUTH_PATH.read_text())


def process_course(course_code: str, creds: dict):
    """
    Log into Piazza for course_code, look for only the newest posts (skipping pinned posts),
    stopping at the first already-stored post, save them, and print a summary.
    """
    # prepare storage
    course_dir = Path("data") / course_code
    course_dir.mkdir(parents=True, exist_ok=True)
    storage_file = course_dir / "posts.json"
    stored = load_stored_posts(storage_file)

    new_posts = []

    # login
    piazza = Piazza()
    piazza.user_login(email=creds["email"], password=creds["password"])
    network = piazza.network(course_code)

    # iterate newest to oldest, skipping pinned posts, stop once we hit a persisted post
    for summary in network.iter_all_posts(limit=None, sleep=RATE_LIMIT):
        # skip any pinned posts so they don't interrupt the chronological check
        if summary.get("bucket_name") == "Pinned" or "pin" in summary.get("tags", []):
            continue

        post_id = str(summary.get("nr"))
        if post_id in stored:
            break

        raw = network.get_post(post_id)
        post = create_post_from_api(raw)

        # rewrite image URLs to full CDN links
        if post.has_image:
            post.image_urls = [
                PIAZZA_DOMAIN + url if url.startswith("/") else url
                for url in post.image_urls
            ]

        # take a snapshot of the fields i care about
        snapshot = {
            "subject": post.subject,
            "content": post.content,
            "has_instructor_answer": post.instructor_answer is not None,
            "has_instructor_endorsement": post.endorsed_answer is not None,
            "has_image": post.has_image,
        }
        if post.instructor_answer:
            snapshot["instructor_answer"] = post.instructor_answer
        if post.endorsed_answer:
            snapshot["endorsed_answer"] = post.endorsed_answer
        if post.has_image:
            snapshot["image_urls"] = post.image_urls

        # record it
        stored[post_id] = snapshot
        new_posts.append(post)

    # persist only if there are new posts, with new posts at the top
    if new_posts:
        new_ids = [str(p.number) for p in new_posts]
        reordered = {}
        for pid in new_ids:
            reordered[pid] = stored[pid]
        for pid, snapshot in stored.items():
            if pid not in new_ids:
                reordered[pid] = snapshot
        save_stored_posts(reordered, storage_file)


if __name__ == "__main__":
    while True:
        # run through every course in auth.json
        for course_code, creds in auth_map.items():
            try:
                process_course(course_code, creds)
            except Exception as e:
                print(f"[ERROR] {course_code}: {e}")

        print(f"Waiting {SCRAPE_INTERVAL} seconds until next update...")
        time.sleep(SCRAPE_INTERVAL)
