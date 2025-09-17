import os
import json
import time
from pathlib import Path
import warnings
from piazza_api import Piazza
from bs4 import MarkupResemblesLocatorWarning
from utils import save_stored_posts, load_stored_posts
from post import create_post_from_api
from datetime import datetime, timezone, timedelta

# suppress html parsing warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# --- configuration ---
AUTH_PATH = Path("auth.json")
PIAZZA_DOMAIN = "https://piazza.com"
RATE_LIMIT = 2.0 # seconds between API calls
SCRAPE_INTERVAL = 10 * 60 # seconds between runs
REFRESH_WINDOW = timedelta(days=7) # how far back to refresh existing posts

# load credentials map
if not AUTH_PATH.exists():
    print("auth.json not found.")
    exit(1)
auth_map = json.loads(AUTH_PATH.read_text())


def process_course(course_code: str, creds: dict):
    """
    Log into Piazza for course_code, scrape newest->oldest.
    - First run: scrape ALL posts (including pinned).
    - Subsequent runs: skip pinned; refresh anything created within REFRESH_WINDOW;
      and stop early once we hit the first non-pinned post that is older than the window AND already stored.
    """
    # prepare storage
    course_dir = Path("data") / course_code
    course_dir.mkdir(parents=True, exist_ok=True)
    storage_file = course_dir / "posts.json"

    # determine first-run BEFORE loading (load_stored_posts may create the file)
    first_run = not storage_file.exists()
    stored = load_stored_posts(storage_file)  # dict[str, snapshot]

    new_posts = []

    # login
    piazza = Piazza()
    piazza.user_login(email=creds["email"], password=creds["password"])
    network = piazza.network(course_code)

    cutoff = datetime.now(timezone.utc) - REFRESH_WINDOW

    # iterate newest to oldest
    for summary in network.iter_all_posts(limit=None, sleep=RATE_LIMIT):
        post_id = str(summary.get("nr"))

        # parse creation time from summary
        created_str = summary.get("created")
        created = None
        if created_str:
            try:
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except Exception:
                created = None

        if not first_run and stored.get(post_id, {}).get("is_pinned", False):
            continue

        raw = network.get_post(post_id)
        is_pinned = bool(raw.get("is_pinned", False))

        # on subsequent runs, skip pinned posts entirely
        if not first_run and is_pinned:
            if post_id in stored and not stored[post_id].get("is_pinned", False):
                stored[post_id]["is_pinned"] = True
                new_posts.append(create_post_from_api(raw))
            continue

        if not first_run and created and created < cutoff and post_id in stored:
            print(f"stopping at post {post_id} in course {course_code} because it is older than 7 days")
            break

        # build post object
        post = create_post_from_api(raw)

        # rewrite image urls to full cdn links
        if post.has_image:
            post.image_urls = [
                PIAZZA_DOMAIN + url if isinstance(url, str) and url.startswith("/") else url
                for url in post.image_urls
            ]

        # snapshot of fields
        snapshot = {
            "subject": post.subject,
            "content": post.content,
            "has_instructor_answer": post.instructor_answer is not None,
            "has_instructor_endorsement": post.endorsed_answer is not None,
            "has_image": post.has_image,
            "is_pinned": is_pinned,
        }
        if post.instructor_answer:
            snapshot["instructor_answer"] = post.instructor_answer
        if post.endorsed_answer:
            snapshot["endorsed_answer"] = post.endorsed_answer
        if post.has_image:
            snapshot["image_urls"] = post.image_urls

        # record if new or changed
        if post_id not in stored or stored[post_id] != snapshot:
            stored[post_id] = snapshot
            new_posts.append(post)

    # persist only if there are new/updated posts
    if new_posts:
        new_ids = [str(p.number) for p in new_posts if hasattr(p, "number")]
        reordered = {}
        for pid in new_ids:
            if pid in stored:
                reordered[pid] = stored[pid]
        for pid, snapshot in stored.items():
            if pid not in reordered:
                reordered[pid] = snapshot

        save_stored_posts(reordered, storage_file)


if __name__ == "__main__":
    while True:
        for course_code, creds in auth_map.items():
            try:
                process_course(course_code, creds)
            except Exception as e:
                print(f"[ERROR] {course_code}: {e}")

        print(f"Waiting {SCRAPE_INTERVAL} seconds until next update...")
        time.sleep(SCRAPE_INTERVAL)
