#!/usr/bin/env python3
import os
import json
import time
from pathlib import Path
import warnings

from piazza_api import Piazza
from post import create_post_from_api
from bs4 import MarkupResemblesLocatorWarning
from dotenv import load_dotenv

# Suppress HTML parsing warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
load_dotenv()

PIAZZA_DOMAIN = "https://piazza.com"
RATE_LIMIT = 2.0  # seconds between API calls
SCRAPE_INTERVAL = 5 * 60  # seconds between full scrapes


def load_stored_posts(path: Path) -> dict:
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_stored_posts(data: dict, path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def main():
    # ---------- Credentials ----------
    email = os.getenv('PIAZZA_EMAIL')
    password = os.getenv('PIAZZA_PASSWORD')
    if not email or not password:
        print("Error: PIAZZA_EMAIL and PIAZZA_PASSWORD must be set.")
        return

    # ---------- Target Courses ----------
    ids_env = os.getenv('PIAZZA_NETWORK_IDS')
    if ids_env:
        network_ids = [nid.strip() for nid in ids_env.split(',') if nid.strip()]
    else:
        single = os.getenv('PIAZZA_NETWORK_ID')
        if not single:
            print("Error: Set PIAZZA_NETWORK_ID or PIAZZA_NETWORK_IDS in environment.")
            return
        network_ids = [single]

    # ---------- Piazza Login ----------
    piazza = Piazza()
    piazza.user_login(email=email, password=password)

    # ---------- Scrape Loop ----------
    for class_code in network_ids:
        course_dir = Path(class_code)
        course_dir.mkdir(parents=True, exist_ok=True)
        storage_file = course_dir / 'posts.json'

        stored = load_stored_posts(storage_file)
        new_posts, changed_posts = [], []

        network = piazza.network(class_code)
        for summary in network.iter_all_posts(limit=None, sleep=RATE_LIMIT):
            post_id = str(summary.get('nr'))
            raw = network.get_post(post_id)
            post = create_post_from_api(raw)

            if post.has_image:
                post.image_urls = [
                    PIAZZA_DOMAIN + url if url.startswith('/') else url
                    for url in post.image_urls
                ]

            snapshot = {
                'subject': post.subject,
                'content': post.content,
                'has_instructor_answer': post.instructor_answer is not None,
                'has_instructor_endorsement': post.endorsed_answer is not None,
                'has_image': post.has_image,
            }
            if post.instructor_answer:
                snapshot['instructor_answer'] = post.instructor_answer
            if post.endorsed_answer:
                snapshot['endorsed_answer'] = post.endorsed_answer
            if post.has_image:
                snapshot['image_urls'] = post.image_urls

            if post_id not in stored:
                stored[post_id] = snapshot
                new_posts.append(post)
            elif stored[post_id] != snapshot:
                stored[post_id] = snapshot
                changed_posts.append(post)

        if new_posts or changed_posts:
            save_stored_posts(stored, storage_file)

        print(f"=== Course: {class_code} ===")
        if new_posts:
            print(f"New posts ({len(new_posts)}):")
            for p in new_posts:
                print(f"  #{p.number}: {p.subject}")
        if changed_posts:
            print(f"Updated posts ({len(changed_posts)}):")
            for p in changed_posts:
                print(f"  #{p.number}: {p.subject}")


if __name__ == '__main__':
    while True:
        main()
        print(f"Waiting {SCRAPE_INTERVAL} seconds until next run...")
        time.sleep(SCRAPE_INTERVAL)
