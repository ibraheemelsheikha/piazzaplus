import os
import json
from piazza_api import Piazza
from post import create_post_from_api
from bs4 import MarkupResemblesLocatorWarning
import warnings
from dotenv import load_dotenv
#warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

load_dotenv()

# Piazza domain prefix for image URLs
PIAZZA_DOMAIN = "https://piazza.com"

# File where fetched post data is stored
STORAGE_FILE = 'posts.json'
# Time (in seconds) to sleep between fetches (to avoid ban)
SLEEP_TIME = 2.0


def load_stored_posts():
    """Load stored posts from STORAGE_FILE. Returns a dict mapping post IDs to saved data."""
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_stored_posts(data):
    """Save the posts dict back to STORAGE_FILE."""
    with open(STORAGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def main():
    # Load credentials from environment
    email = os.getenv('PIAZZA_EMAIL')
    password = os.getenv('PIAZZA_PASSWORD')
    class_code = os.getenv('PIAZZA_NETWORK_ID')

    if not all([email, password, class_code]):
        print("Error: Please set PIAZZA_EMAIL, PIAZZA_PASSWORD, and PIAZZA_NETWORK_ID in environment.")
        return

    # Log in to Piazza
    piazza = Piazza()
    piazza.user_login(email=email, password=password)
    network = piazza.network(class_code)

    # Load previously stored snapshot
    stored = load_stored_posts()
    new_posts = [] # Fresh posts we haven't seen
    changed_posts = [] # Posts whose content/flags have changed

    # Iterate through all posts (newest → oldest)
    for summary in network.iter_all_posts(limit=None, sleep=SLEEP_TIME):
        post_id = str(summary.get('nr'))
        raw = network.get_post(post_id)
        post = create_post_from_api(raw)

        # Prefix any relative image URLs with the Piazza domain
        if post.has_image:
            post.image_urls = [PIAZZA_DOMAIN + url if url.startswith('/') else url for url in post.image_urls]

        # Prepare current snapshot including image flag and URLs
        current_snapshot = {
            'subject': post.subject,
            'content': post.content,
            'has_instructor_answer': post.instructor_answer is not None,
            'has_instructor_endorsement': post.endorsed_answer is not None,
            'has_image': post.has_image,
        }
        if post.instructor_answer is not None:
            current_snapshot['instructor_answer'] = post.instructor_answer
        if post.endorsed_answer is not None:
            current_snapshot['endorsed_answer'] = post.endorsed_answer
        if post.has_image:
            current_snapshot['image_urls'] = post.image_urls

        # Detect new or changed posts
        if post_id not in stored:
            stored[post_id] = current_snapshot
            new_posts.append(post)
        else:
            if stored[post_id] != current_snapshot:
                stored[post_id] = current_snapshot
                changed_posts.append(post)

    # Save updates back to storage
    if new_posts or changed_posts:
        save_stored_posts(stored)

    # Reporting new posts
    if new_posts:
        print(f"New posts ({len(new_posts)}):")
        for p in new_posts:
            print(f"  #{p.number}: {p.subject}")
            print(f"    Content: {p.content[:60]}...")
            if p.instructor_answer:
                print(f"    Instructor answer: {p.instructor_answer}")
            if p.endorsed_answer:
                print(f"    Endorsed student answer: {p.endorsed_answer}")
            if p.has_image:
                print(f"    Contains images: {p.image_urls}")

    # Reporting changed posts
    if changed_posts:
        print(f"Updated posts ({len(changed_posts)}):")
        for p in changed_posts:
            print(f"  #{p.number}: {p.subject}")
            print(f"    Content: {p.content[:60]}...")
            if p.instructor_answer:
                print(f"    Instructor answer: {p.instructor_answer}")
            if p.endorsed_answer:
                print(f"    Endorsed student answer: {p.endorsed_answer}")
            if p.has_image:
                print(f"    Contains images: {p.image_urls}")

if __name__ == '__main__':
    main()
