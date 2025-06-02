import os
import json
from piazza_api import Piazza
from post import create_post_from_api
from bs4 import MarkupResemblesLocatorWarning
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# file fetched post data is stored
STORAGE_FILE = 'posts.json'
# time (in seconds) to sleep between fetches (to avoid ban)
SLEEP_TIME = 1


def load_stored_posts():
    # load stored posts from STORAGE_FILE. returns a dict mapping post IDs to saved data
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_stored_posts(data):
    # save the posts dict back to STORAGE_FILE
    with open(STORAGE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    # load credentials from environment
    email = os.getenv('PIAZZA_EMAIL')
    password = os.getenv('PIAZZA_PASSWORD')
    class_code = os.getenv('PIAZZA_NETWORK_ID')

    if not all([email, password, class_code]):
        print("Error: Please set PIAZZA_EMAIL, PIAZZA_PASSWORD, and PIAZZA_NETWORK_ID in environment.")
        return

    # log in to course
    p = Piazza()
    p.user_login(email=email, password=password)
    network = p.network(class_code)

    # load stored snapshot
    stored = load_stored_posts()
    new_posts = []      # fresh posts we haven't seen
    changed_posts = []  # posts whose content/flags have changed

    # iterate through course posts (newest->oldest)
    for raw in network.iter_all_posts(limit=None, sleep=SLEEP_TIME):
        post_id = str(raw.get('nr'))
        post = create_post_from_api(raw)

        # prepare the minimal dict of fields to track
        current_snapshot = {
            'subject': post.subject,
            'content': post.content,
            'has_instructor_answer': post.instructor_answer is not None,
            'has_instructor_endorsement': post.endorsed_answer is not None,
        }
        # include full answer texts if present
        if post.instructor_answer is not None:
            current_snapshot['instructor_answer'] = post.instructor_answer
        if post.endorsed_answer is not None:
            current_snapshot['endorsed_answer'] = post.endorsed_answer

        if post_id not in stored:
            # brand new post
            stored[post_id] = current_snapshot
            new_posts.append(post)
        else:
            # seen before so check for any changes
            if stored[post_id] != current_snapshot:
                stored[post_id] = current_snapshot
                changed_posts.append(post)

    # store any updates
    if new_posts or changed_posts:
        save_stored_posts(stored)

    # reporting
    if new_posts:
        print(f"New posts ({len(new_posts)}):")
        for p in new_posts:
            print(f"  #{p.number}: {p.subject}")
            # show instructor/student answers if available
            if p.instructor_answer:
                print(f"    Instructor answer: {p.instructor_answer}")
            if p.endorsed_answer:
                print(f"    Endorsed student answer: {p.endorsed_answer}")

    if changed_posts:
        print(f"Updated posts ({len(changed_posts)}):")
        for p in changed_posts:
            print(f"  #{p.number}: {p.subject}")


if __name__ == '__main__':
    main()
