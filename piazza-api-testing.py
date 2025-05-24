import os #to read environment variables (for testing purposes)
from dotenv import load_dotenv #to read from .env (for testing purposes)

from piazza_api import Piazza #to log in and fetch posts
from post import create_post_from_api, Post #to wrap raw piazza data into nice objects

#email and password are stored in a .env file (for testing purposes)
load_dotenv()
PIAZZA_EMAIL = os.environ["PIAZZA_EMAIL"]
PIAZZA_PASSWORD = os.environ["PIAZZA_PASSWORD"]

#logging in to access courses
p = Piazza()
p.user_login(email=PIAZZA_EMAIL, password=PIAZZA_PASSWORD)

#from this code, found that the nid (unique course network id) for APS105 Computer fundamentals 2025 is m5dkvmh5bv87oy
#other course nids can be found in course-nids.txt
'''
courses = p.get_user_classes()
for course in courses:
    print(f"{course['name']}: nid = {course['nid']}")
'''

aps105 = p.network("m5dkvmh5bv87oy") #"logging into" aps105 using its unique nid

sleep_seconds = 1 #number of seconds to wait between posts when iterating through all course posts
limit_posts = 10 #maximum number of posts to iterate through (for testing purposes. real program should iterate through all posts)
posts = [] #array to store all posts

for raw in aps105.iter_all_posts(limit = limit_posts, sleep=sleep_seconds): #iterating through limit_posts number of posts
    post_obj = create_post_from_api(raw) #creating a custom post object
    posts.append(post_obj) #appending to posts array

print(f"Fetched {len(posts)} posts and turned them into post objects")

#after testing, iterates through posts from top to bottom. so pinned first, and then most recent to least recent
for post in posts:
    print(post.number)