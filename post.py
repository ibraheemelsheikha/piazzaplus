from bs4 import BeautifulSoup  # to parse html

class Post:
    # storing most important data as attributes, now including image info
    def __init__(
        self,
        number,
        subject,
        content,
        instructor_answer=None,
        endorsed_answer=None,
        has_image=False,
        image_urls=None,
    ):
        self.number = number
        self.subject = subject
        self.content = content
        self.instructor_answer = instructor_answer
        self.endorsed_answer = endorsed_answer
        self.has_image = has_image
        self.image_urls = image_urls or []


def create_post_from_api(raw):
    """
    Build a Post object from the raw Piazza API response.
    Extracts initial post content, images, instructor answers, and endorsed student answers.
    """
    # Post number
    number = raw.get('nr')

    # subject: direct or from history as fallback
    subject = raw.get('subject', '') or raw.get('history', [{}])[0].get('subject', '')
    subject = subject.strip()

    # initial post content and image detection
    history = raw.get('history', [])
    initial_entry = history[0] if history else {}
    initial_html = initial_entry.get('content', '')
    soup = BeautifulSoup(initial_html, 'html.parser')
    img_tags = soup.find_all('img')
    image_urls = [img['src'] for img in img_tags if img.get('src')]
    has_image = bool(image_urls)
    content = soup.get_text(separator=' ', strip=True)

    # initialize answer fields
    instructor_answer = None
    endorsed_answer = None

    # traverse follow-up children for answers
    for child in raw.get('children', []):
        # use latest history entry for content
        ch_history = child.get('history', [])
        latest = ch_history[-1] if ch_history else {}
        child_html = latest.get('content', '')
        child_text = BeautifulSoup(child_html, 'html.parser').get_text(separator=' ', strip=True)

        # detect an instructor/professor answer
        if child.get('type') == 'i_answer' and instructor_answer is None:
            instructor_answer = child_text or None
            # continue to look for endorsed student answers

        # detect a student answer that has been endorsed by an instructor/professor
        if child.get('type') in ('s_answer', 'followup') and endorsed_answer is None:
            endorsements = child.get('tag_endorse', []) + child.get('tag_good', [])
            if any(e.get('role') in ('instructor', 'professor') for e in endorsements):
                endorsed_answer = child_text or None

        # stop early if both answers found
        if instructor_answer is not None and endorsed_answer is not None:
            break

    # return populated Post
    return Post(
        number=number,
        subject=subject,
        content=content,
        instructor_answer=instructor_answer,
        endorsed_answer=endorsed_answer,
        has_image=has_image,
        image_urls=image_urls,
    )
