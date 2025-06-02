from bs4 import BeautifulSoup # to parse html

class Post:
    # storing most important data as attributes. more can be added later
    def __init__(self, number, subject, content, instructor_answer=None, endorsed_answer=None):
        self.number = number
        self.subject = subject
        self.content = content
        # optional: text of the instructor's answer, if any
        self.instructor_answer = instructor_answer
        # optional: text of a student answer endorsed by instructor/professor
        self.endorsed_answer = endorsed_answer


def create_post_from_api(raw):
    number = raw.get('nr')  # post number

    # parse subject and content from HTML to plain text
    subj_html = raw['history'][0].get('subject', '')
    cont_html = raw['history'][0].get('content', '')
    subject = BeautifulSoup(subj_html, 'html.parser').get_text(separator=' ', strip=True)
    content = BeautifulSoup(cont_html, 'html.parser').get_text(separator=' ', strip=True)

    instructor_answer = None
    endorsed_answer = None

    # scan children for instructor answers and endorsed student answers
    for child in raw.get('children', []):
        ctype = child.get('type')
        # handle instructor answer
        if ctype == 'i_answer' and instructor_answer is None:
            # pull the first revision's HTML
            html = child.get('history', [{}])[0].get('content', '')
            instructor_answer = BeautifulSoup(html, 'html.parser').get_text(separator=' ', strip=True)
            # don't continue, an endorsed student answer may follow
        # handle endorsed student answer
        endorsements = child.get('tag_endorse', []) + child.get('tag_good', [])
        for e in endorsements:
            if e.get('role') in ('instructor', 'professor') and endorsed_answer is None:
                # student answer endorsed by instructor/professor
                html = child.get('history', [{}])[0].get('content', '')
                endorsed_answer = BeautifulSoup(html, 'html.parser').get_text(separator=' ', strip=True)
                break
        # if both found, exit early
        if instructor_answer is not None and endorsed_answer is not None:
            break

    return Post(number=number, subject=subject, content=content, instructor_answer=instructor_answer, endorsed_answer=endorsed_answer)