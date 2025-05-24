from bs4 import BeautifulSoup #to parse html

class Post:
    #storing most important data as attributes. more can be added later from the raw post data extracted by piazza-api
    def __init__(self, number, has_instructor_answer, has_instructor_endorsement, subject, content):
        self.number = number
        self.has_instructor_answer = has_instructor_answer
        self.has_instructor_endorsement = has_instructor_endorsement
        self.subject = subject
        self.content = content

def create_post_from_api(raw): #helper function. raw data is stored as a large dict with some nested lists and dicts
    number = raw['nr'] #post number
    
    #checking if there is an instructors' answer. this is under the children key 
    has_i_answer = False
    for child in raw.get('children', []):
        if child.get('type') == 'i_answer':
            has_i_answer = True
            break
    
    #checking if an answer is endorsed by an instructor/professor
    has_i_endorse = False
    for child in raw.get('children', []):
        endorsements = child.get('tag_endorse', []) + child.get('tag_good', []) #professor is found in tag_good, instructor is found in tag_endorse. we will check both keys
        for e in endorsements:
            if e.get('role') in ('instructor', 'professor'): #if role returned by e.get('role') is in the tuple ('instructor', 'professor'), then there is an instructor/professor endorsement
                has_i_endorse = True
                break
        if has_i_endorse:
            break

    #this is html, not plain text. should parse and extract plain text
    subj_html = raw['history'][0].get('subject', '')
    cont_html = raw['history'][0].get('content', '')

    #parsing html to extract only plain text
    subject = BeautifulSoup(subj_html, 'html.parser').get_text(separator=' ', strip=True)
    content = BeautifulSoup(cont_html, 'html.parser').get_text(separator=' ', strip=True)

    return Post(number, has_i_answer, has_i_endorse, subject, content) #build and return the Post object