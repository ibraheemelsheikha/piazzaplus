import json
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

# custom embeddings class that uses SentenceTransformer under the hood
class SentenceTransformerEmbeddings(Embeddings):
    # initialize with the name of the pretrained SentenceTransformer model
    def __init__(self, model_name: str):
        # load the transformer model by name
        self.model = SentenceTransformer(model_name)
    
    # embed a list of document texts into vectors
    def embed_documents(self, texts: list[str]):
        # use model to encode texts, returns numpy arrays
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        # convert numpy arrays to regular Python lists
        return embeddings.tolist()
    
    # embed a single query string into a numerical vector
    def embed_query(self, text: str):
        # encode the query text the same way as documents
        embedding = self.model.encode(text, convert_to_tensor=False)
        # return the embedding as a Python list
        return embedding.tolist()

# instantiate the embedding model
embedding_model = SentenceTransformerEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

# read the stored .json file containing post subject and content
with open('posts.json', 'r', encoding='utf-8') as f:
    data = json.load(f)  # data is a dict mapping post post ID to post info dicts

# list of combined "subject + content" strings, with optional answers
combined_posts = []
for post in data.values():
    # extract subject, defaulting to empty string if missing, and strip whitespace
    subject = post.get('subject', '').strip()
    # extract content and strip whitespace
    content = post.get('content', '').strip()
    # build the base text: subject + content
    text = subject + ' ' + content
    # include instructor answer if present
    instr_ans = post.get('instructor_answer', '').strip()
    if instr_ans:
        text += ' ' + instr_ans
    # include endorsed student answer if present
    end_ans = post.get('endorsed_answer', '').strip()
    if end_ans:
        text += ' ' + end_ans
    combined_posts.append(text)

# wrap each combined post string in a Document object for the vector store,
# including metadata for post number and subject for later retrieval
documents = []
for post_id, post in data.items():
    subject = post.get('subject', '').strip()
    content = post.get('content', '').strip()
    text = subject + ' ' + content
    # attach post_id and subject in metadata
    document = Document(page_content=text, metadata={'post_id': post_id, 'subject': subject})
    documents.append(document)

# create a Chroma vector database from the list of Document objects
# chroma will use embedding_model to generate embeddings
vector_database = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model
)

query = input("Enter a query here: ")  # blocks execution until input is provided

# similarity search for the top 10 documents matching the query, retrieving scores. uses cosine similarity
results = vector_database.similarity_search_with_score(query, k=10)

# Iterate over the results and print each matched document's post number, subject, and score
for i, (result, score) in enumerate(results, start=1):
    pid = result.metadata['post_id']
    subj = result.metadata['subject']
    print(f"Result {i}: Post #{pid} — {subj} (score: {score:.4f})")
