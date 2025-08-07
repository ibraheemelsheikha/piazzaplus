import logging
logging.disable(logging.WARNING)

from langchain.text_splitter import NLTKTextSplitter
import nltk
nltk.download("punkt", quiet=True)

splitter_2 = NLTKTextSplitter(chunk_size=2, chunk_overlap=1)
splitter_3 = NLTKTextSplitter(chunk_size=3, chunk_overlap=2)

chunks_2 = splitter_2.split_text("Time Complexity Are we expected to know time complexities and their examples? eg O(1) constant time, O(n) linear time, O(n^2) quadratic time, etc. It was briefly covered in lecture so just wanted to double check for the MCQs and TF. Thanks!")
chunks_3 = splitter_3.split_text("Time Complexity Are we expected to know time complexities and their examples? eg O(1) constant time, O(n) linear time, O(n^2) quadratic time, etc. It was briefly covered in lecture so just wanted to double check for the MCQs and TF. Thanks!")

print(chunks_2 + chunks_3)
