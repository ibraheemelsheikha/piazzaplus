from rake_nltk import Rake
import nltk
import re

# ensure NLTK data is available
nltk.download('stopwords')
nltk.download('punkt')

# your list of questions, stripped of the "(#…)" parts
questions = [
    "How can I create a github account?",
    "How do we submit Lab 0?",
    "My output looks the same as the test cases but it still says fail",
    "Should we be rounding in Lab 1?",
    "Do we need to pass hidden test cases for full marks?",
    "What happens if we miss the lab submission deadline?",
    "What’s the difference between truncating, rounding down, and flooring a number?",
    "Shouldn’t a bool variable take up only 1 bit?",
    "Why do we need to use % for printf?",
    "What’s the difference between Python and C?",
    "How come you’re allowed to divide by 0 in C?",
    "Should I memorize ASCII codes for exams?",
    "Can inputs be negative in Lab 2 Part 3?",
    "What’s the difference between camel case and snake case?",
    "If I use strcat or strncat, will the result always be null-terminated?",
    "Does the order in which we declare and implement functions matter?",
    "Does the math library use radians or degrees?",
    "What is the max input for lab 3 part 3?",
    "What is the midterm scope?",
    "What if the user enters 0 as their first input in lab 4 part 2?",
    "What’s the point of returning 0 in main?",
    "Is it necessary to always initialize variables?",
    "What if the user enters something invalid in lab?",
    "What is the scope of a variable when you declare it?",
    "What is the difference between <= and < in loop conditions?",
    "What is a seed when using the rand function?",
    "Can you scanf an entire array?",
    "Can we alter function prototypes or define new functions for lab 7?",
    "What does it mean to free dynamically allocated memory?",
    "How do we measure time for lab 8 part 2?",
    "How are we being marked for lab 8?",
    "What is the difference between int* var and int *var?",
    "Are binary search trees in scope?",
    "Is sorting in scope for the exam?",
    "Can I set to NULL before freeing?",
    "What is a seg fault, and how can I find when/where it happens?",
    "How can I safely allocate memory inside a function?",
    "When should I use recursion vs. loops, and can I combine them?",
    "How many different ways are there of declaring a string in C, and what is the difference between them?",
    "Can you use scanf to take input into an array without a seg fault?"
]

r = Rake()

for idx, text in enumerate(questions, start=1):
    r.extract_keywords_from_text(text)
    phrases = r.get_ranked_phrases()
    # flatten into individual words and strip out any punctuation
    words = []
    for phrase in phrases:
        clean = re.sub(r'[^A-Za-z0-9 ]+', '', phrase)
        words.extend(clean.split())
    print(f"question {idx}:")
    print(" ".join(words))
    print("\n")
