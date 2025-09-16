import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer


def run_lab2_tests():
    print("Lab 2: Count Vectorization\n")

    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    X = vectorizer.fit_transform(corpus)

    print("Vocabulary:")
    print(vectorizer.vocabulary_)
    print("\nDocument-Term Matrix:")
    for row in X:
        print(row)


if __name__ == "__main__":
    run_lab2_tests()
