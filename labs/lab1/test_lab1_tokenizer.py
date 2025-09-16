from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data

def run_lab1_tests():
    print("Lab 1: Text Tokenization")
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    # Câu test
    sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!"
    ]

    # Task 1, 2
    for sent in sentences:
        print(f"\nText: {sent}")
        print(f"SimpleTokenizer: {simple_tokenizer.tokenize(sent)}")
        print(f"RegexTokenizer: {regex_tokenizer.tokenize(sent)}")

    # Task 3: Dataset
    dataset_path = "src/data/UD_English-EWT/en_ewt-ud-train.txt"
    try:
        raw_text = load_raw_text_data(dataset_path)
        sample_text = raw_text[:500]
        print("\nTokenizing Sample Text from UD_English-EWT")
        print(f"Original Sample: {sample_text[:100]}...\n")
        print(f"SimpleTokenizer (first 20): {simple_tokenizer.tokenize(sample_text)[:20]}")
        print(f"RegexTokenizer  (first 20): {regex_tokenizer.tokenize(sample_text)[:20]}")
    except FileNotFoundError:
        print("\nDataset file not found! Please put it in data/UD_English-EWT/")

if __name__ == "__main__":
    run_lab1_tests()
