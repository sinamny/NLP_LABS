import re
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"([.,!?])", r" \1 ", text)
        tokens = text.split()
        return tokens