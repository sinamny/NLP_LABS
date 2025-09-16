from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

class Vectorizer(ABC):
    @abstractmethod
    def fit(self, corpus: List[str]):
        """Học vocabulary từ corpus"""
        pass

    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[int]]:
        """Chuyển danh sách documents thành vectors"""
        pass

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """fit rồi transform"""
        self.fit(corpus)
        return self.transform(corpus)