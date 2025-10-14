import gensim.downloader as api
import numpy as np
from src.preprocessing.regex_tokenizer import RegexTokenizer

class WordEmbedder:
    def __init__(self, model_name: str = "glove-wiki-gigaword-50"):
        self.model = api.load(model_name)
        print(f"Mô hình `{model_name}` đã được tải.")
        self.vector_size = self.model.vector_size
        self.tokenizer = RegexTokenizer()

    def get_vector(self, word: str):
        if word in self.model.key_to_index:
            return self.model[word]
        else:
            print(f"Từ '{word}' không có trong mô hình (OOV).")
            return np.zeros(self.model.vector_size)
        
    def get_similarity(self, word1: str, word2: str):
        if word1 in self.model.key_to_index and word2 in self.model.key_to_index:
            return self.model.similarity(word1, word2)
        else:
            print(f"Một trong hai từ '{word1}' hoặc '{word2}' không có trong mô hình.")
            return None
        
    def get_most_similarity(self, word: str, top_n: int = 10):
        if word in self.model.key_to_index:
            return self.model.most_similar(word, topn = top_n)
        else:
            print(f"Từ '{word}' không có trong mô hình (OOV).")
            return []
        
    def embed_document(self, document: str):
        tokens = self.tokenizer.tokenize(document)
        vectors = []
        
        for token in tokens:
            if token in self.model.key_to_index:
                vectors.append(self.model[token])

        if not vectors:
            print("Không có token hợp lệ trong văn bản — trả về vector 0.")
            return np.zeros(self.vector_size)
        
        return np.mean(vectors, axis = 0)