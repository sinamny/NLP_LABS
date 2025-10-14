import os
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

DATA_PATH = "src/data/UD_English-EWT/en_ewt-ud-train.txt"
SAVE_PATH = "results/word2vec_ewt.model"


def stream_sentences(file_path):
    with open(file_path, "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            yield simple_preprocess(line)

def main():
    print("Đọc dữ liệu huấn luyện.")
    sentences = list(stream_sentences(DATA_PATH))
    print(f"Số câu đọc được: {len(sentences)}")

    print("\nHuấn luyện mô hình Word2Vec (Skip-gram)")
    model = Word2Vec(sentences = sentences, vector_size = 100, window = 5, min_count = 2, workers = 4, sg = 1)

    print("\nLưu mô hình")
    os.makedirs("results", exist_ok=True)
    model.save(SAVE_PATH)
    print(f"Mô hình đã được lưu tại {SAVE_PATH}.")

    print("\nKiểm tra mô hình:")
    try:
        print("Top 10 từ tương tự 'computer':")
        print(f"{'Word':<15}{'Similarity':>10}")
        print("-" * 25)

        for w, s in model.wv.most_similar("computer", topn=10):
            print(f"{w:<15}{s:>10.4f}")
    except KeyError:
        print("Từ 'computer' không có trong tập từ.")

    try:
        sim = model.wv.similarity("king", "queen")
        print(f"Similarity(king, queen): {sim:.4f}")
    except KeyError:
        print("Một trong hai từ 'king' hoặc 'queen' không có trong từ điển!")

if __name__ == "__main__":
    main()