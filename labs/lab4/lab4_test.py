from src.representations.word_embedder import WordEmbedder
import numpy as np

def main():

    embedder = WordEmbedder("glove-wiki-gigaword-50")

    print("Vector for 'king':")
    king_vec = embedder.get_vector('king')
    print(king_vec)

    sim_kq = embedder.get_similarity('king', 'queen')
    sim_km = embedder.get_similarity('king', 'man')
    print("\nSimilarity (king, queen):", sim_kq)
    print("Similarity (king, man):", sim_km)

    print("\nTop 10 most similar to 'computer':")
    ms = embedder.get_most_similarity('computer', top_n=10)
    for word, score in ms:
        print(f"{word}\t{score:.4f}")

    print("\nDocument embedding for 'The queen rules the country.':")
    sent = "The queen rules the country."
    doc_vec = embedder.embed_document(sent)
    print(doc_vec)

if __name__ == "__main__":
    main()