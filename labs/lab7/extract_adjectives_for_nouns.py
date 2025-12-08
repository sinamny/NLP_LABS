import spacy
nlp = spacy.load("en_core_web_md")

text = "The big, fluffy white cat is sleeping on the warm mat."
doc = nlp(text)

for token in doc:
    # Chỉ tìm các danh từ
    if token.pos_ == "NOUN":
        adjectives = []
        # Tìm các tính từ bổ nghĩa (amod) trong các con của danh từ
        for child in token.children:
            if child.dep_ == "amod":
                adjectives.append(child.text)

            if adjectives:
                print(f"Danh từ '{token.text}' được bổ nghĩa bởi các tính từ: {adjectives}")