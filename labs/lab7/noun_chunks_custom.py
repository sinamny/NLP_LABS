import spacy
nlp = spacy.load("en_core_web_md")

# det: mạo từ, amod: tính từ, compound: danh từ ghép
def extract_noun_chunk(token):
    modifiers = []
    for child in token.children:
        if child.dep_ in ["det", "amod", "compound"]:
            modifiers.append(child)

    modifiers_sorted = sorted(modifiers, key=lambda t: t.i) # sắp xếp theo chỉ số từ trong câu
    words = [t.text for t in modifiers_sorted] + [token.text] # ghép modifiers và danh từ chính

    return " ".join(words)

def get_noun_chunks(doc):
    chunks = []
    for token in doc:
        if token.pos_ == "NOUN":
            chunks.append(extract_noun_chunk(token))
    return chunks

doc = nlp("The big white cat slept on the warm mat.")
print(get_noun_chunks(doc))