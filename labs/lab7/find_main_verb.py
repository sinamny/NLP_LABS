import spacy
nlp = spacy.load("en_core_web_md")

def find_main_verb(doc):
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            return token
    return None

doc = nlp("The quick brown fox jumps over the lazy dog.")
print("Main verb:", find_main_verb(doc))
