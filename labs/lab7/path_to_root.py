import spacy
nlp = spacy.load("en_core_web_md")

def get_path_to_root(token):
    path = [token]
    while token.dep_ != "ROOT":
        token = token.head
        path.append(token)
    return path

doc = nlp("The quick fox jumps over the lazy dog.")

token = doc[7] # dog
path = get_path_to_root(token)

print([t.text for t in path])