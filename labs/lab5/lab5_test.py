from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from src.models.text_classifier import TextClassifier
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer


texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]

labels = [1, 0, 1, 0, 1, 0] # 1 for positive, 0 for negative

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size = 0.2, random_state=42)

tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer)
classifier = TextClassifier(vectorizer)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

metrics = classifier.evaluate(y_test, y_pred)
print("Evaluation metrics: ", metrics)



