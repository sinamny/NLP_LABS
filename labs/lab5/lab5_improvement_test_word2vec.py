# lab5_improvement_word2vec.py
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from gensim.models import Word2Vec

# Dataset nhỏ
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Tokenization đơn giản
X_train_tokens = [x.lower().split() for x in X_train]
X_test_tokens = [x.lower().split() for x in X_test]

# Train Word2Vec
w2v_model = Word2Vec(sentences=X_train_tokens, vector_size=100, window=5, min_count=1, workers=1)

def sentence_vector(tokens, model):
    vecs = [model.wv[word] for word in tokens if word in model.wv]
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

X_train_vec = np.array([sentence_vector(tokens, w2v_model) for tokens in X_train_tokens])
X_test_vec = np.array([sentence_vector(tokens, w2v_model) for tokens in X_test_tokens])

# Logistic Regression
clf = LogisticRegression(solver='liblinear')
clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test_vec)

# Evaluation
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred)
}

print("Word2Vec Embedding + Logistic Regression")
print(metrics)

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, lower, regexp_replace
# from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml import Pipeline
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# spark = SparkSession.builder.appName("Task4_Word2Vec").getOrCreate()

# df = spark.read.csv("src/data/sentiments.csv", header=True, inferSchema=True)
# df = (
#     df.withColumn("text", lower(col("text")))
#       .withColumn("text", regexp_replace(col("text"), r"http\S+", ""))
#       .withColumn("text", regexp_replace(col("text"), r"[^a-z\s]", ""))
#       .dropna(subset=["text", "sentiment"])
#       .withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
# )

# train, test = df.randomSplit([0.8, 0.2], seed=42)

# tokenizer = Tokenizer(inputCol="text", outputCol="words")
# remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# # Word2Vec embedding
# word2vec = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered_words", outputCol="features")

# lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

# pipeline = Pipeline(stages=[tokenizer, remover, word2vec, lr])
# model = pipeline.fit(train)
# pred = model.transform(test)

# e_acc = MulticlassClassificationEvaluator(metricName="accuracy")
# e_f1 = MulticlassClassificationEvaluator(metricName="f1")
# print("Word2Vec Embedding")
# print(f"Accuracy: {e_acc.evaluate(pred):.4f}")
# print(f"F1-score: {e_f1.evaluate(pred):.4f}")

# spark.stop()
