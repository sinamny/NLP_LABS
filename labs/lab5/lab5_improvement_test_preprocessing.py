# lab5_improvement_preprocessing.py
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.models.text_classifier import TextClassifier

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

vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', min_df=1, max_df=0.9)


# Train & Evaluate
classifier = TextClassifier(vectorizer)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
metrics = classifier.evaluate(y_test, y_pred)

print("Improved Preprocessing (TF-IDF + Stopwords)")
print(metrics)


# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, lower, regexp_replace
# from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml import Pipeline
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# spark = SparkSession.builder.appName("Task4_Preprocessing").getOrCreate()

# data_path = "src/data/sentiments.csv"
# df = spark.read.csv(data_path, header=True, inferSchema=True)

# # Loại nhiễu
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

# # Giảm nhiễu bằng cách giảm numFeatures
# hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=5000)
# idf = IDF(inputCol="raw_features", outputCol="features")

# lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

# pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
# model = pipeline.fit(train)
# pred = model.transform(test)

# eval_acc = MulticlassClassificationEvaluator(metricName="accuracy")
# eval_f1 = MulticlassClassificationEvaluator(metricName="f1")
# print("Improved Preprocessing")
# print(f"Accuracy: {eval_acc.evaluate(pred):.4f}")
# print(f"F1-score: {eval_f1.evaluate(pred):.4f}")

# spark.stop()
