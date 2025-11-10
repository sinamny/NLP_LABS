from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("NaiveBayes_Model").getOrCreate()

df = spark.read.csv("src/data/sentiments.csv", header=True, inferSchema=True)
df = (
    df.withColumn("text", lower(col("text")))
      .withColumn("text", regexp_replace(col("text"), r"http\S+", ""))
      .withColumn("text", regexp_replace(col("text"), r"[^a-z\s]", ""))
      .dropna(subset=["text", "sentiment"])
      .withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
)

train, test = df.randomSplit([0.8, 0.2], seed=42)

tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
nb = NaiveBayes(smoothing=1.0, featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, nb])
model = pipeline.fit(train)
pred = model.transform(test)

# Evaluate
e_acc = MulticlassClassificationEvaluator(metricName="accuracy")
e_f1 = MulticlassClassificationEvaluator(metricName="f1")

print("Naive Bayes Model")
print(f"Accuracy: {e_acc.evaluate(pred):.4f}")
print(f"F1-score: {e_f1.evaluate(pred):.4f}")

spark.stop()
