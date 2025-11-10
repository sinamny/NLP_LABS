from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

# Tải dữ liệu
data_path = "src/data/sentiments.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Chuyển nhãn thành 0, 1
df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)
df = df.dropna(subset="sentiment")

trainingData, testData = df.randomSplit([0.8, 0.2], seed=42)

tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# Hash -> word_ids
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)

# IDF: giảm trọng số của những tữ xuất hiện quá thường xuyên
idf = IDF(inputCol="raw_features", outputCol="features")

lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])
model = pipeline.fit(trainingData)
predictions = model.transform(testData)

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
f1 = evaluator.evaluate(predictions)

print("Regression Model:")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")







