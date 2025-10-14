from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.functions import lower, regexp_replace

def main():
    print("Khởi tạo SparkSession.")
    spark = SparkSession.builder.appName("Spark Word2Vec Demo").master("local[*]").getOrCreate()

    print("-" * 10)
    print("Đọc dữ liệu")
    df = spark.read.json("src/data/c4-train.00000-of-01024-30K.json")
    df = df.select("text").dropna()
    print(f"Số dòng đọc được: {df.count()}")

    print("-" * 10)
    print("Tiền xử lý văn bản")
    df = df.withColumn("text", lower(regexp_replace("text", "[^a-zA-Z\\s]", "")))

    print("-" * 10)
    print("Tokenization")
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    words_data = tokenizer.transform(df)

    print("-" * 10)
    print("Huấn luyện mô hình Word2Vec (Skip-gram)")
    word2vec = Word2Vec(
        vectorSize=100,
        minCount=5,
        inputCol="words",
        outputCol="vector"
    )
    model = word2vec.fit(words_data)

    # print("-" * 10)
    # print("\nVector của 'computer':")
    # vectors_df = model.getVectors()
    # vectors_df.filter(vectors_df["word"] == "computer").show(truncate=False)

    print("\nTìm các từ tương tự 'computer'")
    synonyms = model.findSynonyms("computer", 10)
    synonyms.show(truncate=False)


    spark.stop()
    print("\nHoàn thành huấn luyện Spark Word2Vec")

if __name__ == "__main__":
    main()
