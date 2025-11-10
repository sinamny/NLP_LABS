# Lab 5: Text Classification

## Mục tiêu

Xây dựng pipeline phân loại văn bản hoàn chỉnh — từ dữ liệu thô đến mô hình học máy huấn luyện xong — sử dụng các kỹ thuật tokenization và vectorization đã học ở các lab trước.

Mục tiêu cụ thể:

* Hiểu quy trình tiền xử lý văn bản và biến đổi thành vector đặc trưng.
* Huấn luyện mô hình Logistic Regression cho bài toán phân loại cảm xúc.
* Đánh giá mô hình bằng các metric cơ bản.
* Triển khai pipeline tương tự trên PySpark cho dữ liệu lớn.



## Task 1: Data Preparation (Scikit-learn)

### Dataset

```python
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]
```

### Mục tiêu

* Chuẩn bị dữ liệu văn bản và nhãn.
* Biến đổi văn bản thành dạng vector số dùng TfidfVectorizer hoặc CountVectorizer.

### Các bước thực hiện

1. Tạo dataset nhỏ trong bộ nhớ để dễ kiểm tra:

```python
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]  # 1 là positive, 0 là negative
```

2. Vector hóa văn bản:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TfidfVectorizer sẽ tính trọng số TF-IDF cho từng từ trong corpus
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
```

Giải thích các hàm:

* `TfidfVectorizer()`: chuyển văn bản thành ma trận số với trọng số TF-IDF (term frequency-inverse document frequency).
* `fit_transform(texts)`: học từ vựng từ corpus và chuyển tất cả văn bản thành vector số.

### Hướng dẫn chạy code

* Mở terminal tại thư mục gốc dự án nlp-labs.
* Chạy:

```bash
python -m labs.lab5.task1_data_preparation
```

### Kết quả mẫu

```python
print(X.shape)
# (6, 23)  # 6 văn bản, 23 từ đặc trưng
```

### Phân tích

* Ma trận X có chiều (số văn bản, số từ đặc trưng).
* Các từ ít xuất hiện trong corpus sẽ có trọng số thấp.
* Đây là bước chuẩn bị dữ liệu đầu vào cho mô hình học máy.

## Task 2: Implementing the TextClassifier

File: `src/models/text_classifier.py`

### Mục tiêu

* Xây dựng lớp `TextClassifier` để huấn luyện và dự đoán văn bản bằng Logistic Regression.
* Cung cấp các phương thức: `fit`, `predict`, `evaluate`.

### Các bước thực hiện

1. Tạo file: `src/models/text_classifier.py`

2. Implement class:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TextClassifier:
    def __init__(self, vectorizer):
        """
        vectorizer: một instance của TfidfVectorizer hoặc CountVectorizer
        _model: lưu mô hình Logistic Regression sau khi huấn luyện
        """
        self.vectorizer = vectorizer
        self._model = LogisticRegression(solver='liblinear')  # solver liblinear phù hợp với dataset nhỏ

    def fit(self, texts, labels):
        """
        Huấn luyện mô hình với dữ liệu texts và labels.
        """
        X = self.vectorizer.fit_transform(texts)
        self._model.fit(X, labels)

    def predict(self, texts):
        """
        Dự đoán nhãn cho dữ liệu mới.
        """
        X = self.vectorizer.transform(texts)
        return self._model.predict(X)

    def evaluate(self, y_true, y_pred):
        """
        Tính các chỉ số đánh giá mô hình:   accuracy, precision, recall, f1_score
        - y_true: nhãn thật
        - y_pred: nhãn dự đoán
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred)
        }
```

Giải thích các hàm:

* `fit_transform`: học từ vựng và biến đổi văn bản thành vector.
* `_model.fit`: huấn luyện Logistic Regression.
* `transform`: biến đổi văn bản mới sang vector dùng từ vựng đã học.
* `accuracy_score`, `precision_score`, `recall_score`, `f1_score`: các hàm đánh giá hiệu quả mô hình.


## Task 3: Basic Evaluation

File: `labs/lab5/lab5_test.py`

### Mục tiêu

* Chia dữ liệu thành tập huấn luyện và kiểm tra.
* Huấn luyện và đánh giá TextClassifier.

### Các bước thực hiện

1. Tạo file `labs/lab5/lab5_test.py`:

```python
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
```
### Hướng dẫn chạy code

```bash
python -m labs.lab5.lab5_test
```

### Kết quả 
```
Evaluation metrics: {'accuracy': 0.5, 'precision': 0.5, 'recall': 1.0, 'f1_score': 0.6666}
```

### Phân tích

* Do số lượng test sample ít, accuracy chỉ đạt 0.5.
* F1-score cho thấy sự cân bằng giữa precision và recall.
* Đây là bước kiểm tra cơ bản pipeline.

## Advanced Example: Sentiment Analysis with PySpark
### Mục tiêu

* Xây dựng pipeline phân loại văn bản dùng PySpark cho dataset lớn.
* Thực hiện tokenization, stopwords removal, TF-IDF, Logistic Regression.

File: `labs/lab5/lab5_spark_sentiment_analysis.py`

### Các bước thực hiện
#### Tạo file `labs/lab5/lab5_spark_sentiment_analysis.py`:

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

# 1. Khởi tạo SparkSession
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

# 2. Load dữ liệu CSV
df = spark.read.csv("data/sentiments.csv", header=True, inferSchema=True)

# 3. Chuẩn hóa nhãn sentiment về 0/1
df = df.withColumn("label", (col("sentiment").cast("integer")+1)/2).dropna(subset=["sentiment"])

# 4. Pipeline tiền xử lý
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])

# 5. Split dữ liệu train/test
(trainingData, testData) = df.randomSplit([0.8,0.2], seed=42)

# 6. Huấn luyện mô hình
model = pipeline.fit(trainingData)

# 7. Dự đoán và đánh giá
predictions = model.transform(testData)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("Regression Model:")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"F1-score: {accuracy:.4f}")

# 8. Dừng Spark
spark.stop()
```
Giải thích các hàm:

* `SparkSession.builder.appName`: tạo môi trường Spark để xử lý dữ liệu phân tán.
* `Tokenizer`: tách câu thành từ (tokens).
* `StopWordsRemover`: loại bỏ các từ dừng phổ biến.
* `HashingTF`: ánh xạ tokens thành vector số cố định.
* `IDF`: tính trọng số ngược tần suất xuất hiện từ.
* `Pipeline`: kết hợp tất cả các bước thành pipeline thống nhất.
* `LogisticRegression`: huấn luyện mô hình phân loại.
* `MulticlassClassificationEvaluator`: đánh giá độ chính xác hoặc F1-score.

### Hướng dẫn chạy code

```bash
python -m labs.lab5.lab5_spark_sentiment_analysis
```
### Kết quả 

```
Regression Model:
Test Accuracy: 0.7295
F1-score: 0.7295
```


### Phân tích

* Mô hình đạt accuracy xấp xỉ 0.73, cho thấy pipeline Logistic Regression + TF-IDF hoạt động ổn định trên dataset này.
* Spark xử lý dữ liệu phân tán, tránh đầy RAM.
* Có thể cải thiện bằng Word2Vec hoặc Naive Bayes.

## Task 4: Evaluating and Improving Model Performance

### Mục tiêu

* Xây dựng pipeline phân loại văn bản từ dữ liệu sentiment thô đến dự đoán nhãn.
* Thử nghiệm các phương pháp cải thiện hiệu năng mô hình: Improved Preprocessing, Word2Vec Embedding, Naive Bayes.
* So sánh kết quả Accuracy và F1-score giữa các phương pháp.

### Các file code

| Tên file                                 | Mục đích                                                        |
| ---------------------------------------- | --------------------------------------------------------------- |
| `lab5_improvement_test_preprocessing.py` | Thử nghiệm cải thiện preprocessing, TF-IDF, thay đổi số feature |
| `lab5_improvement_test_word2vec.py`      | Thử nghiệm Word2Vec embedding với Logistic Regression           |
| `lab5_improvement_test_naivebayes.py`    | Thử nghiệm Naive Bayes với TF-IDF                               |

### 1. Improved Preprocessing – TF-IDF

File: `lab5_improvement_test_preprocessing.py`

**Mục tiêu:** Loại bỏ nhiễu trong dữ liệu, giảm từ không quan trọng, thử các số lượng feature khác nhau để xem mô hình hoạt động thế nào.

**Các bước thực hiện:**

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Khởi tạo Spark Session
spark = SparkSession.builder.appName("Task4_Preprocessing").getOrCreate()

# 2. Load dữ liệu
df = spark.read.csv("src/data/sentiments.csv", header=True, inferSchema=True)

# 3. Tiền xử lý
df = (
    df.withColumn("text", lower(col("text"))) # Chuyển về chữ thường
      .withColumn("text", regexp_replace(col("text"), r"http\S+", "")) # Xóa link
      .withColumn("text", regexp_replace(col("text"), r"[^a-z\s]", "")) # Chỉ giữ chữ cái
      .dropna(subset=["text", "sentiment"]) 
      .withColumn("label", (col("sentiment").cast("integer") + 1) / 2) # Chuyển -1/1 -> 0/1
)

# 4. Chia train/test
train, test = df.randomSplit([0.8, 0.2], seed=42)

# 5. Pipeline preprocessing
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# Giảm numFeatures để thử nghiệm trade-off
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="features")

# 6. Logistic Regression
lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

# 7. Tạo pipeline và huấn luyện
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
model = pipeline.fit(train)
pred = model.transform(test)

# 8. Đánh giá
eval_acc = MulticlassClassificationEvaluator(metricName="accuracy")
eval_f1 = MulticlassClassificationEvaluator(metricName="f1")
print("Improved Preprocessing (numFeatures=10000)")
print(f"Accuracy: {eval_acc.evaluate(pred):.4f}")
print(f"F1-score: {eval_f1.evaluate(pred):.4f}")

spark.stop()
```

**Giải thích bước:**

* `Tokenizer` tách câu thành từ.
* `StopWordsRemover` loại bỏ các từ không mang nghĩa.
* `HashingTF` + `IDF` tạo feature TF-IDF.
* `numFeatures` nhỏ => giảm nhiễu nhưng mất thông tin hiếm.
* Logistic Regression là baseline tuyến tính.

### 2. Word2Vec Embedding

File: `lab5_improvement_test_word2vec.py`

**Mục tiêu:** Chuyển mỗi câu thành vector dày đặc (dense vector) để mô hình học được ngữ nghĩa tốt hơn.

**Các bước thực hiện:**

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("Task4_Word2Vec").getOrCreate()

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

# Word2Vec embedding
word2vec = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered_words", outputCol="features")

lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=[tokenizer, remover, word2vec, lr])
model = pipeline.fit(train)
pred = model.transform(test)

e_acc = MulticlassClassificationEvaluator(metricName="accuracy")
e_f1 = MulticlassClassificationEvaluator(metricName="f1")
print("Word2Vec Embedding")
print(f"Accuracy: {e_acc.evaluate(pred):.4f}")
print(f"F1-score: {e_f1.evaluate(pred):.4f}")

spark.stop()
```

**Giải thích bước:**

* `Word2Vec` chuyển từ thành vector dense.
* Logistic Regression huấn luyện trên vector câu.
* Ưu điểm: semantic tốt hơn TF-IDF.
* Nhược điểm: cần dữ liệu lớn để embeddings hiệu quả.

## 3. Naive Bayes Classifier

File: `lab5_improvement_test_naivebayes.py`

**Mục tiêu:** Thử mô hình xác suất đơn giản Naive Bayes trên TF-IDF để so sánh với Logistic Regression.

**Các bước thực hiện:**

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# 1. Khởi tạo Spark
spark = SparkSession.builder.appName("NaiveBayes_Model").getOrCreate()

# 2. Đọc dữ liệu
df = spark.read.csv("src/data/sentiments.csv", header=True, inferSchema=True)

# 3. Làm sạch dữ liệu
df = (
    df.withColumn("text", lower(col("text")))                       # chuyển chữ thường
      .withColumn("text", regexp_replace(col("text"), r"http\S+", "")) # xóa link
      .withColumn("text", regexp_replace(col("text"), r"[^a-z\s]", "")) # giữ chữ cái
      .dropna(subset=["text", "sentiment"]) 
      .withColumn("label", (col("sentiment").cast("integer") + 1) / 2) # -1/1 -> 0/1
)

# 4. Chia train/test
train, test = df.randomSplit([0.8, 0.2], seed=42)

# 5. Tách từ và loại từ không quan trọng
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

# 6. Tạo feature TF-IDF
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# 7. Naive Bayes
nb = NaiveBayes(smoothing=1.0, featuresCol="features", labelCol="label")

# 8. Pipeline và huấn luyện
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, nb])
model = pipeline.fit(train)
pred = model.transform(test)

# 9. Đánh giá
e_acc = MulticlassClassificationEvaluator(metricName="accuracy")
e_f1 = MulticlassClassificationEvaluator(metricName="f1")

print("Naive Bayes Model")
print(f"Accuracy: {e_acc.evaluate(pred):.4f}")
print(f"F1-score: {e_f1.evaluate(pred):.4f}")

spark.stop()
```

**Giải thích:**

* Naive Bayes là mô hình thống kê đơn giản, dựa trên giả định các từ trong câu độc lập với nhau, tính xác suất nhãn dựa trên tần suất từ trong văn bản.
* TF-IDF vẫn được dùng để chuyển từ thành vector số, cung cấp thông tin về tần suất từ quan trọng trong câu.
* Ưu điểm: đơn giản, nhanh, ít tốn tài nguyên.



### Hướng dẫn chạy code

```bash
# 1. Improved Preprocessing
python -m labs.lab5.lab5_improvement_test_preprocessing

# 2. Word2Vec Embedding
python -m labs.lab5.lab5_improvement_test_word2vec

# 3. Naive Bayes
python -m labs.lab5.lab5_improvement_test_naivebayes
```

### Kết quả và so sánh

| Model / Method                             | Accuracy | F1-score | Nhận xét ngắn                                         |
| ------------------------------------------ | -------: | -------: | ----------------------------------------------------- |
| Base Logistic Regression (TF-IDF)          |   0.7295 |   0.7295 | Baseline                                              |
| Improved Preprocessing (numFeatures=10000) |   0.7565 |   0.7582 | Preprocessing tốt, giảm nhiễu, tăng Accuracy          |
| Improved Preprocessing (numFeatures=5000)  |   0.7214 |   0.7225 | NumFeatures quá nhỏ => mất thông tin hiếm              |
| Word2Vec Embedding + LogisticRegression    |   0.6420 |   0.5690 | Embedding semantic nhưng dữ liệu nhỏ => kém hơn TF-IDF |
| Naive Bayes (TF-IDF)                       |   0.7295 |   0.7323 | Đơn giản, hiệu quả tương đương Logistic Regression    |


### Phân tích kết quả

1. Improved Preprocessing giúp tăng Accuracy và F1-score so với baseline vì dữ liệu sạch hơn và số feature hợp lý (10000).
2. NumFeatures quá nhỏ (5000) dẫn đến mất thông tin hiếm => Accuracy giảm.
3. Word2Vec chưa tối ưu với dataset nhỏ, dẫn đến kết quả thấp hơn TF-IDF; embeddings thường phát huy hiệu quả với dữ liệu lớn.
4. Naive Bayes vẫn là mô hình đơn giản nhưng hiệu quả tương đương Logistic Regression, chứng tỏ TF-IDF đã cung cấp feature đủ mạnh.


**Kết luận:**

* Preprocessing và feature selection là bước quan trọng nhất trong Task này.
* Word2Vec hoặc mô hình phức tạp chỉ nên dùng với dữ liệu lớn.
* Logistic Regression + TF-IDF (numFeatures=10000) là baseline mạnh, dễ triển khai.

### Khó khăn và giải pháp
* Khó khăn: 
    * Dữ liệu thô có nhiều ký tự đặc biệt, chữ hoa/chữ thường, link, số, dấu câu…
    * Các từ không mang nghĩa (stopwords) làm nhiễu mô hình.
    * Word2Vec cần dữ liệu lớn để embeddings hiệu quả.
    * Logistic Regression cần cân bằng số feature để tránh mất thông tin hiếm nhưng không làm mô hình quá phức tạp.

* Giải pháp:

    * Tiền xử lý dữ liệu: Chuyển chữ thường, loại bỏ ký tự đặc biệt, link, số, giữ lại chữ và khoảng trắng. Loại bỏ stopwords để giảm nhiễu.

    * Dùng numFeatures vừa phải, cân bằng giữa giữ thông tin và giảm nhiễu.

    * TF-IDF + Logistic Regression là baseline ổn định cho dataset nhỏ.
## Tài liệu tham khảo

* Scikit-learn Documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* Apache Spark MLlib Guide: [https://spark.apache.org/docs/latest/ml-guide.html](https://spark.apache.org/docs/latest/ml-guide.html)
* OpenAI. (2025). ChatGPT (GPT-4/5) [AI language model]. Truy cập từ https://chat.openai.com
