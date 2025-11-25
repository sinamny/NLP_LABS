# **Lab 5: Phân loại Văn bản với Mạng Nơ-ron Hồi quy (RNN/LSTM)**
## **1. Mục tiêu**
- Hiểu rõ hạn chế của các mô hình phân loại văn bản truyền thống (Bag-of-Words, Word2Vec trung bình).
- Nắm vững kiến trúc và luồng hoạt động của pipeline sử dụng RNN/LSTM cho bài toán phân loại văn bản.
- Tự tay xây dựng, huấn luyện và so sánh hiệu năng giữa các mô hình:
  1. TF-IDF + Logistic Regression (Baseline 1)
  2. Word2Vec (vector trung bình) + Dense Layer (Baseline 2)
  3. Embedding Layer (pre-trained) + LSTM
  4. Embedding Layer (học từ đầu) + LSTM
- Phân tích và đánh giá sức mạnh của mô hình chuỗi trong việc “hiểu” ngữ cảnh của câu.

## **2. Hướng dẫn chạy code**
### **2.1. Cấu trúc thư mục chính**

```
nlp-labs/
│
├── labs/
│   ├── lab1/                     # Lab 1: Tokenizer
│   ├── lab2/                     # Lab 2: Vectorizer
│   ├── lab4/                     # Lab 4: Word embeddings
│   ├── lab5/                     # Lab 5: Text Classification
│   ├── lab5_2/                   # Lab 5: Giới thiệu về RNNs và các bài toán
│   │   ├── lab5_pytorch_intro.py
│   │   ├── lab5_rnns_text_classification.py # Mã nguồn chính cho RNN Text Classification
│   │   ├── lab5_rnn_for_ner.py   
│   │   └── lab5_rnn_pos_tagging.py
│   ├── lab6/                     # Lab 6: Giới thiệu Transformer
│   └── __init__.py
```

> **Chú thích:**
>
> * Tất cả mã nguồn của Lab 5 (bao gồm RNN for NER) nằm trong `labs/lab5_2`.
> * File chính chạy trực tiếp: `lab5_rnns_text_classification.py`.

### **2.2. Cài đặt môi trường (sử dụng `requirements.txt`)**

1. Tạo môi trường Python (Python ≥ 3.10):

```bash
python -m venv nlp-lab-env
source nlp-lab-env/bin/activate   # Linux/Mac
nlp-lab-env\Scripts\activate      # Windows
```

2. Cài đặt tất cả thư viện từ `requirements.txt`:

```bash
pip install -r requirements.txt
```
### **2.3. Chạy Lab 5: RNN Text Classification**
Tất cả mã nguồn được đặt trong:

```
labs/lab5_2/lab5_rnns_text_classification.py
```

```bash
# Mở terminal tại thư mục gốc của dự án nlp-labs
cd nlp-labs

# Chạy Lab 5
 python -m labs.lab5_2.lab5_rnns_text_classification
```

## **3. Bộ dữ liệu**
- Sử dụng tập dữ liệu hwu.tar.gz, gồm các câu truy vấn người dùng và nhãn intent tương ứng.
- Chia thành 3 tập: `train`, `validation`, `test`.

```python
import pandas as pd

df_train = pd.read_csv('src/data/hwu/train.csv', quotechar='"')
df_val   = pd.read_csv('src/data/hwu/val.csv', quotechar='"')
df_test  = pd.read_csv('src/data/hwu/test.csv', quotechar='"')

print("Train shape:", df_train.shape)
print("Validation shape:", df_val.shape)
print("Test shape:", df_test.shape)
```

## **4. Tiền xử lý nhãn (Label Encoding)**

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(pd.concat([df_train['intent'], df_val['intent'], df_test['intent']]))

y_train = le.transform(df_train['intent'])
y_val   = le.transform(df_val['intent'])
y_test  = le.transform(df_test['intent'])
num_classes = len(le.classes_)
```

> Giải thích: `LabelEncoder` chuyển nhãn intent từ dạng text sang dạng số để mô hình ML/DL có thể xử lý.

## **4. Các bước thực hiện**
### **Task 1: Pipeline TF-IDF + Logistic Regression**
- Mục tiêu: Tạo baseline đơn giản để so sánh với mô hình sâu.

- Đặc điểm:
    - Không hiểu thứ tự từ
    - Dựa vào tần suất, phù hợp với văn bản ngắn
    - Dễ bị nhầm khi có phủ định / cấu trúc phức tạp

* Mô hình baseline 1: Biểu diễn văn bản bằng TF-IDF, dự đoán với Logistic Regression.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, log_loss, f1_score

# Tạo pipeline TF-IDF + Logistic Regression
tfidf_lr_pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000),  # Chỉ dùng 5000 từ phổ biến nhất
    LogisticRegression(max_iter=1000)    # Logistic Regression tối đa 1000 vòng lặp
)

# Huấn luyện trên tập train
tfidf_lr_pipeline.fit(df_train['text'], y_train)

# Dự đoán trên tập test
y_pred_tfidf = tfidf_lr_pipeline.predict(df_test['text'])

# Đánh giá mô hình
print(classification_report(y_test, y_pred_tfidf, target_names=le.classes_))

# Tính Test Loss (Log Loss) và Macro F1
y_proba_tfidf = tfidf_lr_pipeline.predict_proba(df_test['text'])
loss_tfidf = log_loss(y_test, y_proba_tfidf, labels=list(range(num_classes)))
f1_tfidf = f1_score(y_test, y_pred_tfidf, average='macro')
```

> Giải thích: TF-IDF bỏ qua thứ tự từ, chỉ dựa vào tần suất từ, do đó khó xử lý các câu có phủ định hoặc ngữ cảnh phức tạp.

### **Task 2: Word2Vec + Dense Layer**
- Mục tiêu: Sử dụng vector dense để biểu diễn từ. Dùng mạng nơ-ron đơn giản để phân loại
* Mỗi từ được biểu diễn bằng vector Word2Vec. 
* Vector của câu = trung bình các vector từ. => Mất thứ tự 
* Mạng Dense Layer dự đoán nhãn.

```python
from gensim.models import Word2Vec
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Huấn luyện mô hình Word2Vec trên dữ liệu train
sentences = [text.split() for text in df_train['text']]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Hàm chuyển câu thành vector trung bình
def sentence_to_avg_vector(text, model):
    words = text.split()
    # Lấy vector của các từ có trong vocab
    vecs = [model.wv[w] for w in words if w in model.wv]
    if len(vecs) > 0:
        return np.mean(vecs, axis=0)  # Trung bình các vector từ
    else:
        return np.zeros(model.vector_size)  # Trường hợp câu rỗng hoặc không có từ trong vocab

# Tạo dữ liệu train/val/test dạng vector trung bình
X_train_avg = np.array([sentence_to_avg_vector(text, w2v_model) for text in df_train['text']])
X_val_avg   = np.array([sentence_to_avg_vector(text, w2v_model) for text in df_val['text']])
X_test_avg  = np.array([sentence_to_avg_vector(text, w2v_model) for text in df_test['text']])

# Xây dựng mạng Dense Layer
model_dense = Sequential([
    Dense(128, activation='relu', input_shape=(w2v_model.vector_size,)),  # 128 neurons, ReLU
    Dropout(0.5),  # Dropout 50% để tránh overfitting
    Dense(num_classes, activation='softmax')  # Output softmax cho đa lớp
])

# Compile & huấn luyện
model_dense.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_dense.fit(X_train_avg, y_train, validation_data=(X_val_avg, y_val), epochs=10, batch_size=32)

# Dự đoán & đánh giá
y_pred_dense = np.argmax(model_dense.predict(X_test_avg), axis=1)
print(classification_report(y_test, y_pred_dense, target_names=le.classes_))

# Test Loss & Macro F1
loss_w2v, acc_w2v = model_dense.evaluate(X_test_avg, y_test, verbose=0)
f1_w2v = f1_score(y_test, y_pred_dense, average='macro')

```

> Giải thích: Mô hình này có vector dày đặc (dense vector), nhưng vẫn bỏ qua thứ tự từ trong câu.

### **Task 3: Embedding Pre-trained + LSTM**

* Sử dụng Word2Vec đã huấn luyện làm **pre-trained embedding**.
* LSTM xử lý chuỗi, giữ thông tin về thứ tự từ và ngữ cảnh.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping

# Tokenizer + padding
tokenizer = Tokenizer(num_words=5000, oov_token="<UNK>")  # max 5000 từ, <UNK> cho từ lạ
tokenizer.fit_on_texts(df_train['text'])
X_train_seq = tokenizer.texts_to_sequences(df_train['text'])
X_val_seq   = tokenizer.texts_to_sequences(df_val['text'])
X_test_seq  = tokenizer.texts_to_sequences(df_test['text'])
max_len = 50  # Chiều dài tối đa của câu
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_val_pad   = pad_sequences(X_val_seq, maxlen=max_len, padding='post')
X_test_pad  = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Tạo ma trận embedding từ Word2Vec
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = w2v_model.vector_size
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# LSTM với pre-trained embeddings
lstm_pretrained = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],  # Khởi tạo bằng Word2Vec
        input_length=max_len,
        trainable=False  # Không update embeddings
    ),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),  # LSTM layer
    Dense(num_classes, activation='softmax')  # Output softmax
])

# Compile & EarlyStopping để tránh overfitting
lstm_pretrained.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lstm_pretrained.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val),
                    epochs=10, batch_size=32, callbacks=[early_stop])

# Dự đoán & đánh giá
y_pred_lstm_pre = np.argmax(lstm_pretrained.predict(X_test_pad), axis=1)
loss_lstm_pre, acc_lstm_pre = lstm_pretrained.evaluate(X_test_pad, y_test, verbose=0)
f1_lstm_pre = f1_score(y_test, y_pred_lstm_pre, average='macro')

```

> Giải thích: Pre-trained embeddings giúp mô hình hiểu nghĩa của từ tốt hơn ngay từ đầu; LSTM giữ được thông tin về thứ tự và ngữ cảnh của câu.

### **Task 4: Embedding học từ đầu + LSTM**

* Embedding layer được học từ đầu, không dùng ma trận pre-trained.

```python
lstm_scratch = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len, trainable=True),  # Học embeddings từ dữ liệu
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(num_classes, activation='softmax')
])
lstm_scratch.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lstm_scratch.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val),
                 epochs=10, batch_size=32, callbacks=[early_stop])

# Dự đoán & đánh giá
y_pred_lstm_scratch = np.argmax(lstm_scratch.predict(X_test_pad), axis=1)
loss_lstm_scratch, acc_lstm_scratch = lstm_scratch.evaluate(X_test_pad, y_test, verbose=0)
f1_lstm_scratch = f1_score(y_test, y_pred_lstm_scratch, average='macro')

```

> Giải thích: Mô hình tự học embeddings từ dữ liệu, có thể thích ứng tốt với dữ liệu chuyên biệt nhưng cần nhiều epoch để hội tụ.

### Task 5: Đánh giá, so sánh và phân tích**
#### **1. So sánh định lượng**
 Bảng so sánh hiệu năng 4 pipeline

| Pipeline                        | F1-score (Macro) | Test Loss  |
|---------------------------------|-----------------|------------|
| TF-IDF + Logistic Regression     | 0.8353          | 1.0502     |
| Word2Vec Avg + Dense             | 0.0787          | 3.4420     |
| Embedding Pre-trained + LSTM     | 0.0408          | 3.4836     |
| Embedding Scratch + LSTM         | 0.0005          | 4.1317     |

**Nhận xét:**
1. TF-IDF + Logistic Regression đạt F1 macro cao nhất và loss thấp nhất. Điều này cho thấy rằng với tập dữ liệu hiện tại, mô hình truyền thống TF-IDF + LR vẫn hoạt động tốt, đặc biệt khi dữ liệu không quá lớn và không cần hiểu ngữ cảnh phức tạp.

2. Word2Vec trung bình + Dense Layer có F1 rất thấp (~0.0787) và loss cao (~3.44). Nguyên nhân chính là trung bình vector từ bỏ qua thứ tự từ và ngữ cảnh, nên mô hình không phân biệt tốt các lớp.

3. LSTM với embeddings pre-trained cũng có F1 thấp (~0.041) và loss cao (~3.48), mặc dù lý thuyết mạnh hơn TF-IDF. Nguyên nhân phổ biến:

   * Tập dữ liệu quá nhỏ để LSTM học tốt.
   * Chiều dài câu giới hạn (max_len=50) có thể cắt mất thông tin.
   * Embedding pre-trained không phù hợp hoàn toàn với ngữ liệu chuyên biệt.

4. LSTM học embedding từ đầu tệ nhất (~0.0005 F1), loss cao nhất (~4.13). Do mô hình khởi tạo embeddings từ đầu cần rất nhiều dữ liệu để hội tụ, mà tập dữ liệu nhỏ nên không học được representations tốt.

#### **2. Phân tích định tính**

```python
examples = [
    "sorry but i think you've got that not right.",
    "is starbucks stock up or down from last quarter",
   "find new trump articles but not from fox news",
]

def predict_all(texts):
    # Dự đoán bằng TF-IDF + LR
    pred_tfidf = tfidf_lr_pipeline.predict(texts)
    
    # Dự đoán bằng Word2Vec trung bình + Dense
    X_ex_avg = np.array([sentence_to_avg_vector(t, w2v_model) for t in texts])
    pred_w2v = np.argmax(model_dense.predict(X_ex_avg), axis=1)
    
    # Dự đoán bằng LSTM pre-trained
    seqs = tokenizer.texts_to_sequences(texts)
    pad = pad_sequences(seqs, maxlen=max_len, padding='post')
    pred_lstm_pre = np.argmax(lstm_pretrained.predict(pad), axis=1)
    
    # Dự đoán bằng LSTM học từ đầu
    pred_lstm_scratch = np.argmax(lstm_scratch.predict(pad), axis=1)
    
    return pred_tfidf, pred_w2v, pred_lstm_pre, pred_lstm_scratch

preds = predict_all(examples)
for i, ex in enumerate(examples):
    print(f"\nExample: {ex}")
    print("TF-IDF + LR:", le.inverse_transform([preds[0][i]])[0])
    print("Word2Vec Avg + Dense:", le.inverse_transform([preds[1][i]])[0])
    print("Embedding Pre-trained + LSTM:", le.inverse_transform([preds[2][i]])[0])
    print("Embedding Scratch + LSTM:", le.inverse_transform([preds[3][i]])[0])

```
##### **Kết quả thực hiện**
| Example sentence                                         | True Label       | TF-IDF + LR     | Word2Vec Avg + Dense | Embedding Pre-trained + LSTM | Embedding Scratch + LSTM |
|----------------------------------------------------------|-----------------|----------------|---------------------|------------------------------|--------------------------|
| sorry but i think you've got that not right             | general_negate  | general_negate | general_negate      | general_affirm               | cooking_recipe           |
| is starbucks stock up or down from last quarter        | qa_stock        | qa_stock       | music_query         | general_affirm               | cooking_recipe           |
| find new trump articles but not from fox news           | news_query      | news_query     | cooking_recipe      | qa_stock                     | cooking_recipe           |

#### Nhận xét

1. **Câu có phủ định** ("sorry but i think you've got that not right"):
   - TF-IDF + LR và Word2Vec Avg + Dense dự đoán đúng nhãn `general_negate`.
   - LSTM với embedding pre-trained và scratch dự đoán sai (`general_affirm` / `cooking_recipe`), có thể do:
     - Tập dữ liệu quá nhỏ, LSTM chưa học được các mẫu phủ định phức tạp.
     - Embedding pre-trained chưa phản ánh đúng ngữ cảnh phủ định.
   - LSTM lý thuyết mạnh hơn nhưng cần dữ liệu đủ lớn để học các phủ định xa trong câu.

2. **Câu truy vấn tài chính** ("is starbucks stock up or down from last quarter"):
   - TF-IDF + LR dự đoán đúng nhãn `qa_stock`.
   - LSTM dự đoán sai (`general_affirm`), Word2Vec Avg + Dense cũng sai (`music_query`).
   - Nguyên nhân có thể do LSTM chưa học được mối quan hệ giữa các từ “stock”, “up/down” và “last quarter” khi dữ liệu quá ít.

3. **Câu truy vấn tin tức có phủ định** ("find new trump articles but not from fox news"):
   - TF-IDF + LR dự đoán đúng nhãn `news_query`.
   - LSTM pre-trained dự đoán nhầm `qa_stock`, LSTM scratch dự đoán `cooking_recipe`.
   - Trung bình vector từ (Word2Vec Avg) không nắm được thứ tự và phủ định, dẫn đến nhầm lẫn.
   - LSTM có khả năng xử lý chuỗi, nhưng khả năng học mối quan hệ từ xa và phủ định vẫn bị hạn chế do dữ liệu nhỏ.

- TF-IDF + Logistic Regression vẫn ổn định trên các câu khó, nhờ học trực tiếp mối quan hệ từ-tần suất.
- Word2Vec Avg + Dense và LSTM (cả pre-trained và scratch) gặp khó khăn với:
  - Câu có phủ định hoặc cấu trúc dài.
  - Các nhãn phụ thuộc vào ngữ cảnh xa.
- LSTM có tiềm năng xử lý chuỗi và giữ thông tin về thứ tự từ, nhưng để phát huy sức mạnh cần:
  - Tập dữ liệu lớn hơn.
  - Embedding phù hợp với domain.
  - Thêm kỹ thuật như attention hoặc Transformer để xử lý thông tin xa.

> Nhìn chung, dữ liệu nhỏ khiến mô hình phức tạp như LSTM không thể vượt trội hơn mô hình TF-IDF + LR đơn giản trong các ví dụ có ngữ cảnh phức tạp.

## **5. Khó khăn và giải pháp**

### **1. Khó khăn gặp phải**

1. **Dữ liệu nhỏ và phân bố nhãn không đều**

   * LSTM và embedding học từ đầu cần nhiều dữ liệu để học thứ tự từ và ngữ cảnh.
   * Một số nhãn ít xuất hiện dẫn đến mô hình không học được pattern, dễ overfit.

2. **Xử lý phủ định và cấu trúc câu phức tạp**

   * Word2Vec trung bình bỏ qua thứ tự từ.
   * LSTM nhỏ, dữ liệu hạn chế, khó nắm được mối quan hệ từ xa trong câu.

3. **Khởi tạo embeddings từ đầu**

   * Embedding học từ dữ liệu nhỏ dẫn đến F1 rất thấp và quá trình huấn luyện chậm.

4. **Câu dài và giới hạn padding**

   * Giới hạn `max_len=50` có thể cắt mất thông tin quan trọng.
   * Một số câu chứa nhiều ý định bị mất bớt thông tin khi padding/truncating.

5. **Chọn hyperparameters và tránh overfitting**

   * Số lượng neuron, dropout, batch size, epochs cần cân nhắc kỹ.
   * LSTM dễ overfit khi dữ liệu ít.

### **2. Giải pháp áp dụng**

1. **Sử dụng mô hình baseline mạnh với dữ liệu nhỏ**

   * TF-IDF + Logistic Regression ổn định và nhanh, đặc biệt trên tập dữ liệu nhỏ.

2. **Tăng kích thước dữ liệu hoặc augment data**

   * Thêm các câu biến thể, dịch ngược (back-translation) hoặc synonym replacement.
   * Giúp LSTM học tốt thứ tự từ và ngữ cảnh.

3. **Sử dụng pre-trained embeddings phù hợp domain**

   * Chọn embeddings từ corpus lớn cùng lĩnh vực (vd: domain-specific Word2Vec hoặc GloVe).
   * Giảm thời gian học embedding, cải thiện chất lượng dự đoán.

4. **Thử nghiệm các kiến trúc nâng cao**

   * Bi-directional LSTM để nắm cả ngữ cảnh trước và sau từ.
   * Attention hoặc Transformer giúp học mối quan hệ từ xa trong câu dài.

5. **Điều chỉnh hyperparameters hợp lý**

   * EarlyStopping để tránh overfitting.
   * Dropout và batch normalization hỗ trợ ổn định khi huấn luyện.

6. **Giám sát định tính và định lượng**

   * Kiểm tra dự đoán trên các câu có phủ định, câu dài hoặc câu phức tạp để đánh giá mô hình thực tế.

> Nhìn chung, lựa chọn mô hình và chiến lược huấn luyện cần cân bằng giữa kích thước dữ liệu, độ phức tạp câu, và tài nguyên huấn luyện.

## 6. Kết luận
### Nhận xét chung về ưu và nhược điểm của các phương pháp

| Phương pháp                       | Ưu điểm                                                                 | Nhược điểm                                                                                     |
|----------------------------------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| **TF-IDF + Logistic Regression**  | - Đơn giản, nhanh, dễ triển khai<br>- Hiệu quả cao trên dữ liệu nhỏ<br>- Tốt với các từ khóa đặc trưng | - Bỏ qua thứ tự từ, ngữ cảnh<br>- Khó xử lý phủ định hoặc câu phức tạp<br>- Không học được embeddings |
| **Word2Vec Avg + Dense Layer**    | - Vector dày đặc, biểu diễn ý nghĩa từ tốt hơn TF-IDF<br>- Mạng Dense dễ huấn luyện | - Trung bình vector bỏ qua thứ tự từ và mối quan hệ từ xa<br>- Hiệu năng kém trên dữ liệu nhỏ và câu dài |
| **Embedding Pre-trained + LSTM**  | - Giữ thông tin thứ tự từ và ngữ cảnh<br>- Pre-trained embedding giúp mô hình hiểu nghĩa từ tốt hơn | - Cần tập dữ liệu lớn để học tốt<br>- Chậm, khó huấn luyện<br>- Dữ liệu nhỏ khiến F1 thấp |
| **Embedding Scratch + LSTM**      | - Có thể học embedding phù hợp với domain<br>- Giữ thứ tự từ, ngữ cảnh như LSTM khác | - Khởi tạo từ đầu cần nhiều dữ liệu<br>- Dễ overfit, F1 và độ chính xác thấp khi dữ liệu hạn chế<br>- Thời gian huấn luyện lâu |

1. **Mô hình đơn giản (TF-IDF + LR)** thể hiện hiệu quả tốt trên tập dữ liệu hiện tại, đặc biệt khi dữ liệu không quá lớn và nhãn không phụ thuộc nhiều vào ngữ cảnh xa.  
2. **Các mô hình embedding và LSTM** về lý thuyết mạnh hơn, có khả năng nắm được thứ tự từ và ngữ cảnh, nhưng dữ liệu nhỏ khiến chúng không phát huy được sức mạnh.  
3. Việc lựa chọn mô hình cần cân nhắc giữa:
   - Kích thước dữ liệu (dữ liệu lớn phù hợp LSTM/embedding, dữ liệu nhỏ phù hợp TF-IDF).  
   - Độ phức tạp của ngữ liệu (câu dài, phủ định, nhiều ý định).  

## 7. Tài liệu tham khảo
1. **Mikolov, T., Chen, K., Corrado, G., & Dean, J.** (2013).
   *Efficient Estimation of Word Representations in Vector Space.*
   arXiv:1301.3781.

2. **Hochreiter, S., & Schmidhuber, J.** (1997).
   *Long Short-Term Memory.*
   Neural Computation, 9(8), 1735–1780.

3. **Jurafsky, D., & Martin, J. H.** (2023).
   *Speech and Language Processing (3rd ed. draft).*
   Chapter: Word2Vec, RNN, LSTM, Sequence Models.

4. **Scikit-learn Documentation.**
   TF-IDF, Logistic Regression, Label Encoding, Pipelines.
   [https://scikit-learn.org/](https://scikit-learn.org/)

5. **TensorFlow / Keras Documentation.**
   LSTM Layer, Embedding Layer, Tokenizer, pad_sequences.
   [https://www.tensorflow.org/](https://www.tensorflow.org/)

6. **Gensim Documentation.**
   Word2Vec model and training options.
   [https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/)

