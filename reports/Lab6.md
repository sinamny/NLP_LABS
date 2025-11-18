# **Lab 5: PyTorch Introduction**

## **1. Mục tiêu**

Bài lab này giúp sinh viên làm quen với PyTorch – một trong những thư viện Deep Learning phổ biến và mạnh mẽ nhất hiện nay. Sau bài thực hành, sinh viên sẽ:

* Hiểu và thao tác với Tensor – cấu trúc dữ liệu cốt lõi của PyTorch.
* Sử dụng Autograd để tính đạo hàm tự động.
* Biết cách xây dựng một mô hình Neural Network đơn giản bằng `nn.Module`.
* Làm quen với hai lớp quan trọng:

  * `nn.Linear`
  * `nn.Embedding`


## **2. Hướng dẫn chạy code**
Tất cả mã nguồn được đặt trong:

```
labs/lab5/lab5_pytorch_intro.py
```

```bash
# Mở terminal tại thư mục gốc của dự án nlp-labs
cd nlp-labs

# Chạy Lab 5
 -m labs.lab5.lab5_pytorch_intro
```


# **PHẦN 1: TENSOR**

Tensor trong PyTorch tương tự như `numpy.ndarray`, nhưng mạnh hơn vì nó có thể chạy trên GPU và có hỗ trợ autograd.


## **Task 1.1 – Tạo Tensor**

### **Các bước thực hiện**

1. Tạo Tensor từ  list bằng `torch.tensor`.
2. Tạo Tensor từ NumPy array bằng `torch.from_numpy`.
3. Tạo Tensor bằng các hàm khởi tạo (`ones_like`, `rand_like`).
4. In ra:

   * `shape`
   * `dtype`
   * `device`

### **Code **

```python
import torch
import numpy as np

# Tạo tensor từ list
x_data = torch.tensor([[1, 2], [3, 4]])
print("Tensor từ list:\n", x_data)

# Tạo tensor từ numpy array
np_array = np.array([[1, 2], [3, 4]])
tensor_from_np = torch.from_numpy(np_array)
print("\nTensor từ Numpy array:\n", tensor_from_np)

# Tensor ones & random
ones_tensor = torch.ones_like(x_data)
rand_tensor = torch.rand_like(x_data, dtype=torch.float)
print("\nOnes Tensor:\n", ones_tensor)
print("\nRandom Tensor:\n", rand_tensor)

# Các thuộc tính của tensor
print("\nShape của tensor:", x_data.shape)
print("Dtype của tensor:", x_data.dtype)
print("Device lưu trữ của tensor:", x_data.device)
```

### **Kết quả**

```
Tensor từ list:
 tensor([[1, 2],
        [3, 4]])

Tensor từ Numpy array:
 tensor([[1, 2],
        [3, 4]])

Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.7485, 0.3161],
        [0.9889, 0.3378]])

Shape của tensor torch.Size([2, 2])
Dtype của tensor: torch.float32
Device lưu trữ của tensor: cpu
```


## **Task 1.2 – Các phép toán Tensor**

### **Các bước thực hiện**

1. Cộng hai tensor cùng kích thước.
2. Nhân toàn bộ tensor với một số.
3. Nhân ma trận bằng toán tử `@`.

### **Code**

```python
print("x_data + x_data =", x_data + x_data)
print("\nx_data * 5 =", x_data * 5)
print("\nx_data @ x_data.T =", x_data @ x_data.T)
```

### **Kết quả**

```python
x_data + x_data = tensor([[2, 4],
        [6, 8]])

x_data * 5 = tensor([[ 5, 10],
        [15, 20]])

x_data @ x_data.T = tensor([[ 5, 11],
        [11, 25]])
```


## **Task 1.3 – Indexing & Slicing**

### **Các bước thực hiện**

1. Lấy hàng đầu tiên.
2. Lấy cột thứ hai.
3. Truy xuất phần tử tại vị trí (2,2).

### **Code**

```python
print("Hàng đầu:", x_data[0])
print("Cột thứ hai:", x_data[:, 1])
print("Phần tử (2, 2): ", x_data[1, 1])
```

### **Kết quả**

```
Hàng đầu: tensor([1, 2])
Cột thứ hai: tensor([2, 4])
Phần tử (2, 2):  tensor(4)
```


## **Task 1.4 – Reshape Tensor**

### **Các bước thực hiện**

1. Tạo tensor 4×4 bằng `torch.rand`.
2. Reshape thành tensor 16×1 bằng `.view()` hoặc `.reshape()`.

### **Code**

```python
tensor4x4 = torch.rand(4, 4)
print("Tensor 4x4:\n", tensor4x4)

reshaped = tensor4x4.reshape(16, 1)
print("\nTensor reshape 16x1:\n", reshaped)
```

### **Kết quả**
```
Tensor 4x4:
 tensor([[0.9199, 0.0490, 0.9510, 0.4633],
        [0.5430, 0.5306, 0.8856, 0.5500],
        [0.1272, 0.2303, 0.1343, 0.9918],
        [0.4935, 0.9565, 0.5976, 0.8111]])

Tensor reshape 16x1:
 tensor([[0.9199],
        [0.0490],
        [0.9510],
        [0.4633],
        [0.5430],
        [0.5306],
        [0.8856],
        [0.5500],
        [0.1272],
        [0.2303],
        [0.1343],
        [0.9918],
        [0.4935],
        [0.9565],
        [0.5976],
        [0.8111]])
```


# **PHẦN 2: AUTOGRAD**

Autograd cho phép PyTorch tự động tính đạo hàm, cực kỳ quan trọng khi huấn luyện mạng nơ-ron.


## **Task 2.1 – Sử dụng Autograd**

### **Các bước thực hiện**

1. Tạo tensor `x` với `requires_grad=True`.
2. Tính:

$y = x + 2$
$z = y^2 \times 3$

3. Gọi `z.backward()` để tính gradient.

### **Code**

```python
x = torch.ones(1, requires_grad=True)
y = x + 2
z = y * y * 3

print("x =", x)
print("y =", y)
print("grad_fn của y:", y.grad_fn)

z.backward()
print("Đạo hàm dz/dx:", x.grad)
```

### **Kết quả**

```
x = tensor([1.], requires_grad=True)
y = tensor([3.], grad_fn=<AddBackward0>)
grad_fn của y: <AddBackward0 object at ...>
Đạo hàm dz/dx: tensor([18.])
```

### **Giải thích**


$y = x + 2 \quad\Rightarrow\quad y = 3$

$z = 3y^2 = 3 \times 9 = 27$

$\frac{dz}{dx} = 6y = 6 \times 3 = 18$

### Lưu ý quan trọng

Gọi `z.backward()` lần thứ 2 sẽ báo lỗi vì đồ thị đã bị giải phóng.
Muốn gọi nhiều lần phải dùng:

```
z.backward(retain_graph=True)
```


# **PHẦN 3: XÂY DỰNG MÔ HÌNH**


## **Task 3.1 – Lớp `nn.Linear`**

### **Các bước thực hiện**

1. Khai báo một layer Linear: input 5 → output 2.
2. Tạo dữ liệu đầu vào kích thước (3, 5).
3. Truyền vào layer để lấy output.

### **Code**

```python
import torch.nn as nn

linear = nn.Linear(5, 2)
inputs = torch.rand(3, 5)

outputs = linear(inputs)
print("Input shape:", inputs.shape)
print("Output shape:", outputs.shape)
print("Output:\n", outputs)
```

### **Kết quả**

```
Input shape: torch.Size([3, 5])
Output shape: torch.Size([3, 2])
...
```


## **Task 3.2 – Lớp `nn.Embedding`**

### **Các bước thực hiện**

1. Tạo embedding gồm 10 từ, mỗi từ có vector 3 chiều.
2. Đầu vào là các chỉ số từ (indices).
3. Trích xuất vector embedding.

### **Code**

```python
embedding = nn.Embedding(10, 3)
indices = torch.tensor([1, 5, 2, 9])
embedded = embedding(indices)

print("Input indices shape:", indices.shape)
print("Embedding output shape:", embedded.shape)
print("Embeddings:\n", embedded)
```

### **Kết quả**
```
Input indices shape: torch.Size([4])
Embedding output shape: torch.Size([4, 3])
Embeddings:
 tensor([[ 0.1813,  0.1492,  0.6733],
        [ 1.6073, -0.9811, -1.2778],
        [-0.5047, -0.7312,  0.7571],
        [ 0.1957, -0.5834, -0.2559]], grad_fn=<EmbeddingBackward0>)
```

## **Task 3.3 – Xây dựng mô hình bằng `nn.Module`**

### **Các bước thực hiện**

1. Tạo class kế thừa `nn.Module`.
2. Khởi tạo:

   * Embedding layer
   * Linear layer
3. Xây dựng hàm `forward`.

### **Code**

```python
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)     # (batch, seq_len, embed_dim)
        x = self.linear(x)        # (batch, seq_len, output_dim)
        return x

model = SimpleModel(10, 3, 4, 2)
input_data = torch.tensor([[1, 3, 5, 2]])
output = model(input_data)

print("Model output shape:", output.shape)
print("Model output:\n", output)
```

### **Kết quả**

```
Model output shape: torch.Size([1, 4, 2]) Model output: tensor([[[ 0.1115, 0.0493], [-0.0101, 0.0770], [ 0.3323, 0.2246], [-0.0048, 0.0609]]], grad_fn=<ViewBackward0>)
```


# **3. Kết luận**

Qua Lab 5:

* Hiểu cách tạo và thao tác tensor.
* Sử dụng được Autograd để tính gradient tự động.
* Làm quen với các layer cơ bản trong `torch.nn`.
* Xây dựng được một mô hình đơn giản kế thừa từ `nn.Module`.


# **Lab 5: Phân loại Văn bản với Mạng Nơ-ron Hồi quy (RNN/LSTM)**
## **Mục tiêu**
- Hiểu rõ hạn chế của các mô hình phân loại văn bản truyền thống (Bag-of-Words, Word2Vec trung bình).
- Nắm vững kiến trúc và luồng hoạt động của pipeline sử dụng RNN/LSTM cho bài toán phân loại văn bản.
- Tự tay xây dựng, huấn luyện và so sánh hiệu năng giữa các mô hình:
  1. TF-IDF + Logistic Regression (Baseline 1)
  2. Word2Vec (vector trung bình) + Dense Layer (Baseline 2)
  3. Embedding Layer (pre-trained) + LSTM
  4. Embedding Layer (học từ đầu) + LSTM
- Phân tích và đánh giá sức mạnh của mô hình chuỗi trong việc “hiểu” ngữ cảnh của câu.

## **Hướng dẫn chạy code**
Tất cả mã nguồn được đặt trong:

```
labs/lab5/lab5_rnns_text_classification.py
```

```bash
# Mở terminal tại thư mục gốc của dự án nlp-labs
cd nlp-labs

# Chạy Lab 5
 -m labs.lab5.lab5_rnns_text_classification
```


## **Bộ dữ liệu**
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

## **Tiền xử lý nhãn (Label Encoding)**

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

## **Task 1: Pipeline TF-IDF + Logistic Regression**
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

## **Task 2: Word2Vec + Dense Layer**
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

## **Task 3: Embedding Pre-trained + LSTM**

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

## **Task 4: Embedding học từ đầu + LSTM**

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

## **Task 5: Đánh giá, so sánh và phân tích**
### **1. So sánh định lượng**
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

### **2. Phân tích định tính**

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
#### **Kết quả**
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
## Kết luận
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

# **Lab 5: – Xây dựng mô hình RNN cho bài toán Part-of-Speech Tagging**
## **1. Mục tiêu**

Bài lab này nhằm xây dựng một mô hình Recurrent Neural Network (RNN) để giải quyết bài toán Part-of-Speech (POS) Tagging trên bộ dữ liệu Universal Dependencies (UD_English-EWT).

Sau khi hoàn thành, mô hình có khả năng:

* Đọc và tiền xử lý dữ liệu CoNLL-U.
* Xây dựng từ điển từ (word vocabulary) và nhãn (POS tag vocabulary).
* Tạo lớp Dataset và DataLoader tùy chỉnh trong PyTorch.
* Xây dựng mô hình RNN cho bài toán gán nhãn từng token.
* Huấn luyện mô hình, tính toán loss/accuracy.
* Dự đoán POS cho một câu mới.
## **2. Bộ dữ liệu (UD_English-EWT)**

Bộ dữ liệu gồm các câu ở định dạng **CoNLL-U**, mỗi token chứa nhiều trường.
Ta chỉ sử dụng:

* **FORM (cột 2)**: từ gốc.
* **UPOS (cột 4)**: nhãn POS theo chuẩn Universal Dependencies.

Một mẫu:

```
1   From    ADP
2   the     DET
3   AP      PROPN
4   comes   VERB
```
## **3. Các bước triển khai**
### **Task 1: Tải và tiền xử lý dữ liệu**
#### **Mục tiêu**

* Đọc file `.conllu`, tách câu dựa vào dòng rỗng..
* Trích xuất cặp `(word, tag)`.

**Hàm `load_conllu()`:**

Hàm load_conllu() xử lý:

* Bỏ qua các dòng comment bắt đầu bằng #.
* Gom các dòng liên tục thành 1 câu, phân tách bằng dòng rỗng.
* Lấy 2 trường: word = column[1], tag = column[3].
* Kết quả trả về: List[List[(word, tag)]].

```python
def load_conllu(file_path):
    """
    Đọc file .conllu và trả về danh sách các câu.
    Mỗi câu là danh sách các tuple (word, upos_tag).
    Lưu ý: file .conllu có các dòng comment bắt đầu '#'.
    """
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            line = line.strip()
            # Dòng rỗng => kết thúc 1 câu
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            # Bỏ qua comment lines
            if line.startswith("#"):
                continue
            parts = line.split('\t')
            # Cột 2 = FORM (word), cột 4 = UPOS (tag) theo chuẩn CoNLL-U
            if len(parts) >= 5:
                word = parts[1]
                tag = parts[3]
                sentence.append((word, tag))
        # Nếu file không kết thúc bằng dòng rỗng, thêm câu cuối
        if sentence:
            sentences.append(sentence)
    return sentences

```

##### **Xây dựng Vocabulary**
Hàm build_vocab():

* Tạo `word_to_ix`, bắt đầu với token đặc biệt `<UNK> = 0`. `word_to_ix` dùng `<UNK>` cho từ OOV. Index tăng dần khi gặp từ mới.

* Tạo `tag_to_ix` từ toàn bộ nhãn xuất hiện trong tập train.

* Trả về `(word_to_ix, tag_to_ix)` và in kích thước từ điển.

```python
def build_vocab(sentences):
    word_to_ix = {"<UNK>": 0}  # index 0 dành cho từ không có trong vocab
    tag_to_ix = {}

    for sent in sentences:
        for word, tag in sent:

            # Nếu từ mới => thêm vào vocab
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

            # Nếu tag mới => thêm vào tập nhãn
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    print(f"Size of word vocabulary: {len(word_to_ix)}")
    print(f"Size of tag set: {len(tag_to_ix)}")
    return word_to_ix, tag_to_ix


```

### **Task 2: Dataset & DataLoader**
Mỗi câu có độ dài khác nhau ⇒ cần padding. PyTorch DataLoader hỗ trợ điều này qua collate_fn.
#### Lớp Dataset
* Dataset cho POS Tagging. Input 1 câu => list từ + list nhãn
* Mỗi item trả về (sentence_indices_tensor, tag_indices_tensor) - tensors chưa pad cho từng câu. Không pad ở bước này mà để collate_fn xử lý.

```python
class POSDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.sentences)

     def __getitem__(self, idx):
        sent = self.sentences[idx]

        # Chuyển từng word → index (OOV → <UNK>)
        sentence_idx = [self.word_to_ix.get(w, self.word_to_ix["<UNK>"]) for w, t in sent]

        # Chuyển tag → index
        tag_idx = [self.tag_to_ix[t] for w, t in sent]

        return torch.tensor(sentence_idx, dtype=torch.long), torch.tensor(tag_idx, dtype=torch.long)

```
#### Hàm collate_fn 
* Do các câu dài ngắn khác nhau => cần padding.
* collate_fn dùng pad_sequence để pad các câu trong bath dùng `path_sequence` — đảm bảo batch có shape [batch_size, max_seq_len_in_batch].
* Pad sentences với padding_value=0 (index của <UNK> hoặc có thể reserve <PAD>=0).
* Pad tags với padding_value = -100 (sử dụng ignore_index=-100 trong CrossEntropyLoss). Vì nn.CrossEntropyLoss(ignore_index=-100) sẽ bỏ qua giá trị này khi tính loss.
* Trả về: sentences_padded [B, L], tags_padded [B, L]
```python
def collate_fn(batch):
    sentences, tags = zip(*batch)

    # Padding sentences (value = 0)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)

    # Padding tags (value = -100 để CrossEntropyLoss bỏ qua)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=-100)

    return sentences_padded, tags_padded

```

### Task 3: Xây dựng mô hình RNN (token classification)
Mô hình gồm 3 phần:

1. Embedding layer: biến chỉ số từ  index => vector d-chiều

2. RNN (vanilla RNN): xử lý chuỗi embedding

3. Linear layer: ánh xạ hidden state + số lượng tag

```python
class SimpleRNNForTokenClassification(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(SimpleRNNForTokenClassification, self).__init__()

        # Embedding: từ index => vector
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0   # vị trí 0 không sinh gradient
        )

        # Vanilla RNN (không dùng LSTM)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Lớp dự đoán tag POS tại mỗi timestep
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        embeds = self.embedding(x)         # [batch, seq_len, emb_dim]
        rnn_out, _ = self.rnn(embeds)      # [batch, seq_len, hidden_dim]
        logits = self.fc(rnn_out)          # [batch, seq_len, num_classes]
        return logits
```

### **Task 4: Huấn luyện mô hình**
Quy trình huấn luyện bao gồm:

* Đưa mô hình lên thiết bị (CPU/GPU)
* Khởi tạo:

  * **Optimizer:** Adam
  * **Loss:** CrossEntropyLoss (bỏ qua padding bằng *ignore_index = -100*)
* Cho mỗi batch:

  1. Forward => lấy logits
  2. Tính loss
  3. Backward
  4. Update trọng số
* Sau mỗi epoch:

  * In **train loss trung bình**
  * Tính **độ chính xác trên dev** bằng hàm `evaluate()`

Lưu ý: Do bài toán gán nhãn theo token, cần **flatten** logits và tags trước khi đưa vào CrossEntropyLoss.

```python
def train_model(model, train_loader, dev_loader, epochs=5, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss bỏ qua vị trí tag == -100
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0

        for batch_idx, (sentences, tags) in enumerate(train_loader, 1):
            sentences = sentences.to(device)
            tags = tags.to(device)

            optimizer.zero_grad()

            logits = model(sentences)     # [B, L, C]

            # Flatten để đưa vào CrossEntropyLoss
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                tags.view(-1)
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        dev_acc = evaluate(model, dev_loader, device)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.4f} - Dev Acc: {dev_acc:.4f}")

```
### **Task 5: Đánh giá và dự đoán**
Hàm evaluate(): Tính accuracy trên các vị trí không phải padding, tức tag != -100


```python

def evaluate(model, data_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sentences, tags in data_loader:
            sentences = sentences.to(device)
            tags = tags.to(device)

            logits = model(sentences)
            preds = torch.argmax(logits, dim=-1)

            # Chỉ tính accuracy với vị trí không phải padding
            mask = (tags != -100)

            correct += (preds[mask] == tags[mask]).sum().item()
            total += mask.sum().item()

    return correct / total if total > 0 else 0.0
```
Hàm predict_sentence():
* Tách câu thành token bằng split().
* Convert token => index.
* Lấy dự đoán bằng argmax.
* Trả về list [(token, predicted_tag)].

```python

def predict_sentence(model, sentence, word_to_ix, tag_to_ix, device='cpu'):
    model.eval()

    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    tokens = sentence.split()

    indices = [word_to_ix.get(t, 0) for t in tokens]  # 0 = <UNK>

    inputs = torch.tensor(indices).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inputs)
        preds = torch.argmax(logits, dim=-1).squeeze(0)

    return [(tok, ix_to_tag[int(p)]) for tok, p in zip(tokens, preds)]

```

## Hướng dẫn chạy code
Tất cả mã nguồn được đặt trong:

```
labs/lab5/lab5_rnn_pos_tagging.py
```

Bạn có thể chạy toàn bộ pipeline (load dữ liệu => train => evaluate => test câu mới) bằng:

```
 -m labs.lab5.lab5_rnn_pos_tagging
```

Lệnh trên sẽ:

1. Load dữ liệu train và dev
2. Xây dựng vocab
3. Tạo Dataset + DataLoader
4. Khởi tạo mô hình RNN
5. Huấn luyện trong số epoch định sẵn
6. In ra:

   * Train Loss từng epoch
   * Dev Accuracy từng epoch
7. In ví dụ dự đoán câu mới

## **Kết quả**
### **Kết quả quá trình train 5 epoch**

| Epoch | Train Loss | Dev Accuracy |
| ----- | ---------- | ------------ |
| 1     | 1.1542     | 0.7336       |
| 2     | 0.6537     | 0.7959       |
| 3     | 0.4896     | 0.8225       |
| 4     | 0.3885     | 0.8409       |
| 5     | 0.3164     | 0.8492       |

### **Nhận xét:**

* Loss giảm đều qua từng epoch => mô hình học tốt.
* Dev Accuracy tăng ổn định từ 0.73 => 0.85.
* Không xuất hiện hiện tượng overfitting rõ rệt.
* Epoch 5 cho kết quả tốt nhất theo Dev Accuracy.

### **Ví dụ câu dự đoán thực tế**

**Input câu:**

```
I love NLP
```

**Output dự đoán:**

```
[('I', 'PRON'), ('love', 'VERB'), ('NLP', 'VERB')]
```

### **Nhận xét:**

* `"I"` => PRON (đúng)
* `"love"` => VERB (đúng)
* `"NLP"` được gán thành VERB vì không nằm trong từ điển (OOV) => mô hình dựa vào ngữ cảnh nhưng vẫn sai.
  => Điều này cho thấy mô hình có thể còn yếu trong xử lý từ OOV và yêu cầu kỹ thuật embedding tốt hơn.

# **Kết luận**

Trong bài lab này, em đã:

* Hiểu rõ cấu trúc dữ liệu CoNLL-U và cách tách dữ liệu cho các tác vụ NLP.
* Tự xây dựng một pipeline hoàn chỉnh từ load dữ liệu => tạo vocab => Dataset/DataLoader => mô hình RNN => huấn luyện => dự đoán.
* Nắm được kỹ thuật tạo mô hình phân loại token (token classification).
* Biết cách xử lý padding cho bài toán sequence labeling.
* Huấn luyện thành công mô hình RNN đạt Dev Accuracy = 0.8492 sau 5 epoch.

Tuy mô hình RNN vanilla còn hạn chế, bài lab đã giúp em củng cố kiến thức về:

* Embedding
* RNN cơ bản
* POS tagging
* Training loop trong PyTorch
* Xử lý batch và mask

Kết quả đạt được khá tốt so với độ phức tạp của mô hình. Với mô hình mạnh hơn (LSTM, BiLSTM, CRF, BERT), hiệu suất sẽ tăng đáng kể.

# **Tài liệu tham khảo**
1. Universal Dependencies v2 – [https://universaldependencies.org](https://universaldependencies.org)  
2. PyTorch Documentation – RNN Module – [https://pytorch.org/docs/stable/generated/torch.nn.RNN.html](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)  
3. PyTorch Documentation – Embedding Layer – [https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)  
4. CoNLL-U Format Specification – [https://universaldependencies.org/format.html](https://universaldependencies.org/format.html)  

5. Stanford CS224N – *Recurrent Neural Networks for NLP* (lecture notes)  

6. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.  
    - Sử dụng TF-IDF và Logistic Regression trong Python.

7. OpenAI (2023). *ChatGPT: Optimizing Language Models for Dialogue*.  
    - Giới thiệu mô hình ngôn ngữ lớn (LLM) và ứng dụng chatbot, embeddings và sequence modeling.

8. Documentation:
    - [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)  
    - [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)  
    - [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
