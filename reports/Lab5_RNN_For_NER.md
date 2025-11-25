# Lab 5: Xây dựng mô hình RNN cho bài toán Nhận dạng Thực thể Tên (NER)

## 1. Mục tiêu
- Xây dựng và huấn luyện mô hình RNN cho bài toán Nhận dạng Thực thể Tên (NER).  
- Làm quen với dữ liệu NER CoNLL 2003, xây dựng vocabulary, Dataset, DataLoader trong PyTorch.  
- Huấn luyện mô hình, đánh giá hiệu năng bằng Accuracy, Precision, Recall, F1-score.  
- Dự đoán các thực thể trên câu mới.  

## 2. Bộ dữ liệu: CoNLL 2003
- Bộ dữ liệu CoNLL 2003 là benchmark tiêu chuẩn cho bài toán NER.
- Nhãn được gán theo định dạng IOB (Inside, Outside, Beginning):
  - `B-PER`: bắt đầu một thực thể tên người (Person)
  - `I-PER`: bên trong thực thể tên người
  - `B-LOC`: bắt đầu thực thể địa điểm (Location)
  - `I-LOC`: bên trong thực thể địa điểm
  - `O`: không phải thực thể (Outside)

**Ví dụ:**
```
U.N. official Ekeus heads for Baghdad .
B-ORG O B-PER O O B-LOC O
```
## 3. Hướng dẫn chạy code
### **3.1. Cấu trúc thư mục chính**
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
│   │   ├── lab5_rnns_text_classification.py
│   │   ├── lab5_rnn_for_ner.py   # Mã nguồn chính cho RNN NER
│   │   └── lab5_rnn_pos_tagging.py
│   ├── lab6/                     # Lab 6: Language Modeling
│   └── __init__.py
```

> **Chú thích:**
>
> * Tất cả mã nguồn của Lab 5 (bao gồm RNN for NER) nằm trong `labs/lab5_2`.
> * File chính chạy trực tiếp: `lab5_rnn_for_ner.py`.

### **3.2. Cài đặt môi trường (sử dụng `requirements.txt`)**

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
### **3.3. Chạy Lab 5: RNN for NER**

Tất cả mã nguồn được đặt trong:

```
labs/lab5_2/lab5_rnn_for_ner.py
```

```bash
# Mở terminal tại thư mục gốc của dự án nlp-labs
cd nlp-labs

# Chạy Lab 5
 python -m labs.lab5_2.lab5_rnn_for_ner
```

## 4. Các bước thực hiện
### **Task 1: Tải và Tiền xử lý Dữ liệu**

**Mục tiêu:**
- Nắm vững cách tải dữ liệu NER chuẩn từ `tensorflow_datasets`.
- Chuyển đổi dữ liệu từ định dạng TensorFlow sang danh sách Python.
- Xây dựng vocabulary từ dữ liệu train cho từ và nhãn.
- Lưu ý: Dữ liệu CoNLL 2003 trên Hugging Face đôi khi không tải được trực tiếp hoặc cần cấu hình tokenizers phức tạp, nên sử dụng `tensorflow_datasets` để đảm bảo tải nhanh, ổn định và chuẩn hóa sẵn các split train/validation/test.

**Giải thích code các bước:**

1. **Tải dữ liệu từ `tensorflow_datasets`:**
```python
import tensorflow_datasets as tfds

ds_train, ds_val = tfds.load('conll2003', split=['train', 'dev'], as_supervised=False)
```

* Bộ dữ liệu trả về là các `tf.data.Dataset` cho train và validation với các trường: `'tokens'` và `'ner'`. Dữ liệu gồm các token và nhãn NER theo định dạng số (sau này ánh xạ sang nhãn string)
* `as_supervised=False` giữ nguyên cấu trúc dictionary.

2. **Chuyển dữ liệu sang list Python:**
-   - Sử dụng hàm `tfds_to_list()` để đọc từng ví dụ, decode token từ byte => string và chuyển nhãn thành list.

```python
def tfds_to_list(ds):
    sentences, tags = [], []
    for ex in tfds.as_numpy(ds):
        tokens = [t.decode("utf-8") for t in ex['tokens']]
        ner_tags = ex['ner'].tolist()  
        sentences.append(tokens)
        tags.append(ner_tags)
    return sentences, tags

train_sentences, train_tags = tfds_to_list(ds_train)
val_sentences, val_tags = tfds_to_list(ds_val)
```

* Mục đích: dễ dàng xử lý với PyTorch, convert sang index.

3. **Tạo vocabulary từ dữ liệu train:**
* `word_to_ix`: ánh xạ từ => index, thêm token `<PAD>` và `<UNK>` để padding và xử lý từ lạ.
* `tag_to_ix`: ánh xạ nhãn string => index, thêm `<PAD>` để đệm nhãn.
* Chuyển nhãn số từ dataset sang nhãn string bằng `tag_names = tfds.builder('conll2003').info.features['ner'].names`.
```python
# Word vocabulary
word_to_ix = {"<PAD>":0, "<UNK>":1}
for sent in train_sentences:
    for w in sent:
        if w not in word_to_ix:
            word_to_ix[w] = len(word_to_ix)

# Tag vocabulary
ner_feature = tfds.builder('conll2003').info.features['ner']
tag_names = ner_feature.names

tag_to_ix = {"<PAD>": 0}
for seq in train_tags:
    for t in seq:
        t_name = tag_names[t]
        if t_name not in tag_to_ix:
            tag_to_ix[t_name] = len(tag_to_ix)

# Chuyển nhãn số sang string
train_tags = [[tag_names[t] for t in seq] for seq in train_tags]
val_tags = [[tag_names[t] for t in seq] for seq in val_tags]

# In ra số lượng từ và nhãn trong vocab
print("Số lượng từ trong vocab:", len(word_to_ix))
print("Số lượng nhãn:", len(tag_to_ix))
```

* `<PAD>`: dùng cho padding, `<UNK>`: từ lạ.
* Chuyển nhãn số sang nhãn string: `B-PER`, `I-ORG`, `O`, ...


### **Task 2: Tạo PyTorch Dataset và DataLoader**

**Mục tiêu:**

* Tạo dataset tuỳ chỉnh để trả về `(sentence_indices, tag_indices)`.
* Tạo DataLoader với padding, batch_size cố định.

**Giải thích code các bước:**

1. **Tạo lớp Dataset tùy chỉnh:**
* Nhận vào danh sách câu, nhãn, và hai từ điển `word_to_ix` & `tag_to_ix`.
* `__getitem__`: chuyển từ thành index (`<UNK>` nếu không có) và nhãn sang index.
* Trả về tuple `(sentence_indices, tag_indices)`.
```python
from torch.utils.data import Dataset
import torch

class NERDataset(Dataset):
    def __init__(self, sentences, tags, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.tags = tags
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent_ids = torch.tensor([self.word_to_ix.get(w, self.word_to_ix["<UNK>"]) for w in self.sentences[idx]])
        tag_ids = torch.tensor([self.tag_to_ix[t] for t in self.tags[idx]])
        return sent_ids, tag_ids
```

2. **Hàm `collate_fn` cho DataLoader để padding batch:**
* Dùng `pad_sequence` để padding tất cả các câu và nhãn trong batch về độ dài của câu dài nhất.
* Padding value cho từ là `<PAD>` index, cho nhãn cũng là `<PAD>` index.

```python
from torch.nn.utils.rnn import pad_sequence

PAD_WORD = word_to_ix["<PAD>"]
PAD_TAG  = tag_to_ix["<PAD>"]    

def collate_fn(batch):
    sentences, tags = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=PAD_WORD)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=PAD_TAG)
    return sentences_padded, tags_padded
```

3. **Khởi tạo DataLoader cho train và validation:**

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(NERDataset(train_sentences, train_tags, word_to_ix, tag_to_ix),
                          batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(NERDataset(val_sentences, val_tags, word_to_ix, tag_to_ix),
                        batch_size=32, shuffle=False, collate_fn=collate_fn)
```


### **Task 3: Xây dựng Mô hình RNN**

**Mục tiêu:**

* Thiết kế mô hình RNN dự đoán nhãn NER cho từng token.
* Hiểu cách embedding, RNN, linear layer kết hợp.

**Giải thích code các bước:**
1. **Kiến trúc mô hình:**

   * `nn.Embedding`: chuyển từ index => vector embedding.
   * `nn.RNN`: xử lý chuỗi embedding, thu được hidden states cho từng token.
   * `nn.Linear`: ánh xạ hidden states sang số lượng nhãn NER.

```python
import torch.nn as nn

class SimpleRNNForNER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super().__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # RNN layer
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        # Linear layer
        self.fc = nn.Linear(hidden_dim, num_tags)
```
2. **Forward pass:**

   * Embedding => RNN => Linear => logits (B, seq_len, num_tags)

```python
    def forward(self, x):
        emb = self.embedding(x)          # (B, L, emb_dim)
        rnn_out, _ = self.rnn(emb)       # (B, L, hidden_dim)
        logits = self.fc(rnn_out)        # (B, L, num_tags)
        return logits
```

* `nn.Embedding`: index => vector dense
* `nn.RNN`: xử lý chuỗi token
* `nn.Linear`: hidden_dim => số lượng nhãn


### **Task 4: Huấn luyện Mô hình**

**Mục tiêu:**

* Huấn luyện mô hình RNN với CrossEntropyLoss, bỏ qua padding.
* Quan sát sự hội tụ của loss qua các epoch.

**Giải thích code các bước:**

1. **Khởi tạo mô hình, optimizer, loss function**

   * Sử dụng `nn.CrossEntropyLoss(ignore_index=PAD_TAG)` để bỏ qua các token padding.
   * Optimizer: `Adam` với lr=1e-3

2. **Vòng lặp huấn luyện**

   * Lặp qua mỗi batch:

     1. Xóa gradient cũ: `optimizer.zero_grad()`
     2. Forward pass: `logits = model(sent_batch)`
     3. Tính loss: `loss = criterion(logits.view(-1, num_tags), tag_batch.view(-1))`
     4. Backward pass: `loss.backward()`
     5. Cập nhật trọng số: `optimizer.step()`

```python
for epoch in range(3):
    total_loss = 0
    for sent_batch, tag_batch in train_loader:
        ...
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
```

* `ignore_index=PAD_TAG`: bỏ qua token padding khi tính loss.


### **Task 5: Đánh giá Mô hình**

**Mục tiêu:**

* Tính accuracy token-level và seqeval metrics (Precision, Recall, F1).
* Dự đoán nhãn cho câu mới.

**Giải thích code các bước:**
1. **Hàm `evaluate_ner()`**

   * Đặt mô hình ở chế độ đánh giá: `model.eval()`
   * Không tính gradient: `with torch.no_grad():`
   * Dự đoán logits => `argmax` => lấy nhãn dự đoán.
   * Mask padding token để tính accuracy đúng.
   * Chuẩn bị dữ liệu cho `seqeval` để đánh giá Precision, Recall, F1.

```python
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

ix_to_tag = {v:k for k,v in tag_to_ix.items()}

def evaluate_ner(model, loader, ix_to_tag):
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for sent_batch, tag_batch in loader:
            logits = model(sent_batch)
            preds = logits.argmax(-1)
            mask = tag_batch != PAD_TAG
            correct += ((preds == tag_batch) & mask).sum().item()
            total += mask.sum().item()
            for i in range(sent_batch.size(0)):
                pred_seq, label_seq = [], []
                for j in range(sent_batch.size(1)):
                    if mask[i,j]:
                        pred_seq.append(ix_to_tag[preds[i,j].item()])
                        label_seq.append(ix_to_tag[tag_batch[i,j].item()])
                all_preds.append(pred_seq)
                all_labels.append(label_seq)
    print(f"Token-level Accuracy: {correct/total:.4f}")
    print(classification_report(all_labels, all_preds))
    print("F1-score:", f1_score(all_labels, all_preds))
```
2. **Dự đoán câu mới**
* Chia câu thành token => chuyển sang index => forward pass => argmax => ánh xạ index → nhãn.
* Hàm `predict_sentence` dùng để dự đoán nhãn cho câu mới:

```python
def predict_sentence(model, sentence):
    model.eval()
    tokens = sentence.split()
    ids = torch.tensor([word_to_ix.get(w, word_to_ix["<UNK>"]) for w in tokens]).unsqueeze(0)
    with torch.no_grad():
        logits = model(ids)
        preds = logits.argmax(-1).squeeze().tolist()
    return [(token, ix_to_tag[pred]) for token, pred in zip(tokens, preds)]

example = "VNU University is located in Hanoi"
print(predict_sentence(model, example))
```
## **5. Kết quả thực hiện**
```
Số lượng từ trong vocab: 23625
Số lượng nhãn: 10
Epoch 1: Loss = 273.9806
Epoch 2: Loss = 150.5590
Epoch 3: Loss = 101.5341
Token-level Accuracy: 0.9043
Classification report (seqeval):
              precision    recall  f1-score   support

         LOC       0.50      0.77      0.61      1837
        MISC       0.58      0.55      0.56       922
         ORG       0.31      0.51      0.39      1341

   micro avg       0.48      0.61      0.54      5942
   macro avg       0.50      0.59      0.53      5942
weighted avg       0.51      0.61      0.54      5942

F1-score: 0.5360503890329751
Precision: 0.4788825632199126
Recall: 0.6087176035005049
Ví dụ: [('VNU', 'B-ORG'), ('University', 'I-ORG'), ('is', 'O'), ('located', 'O'), ('in', 'O'), ('Hanoi', 'B-LOC')]
```
**Độ chính xác trên tập validation:**  
- Token-level Accuracy: **0.9043**


**Chi tiết đánh giá NER (bằng thư viện seqeval):**

```

```
          precision    recall  f1-score   support

     LOC       0.50      0.77      0.61      1837
    MISC       0.58      0.55      0.56       922
     ORG       0.31      0.51      0.39      1341
```

micro avg       0.48      0.61      0.54      5942
macro avg       0.50      0.59      0.53      5942
weighted avg       0.51      0.61      0.54      5942

```

- F1-score: **0.5361**  
- Precision: **0.4789**  
- Recall: **0.6087**

**Ví dụ dự đoán câu mới:**

- Câu: `"VNU University is located in Hanoi"`  
- Dự đoán: `[('VNU', 'B-ORG'), ('University', 'I-ORG'), ('is', 'O'), ('located', 'O'), ('in', 'O'), ('Hanoi', 'B-LOC')]`

**Nhận xét kết quả:**  
- Mô hình RNN đơn giản có thể đạt độ chính xác token-level ~90%, cho thấy khả năng học ngữ cảnh cơ bản.  
- Tuy nhiên, F1-score trung bình (~0.54) cho thấy mô hình còn hạn chế trong việc nhận dạng chính xác các thực thể, đặc biệt là ORG.  
- Nguyên nhân: mô hình chưa sử dụng bi-directional RNN, dropout, hay CRF layer, các kỹ thuật nâng cao có thể cải thiện khả năng dự đoán thực thể liên tục.  
- Mô hình hoạt động tốt với thực thể LOC nhưng chưa ổn định với ORG và MISC, điều này phản ánh sự thiếu cân bằng dữ liệu và độ khó của từng loại thực thể.

**Kết luận:**  
- Mô hình RNN cơ bản là bước đầu để hiểu cách áp dụng mạng nơ-ron hồi quy cho bài toán NER.  
- Các bước tiếp theo để cải thiện: sử dụng LSTM/GRU, RNN hai chiều (bi-RNN), thêm lớp CRF, tăng số lượng epoch, hoặc tăng embedding dimension.

## 6. Khó khăn và hướng giải quyết

**Khó khăn gặp phải:**

1. **Tiền xử lý dữ liệu NER:**  
   - Dữ liệu từ Hugging Face ở dạng `DatasetDict` với nhãn số nguyên, cần ánh xạ về tên nhãn (B-PER, I-LOC, …) trước khi huấn luyện.  
   - Một số từ xuất hiện rất ít lần hoặc không có trong từ điển (`<UNK>`), ảnh hưởng đến khả năng tổng quát của mô hình.

2. **Padding cho batch:**  
   - Các câu trong tập dữ liệu có độ dài khác nhau, cần xử lý padding cho cả từ và nhãn.  
   - Cần đảm bảo `CrossEntropyLoss` bỏ qua các vị trí padding (`ignore_index=PAD_TAG`).

3. **Hiệu năng mô hình RNN cơ bản:**  
   - RNN 1 chiều đơn giản gặp khó khăn trong việc học mối quan hệ dài hạn giữa các token.  
   - Khó nhận dạng chính xác các thực thể ít xuất hiện (ORG, MISC).

**Hướng giải quyết:**

- Sử dụng từ điển `<PAD>` và `<UNK>` để xử lý các từ lạ và padding.  
- Viết hàm `collate_fn` để pad đồng bộ cả câu và nhãn trong batch.  
- Sử dụng thư viện `seqeval` để đánh giá F1, precision, recall chuẩn cho NER, không chỉ accuracy token-level.  
- Có thể nâng cấp mô hình bằng LSTM/GRU, bi-directional RNN, hoặc thêm CRF layer để cải thiện kết quả.


## 7. Kết luận

- Lab này giúp nắm vững quy trình xây dựng mô hình RNN cho bài toán NER từ đầu đến cuối:  
  1. Tải và tiền xử lý dữ liệu CoNLL 2003.  
  2. Xây dựng từ điển từ và nhãn.  
  3. Tạo Dataset và DataLoader trong PyTorch.  
  4. Xây dựng và huấn luyện mô hình RNN.  
  5. Đánh giá hiệu năng mô hình bằng token-level accuracy và seqeval F1-score.

- Mô hình RNN cơ bản đạt **token-level accuracy ~90%**, tuy nhiên F1-score trung bình (~0.54) cho thấy còn hạn chế trong việc nhận dạng thực thể phức tạp.  
- Đây là nền tảng để tiếp tục nâng cao mô hình với các kỹ thuật như Bi-LSTM, CRF, attention, hoặc Transformer cho NER.

## 8. Tài liệu tham khảo

1. Tjong Kim Sang, Erik F., and Fien De Meulder. “Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition.” *arXiv preprint cs/0306050*, 2003.  
2. PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)  
3. seqeval Library for NER Evaluation: [https://github.com/chakki-works/seqeval](https://github.com/chakki-works/seqeval)  
4. Jurafsky, Daniel, and James H. Martin. *Speech and Language Processing*, 3rd Edition, draft chapters available online.  


