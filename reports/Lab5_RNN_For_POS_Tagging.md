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
│   │   ├── lab5_rnn_for_ner.py   
│   │   └── lab5_rnn_pos_tagging.py # Mã nguồn chính cho RNN For POS Tagging
│   ├── lab6/                     # Lab 6: Giới thiệu Transformer
│   └── __init__.py
```

> **Chú thích:**
>
> * Tất cả mã nguồn của Lab 5 (bao gồm RNN for NER) nằm trong `labs/lab5_2`.
> * File chính chạy trực tiếp: `lab5_rnn_pos_tagging.py`.
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
### **3.3. Chạy Lab 5: RNN for POS Tagging**
Tất cả mã nguồn được đặt trong:

```
labs/lab5_2/lab5_rnn_pos_tagging.py
```

Bạn có thể chạy toàn bộ pipeline (load dữ liệu => train => evaluate => test câu mới) bằng:

```
 python -m labs.lab5_2.lab5_rnn_pos_tagging
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

## **4. Các bước triển khai**
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


## **5. Kết quả**
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

## **6. Khó khăn và hướng giải quyết**

### Khó khăn gặp phải

1. **Xử lý dữ liệu CoNLL-U**
   - Dữ liệu có nhiều comment lines và câu có độ dài khác nhau.
   - Cần đảm bảo không bỏ sót token nào và giữ đúng thứ tự token-tag.

2. **Padding và batching**
   - Các câu có độ dài khác nhau gây khó khăn khi tạo batch.
   - Nếu không pad đúng, CrossEntropyLoss sẽ tính toán sai hoặc báo lỗi shape mismatch.
   
3. **Xử lý từ OOV (Out-of-Vocabulary)**
   - Từ không có trong vocab (`<UNK>`) dễ dẫn đến dự đoán sai.
   - Embedding cho các từ OOV thường không đủ ngữ cảnh, ảnh hưởng accuracy.

4. **Gradient và huấn luyện RNN**
   - RNN vanilla có hạn chế trong việc học dependencies dài.
   - Mô hình dễ gặp vấn đề vanishing/exploding gradient nếu seq_len quá dài.

5. **Đánh giá và masking**
   - Cần đảm bảo ignore padding tokens khi tính loss và accuracy.
   - Sai sót trong mask có thể làm kết quả đánh giá không chính xác.

### Hướng giải quyết

- **Dữ liệu CoNLL-U:** Luôn kiểm tra số token sau khi load; loại bỏ comment và xử lý câu trống đúng cách.
- **Padding & batching:** Sử dụng `pad_sequence` với `collate_fn` và `ignore_index=-100` cho loss.
- **Từ OOV:** Dùng token `<UNK>` cho từ không có trong vocab; cân nhắc sử dụng pre-trained embeddings (GloVe, FastText) để cải thiện.
- **Gradient RNN:** Giới hạn sequence length, sử dụng clip gradients, hoặc chuyển sang LSTM/BiLSTM để học dependencies dài h

## **7. Kết luận**

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

## **8. Tài liệu tham khảo**
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
