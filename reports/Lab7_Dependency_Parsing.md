# Lab 7: Phân tích cú pháp phụ thuộc (Dependency Parsing)

## 1. Mục tiêu

- Hiểu và sử dụng thư viện spaCy để phân tích cú pháp phụ thuộc của câu tiếng Anh.

- Trực quan hoá cây phụ thuộc bằng displaCy để nắm bắt cấu trúc ngữ pháp (head — dependent).

- Lập trình truy cập và duyệt (traverse) cây phụ thuộc để trích xuất thông tin ngữ nghĩa: chủ ngữ, tân ngữ, tính từ bổ nghĩa, cụm danh từ, và đường đi tới gốc.

- Viết các hàm tiện ích thực hiện các tác vụ thông dụng: tìm động từ chính (ROOT), trích xuất noun-chunks tuỳ chỉnh, tìm triplet (subject, verb, object), và tìm đường đi từ token tới gốc.

## 2. Hướng dẫn chạy code

### 2.1. Cấu trúc thư mục chính
```
nlp-labs/
│
├── labs/
│ ├── lab1/     # Lab 1: Tokenizer
│ ├── lab2/     # Lab 2: Vectorizer
│ ├── lab4/     # Lab 4: Word Embeddings
│ ├── lab5/     # Lab 5: Text Classification
│ ├── lab5_2/   # Lab 5: Giới thiệu về RNNs và các bài toán
│ ├── lab6/     # Lab 6: Transformers
│ └── lab7/     # Lab 7: Dependency Parsing
│       ├── dependency_visualization.py # Task: Trực quan hoá cây phụ thuộc (displaCy)
│       ├── token_dependency_info.py # Task: In thông tin token (dep, head, children, pos)
│       ├── extract_subject_verb_object.py# Task: Trích xuất (subject, verb, object) từ câu
│       ├── extract_adjectives_for_nouns.py# Task: Tìm tính từ bổ nghĩa cho danh từ (amod)
│       ├── find_main_verb.py # Task: Hàm find_main_verb(doc) trả về token ROOT/VERB
│       ├── noun_chunks_custom.py # Task: Cài đặt trích xuất cụm danh từ tuỳ chỉnh
│       ├── path_to_root.py # Task: Hàm get_path_to_root(token) trả về đường đi tới ROOT
```

**Chú thích file => task**

- `dependency_visualization.py`: khởi tạo `Doc` và chạy `displacy.serve(doc, style='dep')` để mở trình duyệt xem cây.

- `token_dependency_info.py`: in bảng chi tiết cho từng token (text, dep_, head.text, head.pos_, children).

- `extract_subject_verb_object.py`: duyệt tokens, tìm VERB và con `nsubj`/`dobj` để in triplet.

- `extract_adjectives_for_nouns.py`: duyệt NOUN và lấy con có dep_ == 'amod'.

- `find_main_verb.py`: hàm trả về token có dep_ == 'ROOT' hoặc token.pos_ == 'VERB' & token.head == token.

- `noun_chunks_custom.py`: triển khai simple noun chunk extractor bằng cách gom noun + các dependent thuộc loại det, amod, compound.

- `path_to_root.py`: hàm `get_path_to_root(token)` lặp lên `token.head` cho tới khi `token.dep_ == 'ROOT'`.
### **2.2. Cài đặt môi trường (sử dụng `requirements.txt`)**

1. Tạo môi trường Python (Python ≥ 3.10):

```bash
python -m venv nlp-lab-env
source nlp-lab-env/bin/activate   # Linux/Mac
nlp-lab-env\Scripts\activate      # Windows
```

2. Cài đặt tất cả thư viện từ `requirements.txt` và tải model tiếng Anh (en_core_web_md):

```bash
pip install -r requirements.txt
# Tải model tiếng Anh (en_core_web_md)
python -m spacy download en_core_web_md
```
### **2.3. Chạy Lab 7: Lệnh chạy từng file
- Chạy trực quan hoá (mở trình duyệt):

```bash
# Mở terminal tại thư mục gốc của dự án nlp-labs
cd nlp-labs

python labs/lab6/dependency_visualization.py
# Sau đó truy cập http://127.0.0.1:5000 để xem cây (nhấn Ctrl+C để dừng server)

```
- In thông tin token:

```bash
python labs/lab6/token_dependency_info.py
# Kết quả: bảng dạng TEXT | DEP | HEAD TEXT | HEAD POS | CHILDREN

```
- Trích xuất SVO triplet:

```bash
python labs/lab6/extract_subject_verb_object.py
# Kết quả: Found Triplet: (cat, chased, mouse) (ví dụ)

```
- Tìm tính từ bổ nghĩa cho danh từ:

```bash
python labs/lab6/extract_adjectives_for_nouns.py
# Kết quả: Danh từ 'cat' được bổ nghĩa bởi: ['big','fluffy','white']

```
- Chạy unit test nhanh cho hàm `find_main_verb` và `get_path_to_root` (nếu file có test):

```bash
python -m pytest labs/lab6/tests -q

```

## 3. Phân tích câu và trực quan hóa bằng spaCy
### Phần 2. Phân tích câu và trực quan hóa
#### 2.1 Tải mô hình và phân tích câu 
##### Cách thực hiện
- Sử dụng spaCy để tải mô hình tiếng Anh `en_core_web_md`.
- Gọi `nlp(text)` để chạy toàn bộ pipeline (tokenization, POS tagging, dependency parsing).
- Kết quả thu được là đối tượng `Doc` chứa toàn bộ token và quan hệ phụ thuộc.

```python
import spacy
from spacy import displacy

# Tải mô hình tiếng Anh đã cài đặt
nlp = spacy.load("en_core_web_md")

# Câu ví dụ
text = "The quick brown fox jumps over the lazy dog."

# Phân tích câu với pipeline của spaCy
doc = nlp(text)

```

#### 2.2. Trực quan hóa cây phụ thuộc
```python
# Khởi chạy server tại http://127.0.0.1:5000
# Nhấn Ctrl+C trong terminal để dừng server
displacy.serve(doc, style="dep")

```
### Phần 3. Truy cập các thành phần trong cây phụ thuộc

### Phần 4. Duyệt cây phụ thuộc để trích xuất thông tin
#### 4.1. Bài toán: Tìm chủ ngữ và tân ngữ của một động từ

#### 4.2. Bài toán: Tìm các tính từ bổ nghĩa cho một danh từ

### Phần 5. Bài tập luyện tập
#### Bài 1: Tìm động từ chính của câu
#### Bài 2: Trích xuất các cụm danh từ Noun Chunks
#### Bài 3: Tìm đường đi ngắn nhất trong cây
### 3.2. Truy cập thuộc tính token

Dùng câu:

    Apple is looking at buying U.K. startup for $1 billion

Mỗi token có các thuộc tính: - token.text - token.dep\_ -
token.head.text - token.children

Kết quả như trong hướng dẫn.

### 3.3. Trích xuất (chủ ngữ, động từ, tân ngữ)

``` python
text = "The cat chased the mouse and the dog watched them."
doc = nlp(text)

for token in doc:
    if token.pos_ == "VERB":
        subject = ""
        obj = ""
        for child in token.children:
            if child.dep_ == "nsubj":
                subject = child.text
            if child.dep_ == "dobj":
                obj = child.text
        if subject and obj:
            print(f"Found Triplet: ({subject}, {token.text}, {obj})")
```

**Kết quả:**

    Found Triplet: (cat, chased, mouse)


### 3.4. Trích xuất tính từ bổ nghĩa danh từ

``` python
text = "The big, fluffy white cat is sleeping on the warm mat."
doc = nlp(text)

for token in doc:
    if token.pos_ == "NOUN":
        adjectives = [child.text for child in token.children if child.dep_ == "amod"]
        if adjectives:
            print(f"Danh từ '{token.text}' được bổ nghĩa bởi: {adjectives}")
```

**Kết quả:**

    Danh từ 'cat' được bổ nghĩa bởi: ['big', 'fluffy', 'white']
    Danh từ 'mat' được bổ nghĩa bởi: ['warm']



## 4. Bài tập tự luyện + Lời giải

### Bài 1. Tìm động từ chính (ROOT)

``` python
def find_main_verb(doc):
    for token in doc:
        if token.dep_ == "ROOT":
            return token
```

### Bài 2. Trích xuất noun chunks thủ công

``` python
def extract_noun_chunk(token):
    children = [child.text for child in token.children if child.dep_ in ["det", "amod", "compound"]]
    return " ".join(children + [token.text])
```

### Bài 3. Đường đi từ token → ROOT

``` python
def get_path_to_root(token):
    path = []
    while token.dep_ != "ROOT":
        path.append(token)
        token = token.head
    path.append(token)
    return path
```

## 5. Tổng kết

Bạn đã học: - Cách phân tích cú pháp phụ thuộc spaCy. - Cách duyệt cây
cú pháp. - Trích xuất thông tin: triplet (S--V--O), noun modifiers,... -
Viết hàm xử lý cấu trúc ngữ pháp nâng cao.
