# Week 2: NLP Preprocessing & Count Vectorization

## Lab 1: Text Tokenization

### Mục tiêu
Hiểu và triển khai bước tiền xử lý văn bản cơ bản: **tokenization**. Tạo cả tokenizer đơn giản và tokenizer nâng cao bằng regex.

### Mô tả công việc

1. **Chuẩn bị interface Tokenizer**  
   - Định nghĩa trong `src/core/interfaces.py` một abstract base class `Tokenizer` với phương thức:
     ```python
     tokenize(self, text: str) -> list[str]
     ```
   - Đây là phần **core** để tách riêng logic xử lý token khỏi cách load dữ liệu.

2. **SimpleTokenizer**  
   - File code: `src/preprocessing/simple_tokenizer.py`  
   - Chuyển toàn bộ văn bản về chữ thường.  
   - Tách từ dựa trên khoảng trắng.  
   - Xử lý các dấu câu cơ bản (`.`, `,`, `!`, `?`) thành token riêng.  
   - Kết hợp với interface `Tokenizer` để có chuẩn phương thức chung.

3. **RegexTokenizer**  
   - File code: `src/preprocessing/regex_tokenizer.py`  
   - Sử dụng biểu thức chính quy `\w+|[^\w\s]` để tách token chi tiết hơn, ví dụ "isn't" -> `isn` + `'` + `t`.

4. **Load dữ liệu từ dataset**  
   - File loader: `src/core/dataset_loaders.py`  
   - Load dữ liệu UD_English-EWT từ `data/UD_English-EWT/en_ewt-ud-train.txt`.  
   - Lấy 500 ký tự đầu để thử tokenization và so sánh output của SimpleTokenizer và RegexTokenizer.

5. **Test/demo Lab 1**  
   - File: `labs/lab1/test_lab1_tokenizer.py`  
   - Chạy thử tokenizer trên câu mẫu và sample từ dataset, in ra token.

### Cách chạy Lab 1
```bash
python -m labs.lab1.test_lab1_tokenizer
````

### Kết quả mẫu

#### Ví dụ câu test

```
Text: Hello, world! This is a test.
SimpleTokenizer: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
RegexTokenizer: ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Text: NLP is fascinating... isn't it?
SimpleTokenizer: ['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']
RegexTokenizer: ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Text: Let's see how it handles 123 numbers and punctuation!
SimpleTokenizer: ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
RegexTokenizer: ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
```

#### Sample dataset UD\_English-EWT (first 20 tokens)

```
SimpleTokenizer: ['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim', ',']
RegexTokenizer: ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']
```

### Giải thích kết quả

* **SimpleTokenizer**: dễ đọc, giữ nguyên từ gốc và một số dấu câu liền nhau (ví dụ: "let's").
* **RegexTokenizer**: chi tiết hơn, tách ký tự đặc biệt như `'` ra riêng, phù hợp với pipeline NLP cần token chính xác.

---

## Lab 2: Count Vectorization

### Mục tiêu

Biểu diễn văn bản dưới dạng số (Bag-of-Words) để sử dụng cho các mô hình học máy.

### Mô tả công việc

1. **Định nghĩa interface Vectorizer**

   * File: `src/core/interfaces.py`
   * Abstract base class `Vectorizer` với phương thức:

     ```python
     fit(self, corpus: list[str])
     transform(self, documents: list[str]) -> list[list[int]]
     fit_transform(self, corpus: list[str]) -> list[list[int]]
     ```

2. **Triển khai CountVectorizer**

   * File: `src/representations/count_vectorizer.py`
   * Nhận một tokenizer từ Lab 1.
   * Tạo `vocabulary_` từ tập hợp các token duy nhất.
   * Chuyển danh sách văn bản thành **document-term matrix**.

3. **Test/demo Lab 2**

   * File: `labs/lab2/test_lab2_vectorizer.py`
   * Chạy CountVectorizer trên corpus mẫu và in ra vocabulary, document-term matrix.

### Cách chạy Lab 2

```bash
python -m labs.lab2.test_lab2_vectorizer
```

### Kết quả mẫu

#### Sample corpus

```python
corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]
```

* **Vocabulary**:

```
{'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}
```

* **Document-Term Matrix**:

```
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]
```

### Giải thích kết quả

* Vocabulary gồm tất cả token duy nhất từ corpus.
* Mỗi hàng trong document-term matrix là một văn bản, các giá trị là số lần xuất hiện token.
* CountVectorizer sẵn sàng dùng cho các mô hình học máy như Naive Bayes hoặc Logistic Regression.
* Lab 1 tập trung vào **tiền xử lý văn bản**, Lab 2 tập trung vào **biểu diễn văn bản thành số**.
* Document-term matrix trực quan hóa tần suất token, thuận tiện cho các bước NLP tiếp theo.
