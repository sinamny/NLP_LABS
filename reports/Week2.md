# Week 2: NLP Preprocessing & Count Vectorization

## Lab 1: Text Tokenization

### Mô tả công việc
Trong Lab 1, tôi đã triển khai các bước xử lý văn bản cơ bản:

1. **SimpleTokenizer**  
   - Chuyển toàn bộ văn bản về chữ thường.  
   - Tách từ dựa trên khoảng trắng.  
   - Xử lý các dấu câu cơ bản (`.`, `,`, `!`, `?`) bằng cách tách chúng ra thành token riêng.  

2. **RegexTokenizer**  
   - Sử dụng biểu thức chính quy `\w+|[^\w\s]` để tách token.  
   - Tokenizer này tách chi tiết hơn, ví dụ các ký tự `'` trong từ "isn't" được tách riêng thành `isn` + `'` + `t`.  

3. **Task 3: Tokenization với UD_English-EWT Dataset**  
   - Load một phần dữ liệu thực từ file `en_ewt-ud-train.txt`.  
   - Thử token hóa 500 ký tự đầu tiên để so sánh output giữa SimpleTokenizer và RegexTokenizer.

### Kết quả chạy code

#### Câu test ví dụ
```

Input: Hello, world! This is a test.
SimpleTokenizer -> \['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
RegexTokenizer  -> \['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

Input: NLP is fascinating... isn't it?
SimpleTokenizer -> \['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']
RegexTokenizer  -> \['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

Input: Let's see how it handles 123 numbers and punctuation!
SimpleTokenizer -> \["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
RegexTokenizer  -> \['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

```

#### Sample từ dataset UD_English-EWT
```

SimpleTokenizer (first 20 tokens): \['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim', ',']
RegexTokenizer  (first 20 tokens): \['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']

````

### Giải thích kết quả
- **SimpleTokenizer** dễ đọc, giữ các từ gốc và một số dấu câu liền nhau (vd: "let's").  
- **RegexTokenizer** chi tiết hơn, tách các ký tự đặc biệt như `'` ra riêng, phù hợp cho các pipeline NLP cần token chính xác.


## Lab 2: Count Vectorization

### Mô tả công việc
1. Định nghĩa interface `Vectorizer` với các phương thức `fit`, `transform`, `fit_transform`.  
2. Triển khai **CountVectorizer**:
   - Nhận một tokenizer (Simple hoặc Regex) từ Lab 1.  
   - Tạo `vocabulary_` từ tập hợp các token duy nhất.  
   - Chuyển đổi danh sách văn bản thành **document-term matrix**.

### Kết quả chạy code

#### Sample corpus
```python
corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]
````

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

* Vocabulary bao gồm tất cả token duy nhất từ corpus.
* Mỗi hàng trong document-term matrix tương ứng với một văn bản, các giá trị là số lần xuất hiện của token trong văn bản đó.
* CountVectorizer này có thể dùng cho các mô hình học máy, ví dụ Naive Bayes hay Logistic Regression.

**Kết luận:**

* Lab 1 tập trung vào tiền xử lý text, Lab 2 tập trung vào biểu diễn văn bản dưới dạng số.
* Việc tách riêng Task 1, Task 2 giúp dễ dàng so sánh SimpleTokenizer và RegexTokenizer.
* Document-term matrix cho thấy trực quan tần suất xuất hiện token trong corpus.
