# Lab 1: Text Tokenization

## Mục tiêu
Hiểu và triển khai bước tiền xử lý văn bản cơ bản: tokenization. Tạo cả tokenizer đơn giản và tokenizer nâng cao bằng regex.

## Mô tả công việc

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

## Cách chạy Lab 1
```bash
python -m labs.lab1.test_lab1_tokenizer
```

## Kết quả

### Ví dụ câu test

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

### Sample dataset UD\_English-EWT (first 20 tokens)

```
SimpleTokenizer: ['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim', ',']
RegexTokenizer: ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']
```

## Giải thích kết quả

* **SimpleTokenizer**: dễ đọc, giữ nguyên từ gốc và một số dấu câu liền nhau (ví dụ: "let's").
* **RegexTokenizer**: chi tiết hơn, tách ký tự đặc biệt như `'` ra riêng, phù hợp với pipeline NLP cần token chính xác.

* Lab 1 tập trung vào **tiền xử lý văn bản**, 

## Khó khăn và cách giải quyết

Trong quá trình thực hiện Lab 1, bạn có thể gặp phải những khó khăn sau:

| Khó khăn | Mô tả chi tiết | Cách giải quyết đề xuất |
| :--- | :--- | :--- |
| **1. Xử lý các trường hợp dấu câu và từ viết tắt phức tạp** | Các trường hợp như từ có dấu gạch nối (`"state-of-the-art"`), từ viết tắt (`"isn't"`, `"Dr."`), và ranh giới từ phức hợp (`"New York"`) rất khó xử lý bằng quy tắc đơn giản. | **Với RegexTokenizer**: Sử dụng biểu thức chính quy tinh vi hơn. Ví dụ, để tách tốt hơn các từ viết tắt, bạn có thể thử sử dụng mẫu như `r"\w+(?:'\w+)?|[^\w\s]"`. Đối với trường hợp như `"D. Trump"`, có thể sử dụng quy tắc lookaround, ví dụ: `r"(?<=[A-Za-z])\.\s+"` để chỉ tách khi dấu chấm đứng sau một chữ cái và có khoảng trắng. |
| **2. Cân bằng giữa việc giữ thông tin và hiệu suất** | **SimpleTokenizer** giữ nguyên `"let's"` giúp dễ đọc nhưng mất thông tin cấu trúc. **RegexTokenizer** tách quá chi tiết (`'isn' + "'" + 't'`) có thể làm tăng độ phức tạp cho các bước xử lý sau. | **Xác định rõ yêu cầu của tác vụ cuối cùng**: Đối với các tác vụ như phân loại văn bản, việc phân tách thô có thể chấp nhận được. Đối với các tác vụ như phân tích cú pháp (parsing) hoặc dịch máy, có thể cần độ chi tiết cao hơn. Lựa chọn hoặc thiết kế tokenizer phù hợp với mục tiêu. |
| **3. Xử lý ngôn ngữ không phải tiếng Anh và ký tự đặc biệt** | Biểu thức mặc định `\w+` chủ yếu khớp ký tự từ tiếng Anh (A-Z, a-z, 0-9, _) và sẽ bỏ sót các ký tự có dấu (ví dụ: `"café"`, `"tiếng Việt"`). | Sử dụng các biểu thức chính quy hỗ trợ Unicode tổng quát hơn. Ví dụ, `r"\p{L}+"` khớp với mọi ký tự chữ cái (yêu cầu thư viện `regex` của Python thay vì `re` tiêu chuẩn). |
| **4. Tích hợp tokenizer với các bước xử lý tiếp theo (ví dụ: vector hóa)** | Làm thế nào để tokenizer tùy chỉnh (như `RegexTokenizer`) có thể được kết nối trơn tru với các bước tiếp theo như `CountVectorizer`. | Tuân thủ nghiêm ngặt **Abstract Interface** `Tokenizer`. Đảm bảo rằng phương thức `tokenize` luôn trả về `list[str]`. Bất kỳ tokenizer nào tuân thủ giao diện này đều có thể được gọi một cách thống nhất bởi pipeline xử lý phía sau. |

## Kết luận

Lab 1 tập trung vào bước nền tảng **tiền xử lý văn bản** thông qua việc triển khai hai loại tokenizer. Qua bài lab này, chúng ta có thể rút ra một số điểm chính:

1.  **Sự đánh đổi (Trade-off)**: Không có một phương pháp tokenization nào là hoàn hảo cho mọi tình huống. `SimpleTokenizer` nhanh và dễ hiểu nhưng thiếu chính xác trong các trường hợp phức tạp. `RegexTokenizer` linh hoạt và chi tiết hơn nhưng có thể tạo ra quá nhiều token và đòi hỏi kiến thức về biểu thức chính quy. Việc lựa chọn phụ thuộc vào yêu cầu cụ thể của ứng dụng NLP.
2.  **Tầm quan trọng của việc chuẩn hóa (Standardization)**: Việc định nghĩa một abstract base class (`Tokenizer`) đóng vai trò then chốt trong việc tạo ra một hệ thống module, dễ bảo trì và mở rộng. Nó cho phép chúng ta dễ dàng hoán đổi các thuật toán tokenization khác nhau mà không cần thay đổi code ở các phần khác.
3.  **Tokenization là bước cơ bản nhưng quan trọng**: Chất lượng của tokenization ảnh hưởng trực tiếp đến hiệu suất của toàn bộ pipeline NLP phía sau (như vector hóa, huấn luyện mô hình). Một lỗi ở bước này có thể khuếch đại thành lỗi lớn ở đầu ra cuối cùng.

## Tài liệu tham khảo

1.  **Tài liệu chính thức**:
    *   **Python `re` module**: [https://docs.python.org/3/library/re.html](https://docs.python.org/3/library/re.html) - Tài liệu chính thức về biểu thức chính quy trong Python.
    *   **Natural Language Toolkit (NLTK) - Chapter 3: Processing Raw Text**: [https://www.nltk.org/book/ch03.html](https://www.nltk.org/book/ch03.html) - Cung cấp lý thuyết sâu và ví dụ thực tế về tokenization, stemming, lemmatization.

