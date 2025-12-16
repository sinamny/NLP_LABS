# Lab 2: Count Vectorization

## Mục tiêu

Biểu diễn văn bản dưới dạng số (Bag-of-Words) để sử dụng cho các mô hình học máy.

## Mô tả công việc

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

## Cách chạy Lab 2

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
* Lab 2 tập trung vào **biểu diễn văn bản thành số**.
* Document-term matrix trực quan hóa tần suất token, thuận tiện cho các bước NLP tiếp theo.
## Khó khăn và cách giải quyết
| Khó khăn | Mô tả chi tiết | Cách giải quyết đề xuất |
| :--- | :--- | :--- |
| **1. Bùng nổ từ vựng và kích thước vector** | Số lượng từ riêng biệt (unique tokens) trong corpus lớn có thể lên đến hàng trăm nghìn hoặc triệu. Điều này tạo ra vector đặc trưng có số chiều cực cao, dẫn đến tiêu tốn nhiều bộ nhớ và thời gian tính toán cho các mô hình học máy phía sau. | 1. **Giới hạn kích thước từ vựng (`max_features`)**: Chỉ giữ lại N từ xuất hiện thường xuyên nhất.<br>2. **Lọc theo tần suất tài liệu (`min_df`, `max_df`)**: Bỏ các từ quá hiếm (ví dụ: xuất hiện dưới 2 tài liệu) hoặc quá phổ biến (ví dụ: xuất hiện trong hơn 95% tài liệu, thường là stop words).<br>3. **Sử dụng N-gram một cách thận trọng**: `ngram_range=(1,2)` giúp nắm bắt cụm từ như `"new york"`, nhưng sẽ làm tăng số chiều theo cấp số nhân. Cần kết hợp với các phương pháp lọc trên. |
| **2. Vấn đề từ mới ngoài từ điển (Out-Of-Vocabulary - OOV)** | Khi áp dụng mô hình `transform` lên dữ liệu mới, có thể xuất hiện các từ chưa có trong `vocabulary_` được xây dựng từ tập huấn luyện. | Cách xử lý chuẩn là **bỏ qua** các từ OOV. Trong quá trình `transform`, chỉ đếm những từ có mặt trong `vocabulary_`. Điều này đòi hỏi tập dữ liệu huấn luyện phải đủ lớn và đại diện để bao phủ phần lớn từ vựng quan trọng. Có thể xem xét thêm một token đặc biệt `"<UNK>"` (unknown) để gom các từ hiếm trong tập huấn luyện và từ OOV. |
| **3. Nhiễu từ các từ quá phổ biến (Stop Words)** | Các từ dừng (stop words) như `"the"`, `"a"`, `"is"` xuất hiện trong hầu hết văn bản nhưng mang rất ít thông tin cho nhiều tác vụ (như phân loại chủ đề), có thể làm "lấn át" các từ khóa quan trọng. | Triển khai một **bộ lọc stop words**. Có thể tích hợp bằng cách:<br>1. Cho phép người dùng truyền vào một danh sách stop words tùy chỉnh.<br>2. Sử dụng danh sách stop words có sẵn từ các thư viện như NLTK.<br>3. Loại bỏ các từ này trong quá trình xây dựng `vocabulary_` ở hàm `fit`, hoặc lọc chúng dựa trên `max_df`. |
| **4. Mất thông tin về thứ tự và ngữ nghĩa của từ** | Mô hình Bag-of-Words vốn dĩ **không quan tâm đến thứ tự từ** và **ngữ cảnh**, khiến hai câu `"good movie"` và `"not good movie"` có biểu diễn khá giống nhau, dẫn đến mất thông tin quan trọng. | Hiểu rằng đây là **hạn chế cố hữu** của mô hình Bag-of-Words đơn giản. Nó đóng vai trò là **nền tảng** và **điểm chuẩn (baseline)**. Để khắc phục một phần, có thể sử dụng **N-gram** (giữ thứ tự cục bộ). Để nắm bắt ngữ nghĩa, cần chuyển sang các mô hình tiên tiến hơn như **TF-IDF** (điều chỉnh trọng số), **Word Embeddings** (Word2Vec, GloVe) hoặc **các mô hình dựa trên ngữ cảnh** (BERT). |
| **5. Hiệu suất và lưu trữ cho ma trận thưa** | Ma trận Document-Term có đặc điểm là **rất thưa** (sparse), tức là hầu hết các phần tử là 0. Lưu trữ dưới dạng list of lists hoặc mảng numpy thông thường sẽ rất lãng phí bộ nhớ. | Sử dụng cấu trúc **Ma trận thưa (Sparse Matrix)** từ thư viện `scipy.sparse` (ví dụ định dạng CSR - Compressed Sparse Row). Điều này giảm đáng kể dung lượng bộ nhớ và tăng tốc các phép toán đại số tuyến tính cho các tập dữ liệu lớn. |

## Kết luận

Lab 2 đã giới thiệu và triển khai một kỹ thuật biểu diễn văn bản cổ điển nhưng vô cùng quan trọng: **Mô hình Bag-of-Words (BoW)** thông qua **CountVectorizer**. Chúng ta có thể rút ra một số bài học chính:

1.  **Chuyển đổi thành công từ ngữ nghĩa sang số học**: CountVectorizer đã giải quyết bài toán cơ bản nhất trong NLP - biến đổi dữ liệu văn bản phi cấu trúc thành dạng số có cấu trúc mà các thuật toán học máy có thể xử lý được. Đây là bước đệm không thể thiếu.
2.  **Tính đơn giản và hiệu quả**: Ưu điểm lớn của BoW là dễ hiểu, dễ triển khai và cho kết quả khá tốt trong nhiều tác vụ phân loại văn bản đơn giản, đặc biệt khi kết hợp với các mô hình như Naive Bayes.
3.  **Nhận diện rõ hạn chế**: Qua lab, chúng ta thấy rõ các nhược điểm của BoW: (1) **Bỏ qua thứ tự từ**, (2) **Bỏ qua ngữ nghĩa** (coi mọi từ là độc lập), và (3) **Dễ bị ảnh hưởng bởi từ quá phổ biến hoặc quá hiếm**. Việc nhận thức được những hạn chế này là động lực để tìm hiểu các mô hình biểu diễn văn bản phức tạp hơn.
4.  **Tầm quan trọng của việc tinh chỉnh và tiền xử lý**: Lab cho thấy chất lượng của `CountVectorizer` phụ thuộc rất nhiều vào `Tokenizer` từ Lab 1 và các bước tinh chỉnh như lọc stop words, lựa chọn `max_features`. Điều này khẳng định lại rằng pipeline NLP là một chuỗi các bước xử lý có mối liên hệ chặt chẽ với nhau.

CountVectorizer là một công cụ mạnh mẽ trong hộp công cụ của một kỹ sư NLP, đặc biệt hữu ích để tạo prototype nhanh và thiết lập baseline. Tuy nhiên, để giải quyết các bài toán phức tạp hơn, chúng ta cần tiến tới các kỹ thuật như TF-IDF, Word Embeddings và các mô hình Deep Learning.

## Tài liệu tham khảo

1.  **Tài liệu chính thức & Sách giáo khoa**:
    *   **scikit-learn Documentation: CountVectorizer**: [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) - Tài liệu tham khảo toàn diện về cách triển khai thực tế trong thư viện phổ biến, với đầy đủ các tham số.
    *   **Jurafsky & Martin - Speech and Language Processing (3rd ed. draft)**: Chương 6 "Vector Semantics and Embeddings" cung cấp nền tảng lý thuyết xuất sắc về biểu diễn từ, từ Bag-of-Words đến embeddings.
