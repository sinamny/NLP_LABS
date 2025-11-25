# Lab 6. Giới thiệu về Transformer

## 1. Mục tiêu
- Ôn lại kiến thức cơ bản về kiến trúc Transformer.
- Sử dụng các mô hình Transformer tiền huấn luyện (pretrained models) để thực hiện các tác vụ NLP cơ bản.
- Làm quen với thư viện `transformers` của Hugging Face.

## 2. Hướng dẫn chạy code
### **2.1. Cấu trúc thư mục chính**
```
nlp-labs/
│
├── labs/
│   ├── lab1/                     # Lab 1: Tokenizer
│   ├── lab2/                     # Lab 2: Vectorizer
│   ├── lab4/                     # Lab 4: Word Embeddings
│   ├── lab5/                     # Lab 5: Text Classification
│   ├── lab5_2/                   # Lab 5: Giới thiệu về RNNs và các bài toán  
│   └── lab6/                     # Lab 6: Transformers
│       ├── lab6_masked_language_modeling.py 
│       ├── lab6_next_token_prediction.py
│       └── lab6_sentence_embedding.py

```

Các file Python trong Lab 6 tương ứng với từng bài tập:

| File | Mục đích |
| ------------- | ------------- |
| lab6_masked_language_modeling.py | Dự đoán từ bị che (Masked Language Modeling) |
| lab6_next_token_prediction.py | Sinh từ tiếp theo (Next Token Prediction) |
| lab6_sentence_embedding.py | Tính vector biểu diễn của câu (Sentence Embedding) |

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
### **2.3. Chạy Lab 6: ách chạy từng bài tập**
```bash
# Mở terminal tại thư mục gốc của dự án nlp-labs
cd nlp-labs

# Chạy bài tập Masked Language Modeling
python -m lab6.lab6_masked_language_modeling

# Chạy bài tập Next Token Prediction
python -m lab6.lab6_next_token_prediction

# Chạy bài tập Sentence Embedding
python -m lab6.lab6_sentence_embedding

```

> Lưu ý: Nếu thư mục `lab6` chưa có `__init__.py`, hãy tạo file trống này để Python nhận dạng là package.
## 3. Các bước thực hiện
### 3.1. Bài 1: Khôi phục Masked Token (Masked Language Modeling)

**Mục tiêu:**
- Dự đoán từ bị che trong một câu.
- Sử dụng mô hình Encoder-only (BERT), vốn có khả năng nhìn hai chiều, giúp hiểu ngữ cảnh xung quanh từ bị thiếu.

**Giải thích bước làm:**

```python
from transformers import pipeline

# 1. Tải pipeline "fill-mask"
# Pipeline này tự động tải mô hình BERT phù hợp cho tác vụ Masked LM
mask_filler = pipeline("fill-mask")

# 2. Câu đầu vào với token [MASK]
input_sentence = "Hanoi is the <mask> of Vietnam."

# 3. Thực hiện dự đoán, trả về top 5 từ khả dĩ nhất
predictions = mask_filler(input_sentence, top_k=5)

# 4. In kết quả
print(f"Câu gốc: {input_sentence}")
for pred in predictions:
 print(f"Dự đoán: '{pred['token_str']}' với độ tin cậy: {pred['score']:.4f}")
 print(f" -> Câu hoàn chỉnh: {pred['sequence']}")

```
- **Giải thích:**
- `pipeline("fill-mask")` cung cấp một API cao cấp để che từ và dự đoán từ đó.
- `top_k=5` trả về 5 dự đoán khả dĩ nhất.
- Kết quả hiển thị từ được dự đoán cùng với độ tin cậy và câu hoàn chỉnh.

### 3.2. Bài 2: Dự đoán từ tiếp theo (Next Token Prediction)

**Mục tiêu:**

- Sinh văn bản tiếp theo dựa trên một câu mồi.

- Sử dụng mô hình Decoder-only (GPT), vốn chỉ nhìn một chiều, dựa trên các token trước để dự đoán token tiếp theo.

**Giải thích bước làm:**

```python
from transformers import pipeline

# 1. Tải pipeline "text-generation"
# Pipeline tự động tải mô hình GPT-2 hoặc tương tự
generator = pipeline("text-generation")

# 2. Câu mồi
prompt = "The best thing about learning NLP is"

# 3. Sinh văn bản, max_length = 50
generated_texts = generator(prompt, max_length=50, num_return_sequences=1)

# 4. In kết quả
print(f"Câu mồi: '{prompt}'")
for text in generated_texts:
 print("Văn bản được sinh ra:")
 print(text['generated_text'])

```
- **Giải thích:**
- `pipeline("text-generation")` trừu tượng hóa việc sinh văn bản.
- `max_length` là tổng chiều dài tối đa của câu mồi + phần sinh ra.
- `num_return_sequences=1` trả về 1 chuỗi kết quả.

### 3.3. Bài 3: Tính toán vector biểu diễn của câu (Sentence Embedding)

**Mục tiêu:**

- Chuyển câu thành vector số có chiều cố định (dim = 768 với `bert-base-uncased`).

- Vector này nắm bắt ngữ nghĩa của cả câu và có thể dùng cho phân loại, tìm kiếm tương đồng, v.v.

- Sử dụng Mean Pooling kết hợp `attention_mask` để bỏ qua token padding.

**Giải thích bước làm:**

```python
import torch
from transformers import AutoTokenizer, AutoModel

# 1. Load BERT model và tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. Input sentence
sentences = ["This is a sample sentence."]

# 3. Tokenize, padding & truncation
inputs = tokenizer(
 sentences,
 padding=True,
 truncation=True,
 return_tensors='pt'
)

# 4. Forward pass, không tính gradient để tiết kiệm bộ nhớ
with torch.no_grad():
 outputs = model(**inputs)

# 5. Lấy hidden states của tất cả token
last_hidden_state = outputs.last_hidden_state # shape: (1, seq_len, 768)

# 6. Mean Pooling với attention_mask để bỏ token padding
attention_mask = inputs["attention_mask"]
mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
sentence_embedding = sum_embeddings / sum_mask

# 7. In kết quả
print("Vector biểu diễn:")
print(sentence_embedding)
print("\nKích thước vector:", sentence_embedding.shape)

```
- **Giải thích:**
- `last_hidden_state` chứa vector của tất cả token.
- `attention_mask` loại bỏ các token padding khi tính trung bình.
- `sentence_embedding` là vector **768 chiều**, đại diện cho câu đầu vào.


## 4. Kết quả thực hiện

### 4.1. Bài 1: Khôi phục Masked Token (Masked Language Modeling)
**Kết quả chạy:**

```
Câu gốc: Hanoi is the <mask> of Vietnam.
Dự đoán: ' capital' với độ tin cậy: 0.9341
 -> Câu hoàn chỉnh: Hanoi is the capital of Vietnam.
Dự đoán: ' Republic' với độ tin cậy: 0.0300
 -> Câu hoàn chỉnh: Hanoi is the Republic of Vietnam.
Dự đoán: ' Capital' với độ tin cậy: 0.0105
 -> Câu hoàn chỉnh: Hanoi is the Capital of Vietnam.
Dự đoán: ' birthplace' với độ tin cậy: 0.0054
 -> Câu hoàn chỉnh: Hanoi is the birthplace of Vietnam.
Dự đoán: ' heart' với độ tin cậy: 0.0014
 -> Câu hoàn chỉnh: Hanoi is the heart of Vietnam.
```

**Trả lời câu hỏi:**
1. **Mô hình đã dự đoán đúng từ “capital” không?**
Có, từ `capital` được mô hình dự đoán với độ tin cậy cao nhất 0.9341, chính xác là từ phù hợp để hoàn thiện câu.

2. **Tại sao các mô hình Encoder-only như BERT lại phù hợp cho tác vụ này?**

    - BERT là mô hình hai chiều (bidirectional), có khả năng nhìn ngữ cảnh ở cả hai phía trái và phải của token bị che.

    - Điều này giúp mô hình dự đoán từ [MASK] chính xác hơn vì hiểu được toàn bộ ngữ cảnh trong câu.

### 4.2. Bài 2: Dự đoán từ tiếp theo (Next Token Prediction)

**Kết quả chạy**
```
Câu mồi: 'The best thing about learning NLP is'
Văn bản được sinh ra:
The best thing about learning NLP is that it doesn't just mean learning something new, it means learning something new, and that means learning something new. We all learn what we want to learn, and we have different ways of doing that, so we can all learn what we want to learn.

I have come to the realization that my best learning is learning to find things. That's the key to being a good NLP learner.

I get it. I get it. I think I learned that a lot. I've learned that I like to find things, I've learned that I like to be funny, I've learned that I like to have fun. I've learned that I like to be a musician and I've learned that I like to be a good writer. It's learning to find something new, finding something new, and it's not about finding new things. It's about finding something.

I'm going to be saying this for what I want to say about the NLP. The NLP is how you learn to be good at something.

Not a lot of people are able to do it. Not a lot of people can learn it.

I think it's really important that we get to know ourselves, and what we're learning
```
**Trả lời câu hỏi:**
1. **Kết quả sinh ra có hợp lý không?**

    - Kết quả hợp lý về ngữ cảnh, các token tiếp theo tiếp nối câu mồi một cách tự nhiên, mặc dù có lặp lại một số ý.

    - Nhược điểm: có lặp lại ý tưởng, nhưng đây là vấn đề phổ biến của GPT khi sinh văn bản ngắn và số lượng token tối đa nhỏ.

    - Điều này minh họa khả năng mô hình dự đoán token tiếp theo dựa trên chuỗi trước đó.

2. **Tại sao các mô hình Decoder-only như GPT lại phù hợp cho tác vụ này?**

    - GPT là mô hình một chiều (unidirectional), dự đoán token tiếp theo dựa trên chuỗi trước đó. 

    - Cấu trúc này lý tưởng cho tác vụ sinh văn bản, vì việc dự đoán token phải dựa trên các token đã xuất hiện, không cần thông tin tương lai.

    - Nếu dùng Encoder-only, nó sẽ không sinh văn bản tự nhiên, vì mô hình không được huấn luyện để sinh token mới mà chỉ để dự đoán từ bị che. 
# 4.3. Bài 3: Tính toán Vector biểu diễn của câu (Sentence Embedding)
**Kết quả chạy:**

```
Vector biểu diễn:
tensor([[-6.3874e-02, -4.2837e-01, -6.6779e-02, -3.8430e-01, -6.5785e-02,...,  -1.7534e-01, -1.2388e-01,  3.1970e-01]])
Kích thước vector: torch.Size([1, 768])
```
        
  **Trả lời câu hỏi:**
1. **Kích thước (chiều) của vector biểu diễn là bao nhiêu? Con số này tương ứng với tham số nào của mô hình BERT?**

    - Kích thước vector là 768 chiều, tương ứng với hidden size (`hidden_size`) của mô hình `bert-base-uncased`.

    - Mỗi chiều đại diện cho một khía cạnh của ngữ nghĩa câu được mô hình học.
2. **Tại sao chúng ta cần sử dụng attention_mask khi thực hiện Mean Pooling?**

    - `attention_mask` giúp loại bỏ các token padding trong quá trình tính trung bình.

    - Nếu không loại bỏ, vector padding sẽ làm mất chính xác ngữ nghĩa câu, đặc biệt với các câu có độ dài khác nhau.

## 5. Khó khăn và giải pháp

### 5.1. Khó khăn

1. **Hiểu các loại mô hình Transformer**  
   - Việc phân biệt giữa **Encoder-only (BERT), Decoder-only (GPT), và Encoder-Decoder (T5)** đôi khi gây nhầm lẫn.  
   - Cần nhớ mô hình Encoder-only phù hợp cho tác vụ **Masked Language Modeling**, Decoder-only cho **Next Token Prediction**.

2. **Cài đặt và tải mô hình lớn**  
   - Một số mô hình như GPT-2 hoặc BERT-base khá nặng (~500MB - 1GB).  
   - Tải lần đầu thường mất thời gian, đặc biệt trên mạng chậm.

3. **Quản lý token padding và attention mask**  
   - Khi tính **Sentence Embedding**, nếu không sử dụng `attention_mask`, các token padding sẽ làm kết quả **bị sai lệch**.  

4. **Sinh văn bản với text-generation**  
   - Kết quả sinh ra đôi khi **lặp từ hoặc ý tưởng**, do mô hình GPT tối ưu theo xác suất token, không hiểu logic dài hạn.


### 5.2. Giải pháp

1. **Hiểu rõ loại mô hình và tác vụ**  
   - Sử dụng Encoder-only cho Masked LM, Decoder-only cho Next Token Prediction, và Encoder-Decoder cho các tác vụ dịch/biến đổi văn bản.

2. **Sử dụng cache và môi trường ảo**  
   - Cài đặt Hugging Face cache giúp tải lại mô hình nhanh hơn.  
   - Tạo môi trường Python riêng để tránh xung đột thư viện.

3. **Sử dụng attention_mask khi pooling**  
   - Khi tính **Mean Pooling**, nhân hidden states với `attention_mask` để loại bỏ padding.  

4. **Điều chỉnh tham số sinh văn bản**  
   - Sử dụng `max_length`, `temperature`, `top_k` hoặc `top_p` để kiểm soát chất lượng văn bản sinh ra.

## 6. Kết luận

- Lab 6 giúp ôn tập kiến trúc Transformer và cách áp dụng các mô hình pre-trained trong NLP.  
- Các tác vụ cơ bản đã thực hiện thành công:
  1. **Masked Language Modeling:** dự đoán từ bị che bằng BERT.  
  2. **Next Token Prediction:** sinh từ tiếp theo bằng GPT.  
  3. **Sentence Embedding:** trích xuất vector biểu diễn câu bằng BERT.  
- Các kỹ thuật như attention_mask, Mean Pooling, và pipeline của Hugging Face là những công cụ quan trọng để triển khai nhanh các ứng dụng NLP.  
- Hiểu rõ sự khác biệt giữa mô hình Encoder và Decoder giúp chọn đúng mô hình cho từng tác vụ.


## 7. Tài liệu tham khảo

1. Hugging Face Transformers Documentation:  
   [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)

2. Vaswani et al., “Attention is All You Need”, 2017.  
   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

3. Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”, 2018.  
   [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

4. Radford et al., “Language Models are Unsupervised Multitask Learners”, 2019 (GPT-2).  
   [https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
