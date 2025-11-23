import torch
from transformers import AutoTokenizer, AutoModel

# 1. Load model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. Input sentence
sentences = ["This is a sample sentence."]

# 3. Tokenize
# padding=True: đệm các câu ngắn hơn để có cùng độ dài
# truncation=True: cắt các câu dài hơn
# return_tensors='pt': trả về kết quả dưới dạng Pytorch tensors
inputs = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors='pt'
)

# 4. Đưa qua mô hình để lấy hidden states
# torch.no_grad() để không tính toán gradient, tiết kiệm bộ nhớ
# 4. Forward pass
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state  # shape: (1, seq_len, 768)

# 5. Mean Pooling with attention mask
# Để tính trung bình chính xác, chúng ta cần bỏ qua các token đệm (padding tokens)
attention_mask = inputs["attention_mask"]
mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

sentence_embedding = sum_embeddings / sum_mask

# 6. Print result
print("Vector biểu diễn:")
print(sentence_embedding)
print("\nKích thước vector:", sentence_embedding.shape)
