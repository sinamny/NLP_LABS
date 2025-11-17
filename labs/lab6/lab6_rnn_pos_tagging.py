import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

# Task 1: Tải và tiền xử lý dữ liệu

def load_conllu(file_path):
    """
    Đọc file .conllu và trả về danh sách các câu
    Mỗi câu là danh sách các tuple (word, upos_tag)
    """
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            elif not line.startswith("#"):
                parts = line.split('\t')
                if len(parts) >= 5:
                    word = parts[1]
                    tag = parts[3]
                    sentence.append((word, tag))
        # thêm câu cuối nếu file không kết thúc bằng dòng trống
        if sentence:
            sentences.append(sentence)
    return sentences

# Xây dựng Vocabulary

def build_vocab(sentences):
    """
    Tạo word_to_ix và tag_to_ix từ dữ liệu huấn luyện
    """
    word_to_ix = {"<UNK>": 0}  # token cho từ không có trong vocab
    tag_to_ix = {}
    for sent in sentences:
        for word, tag in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    print(f"Size of word vocabulary: {len(word_to_ix)}")
    print(f"Size of tag set: {len(tag_to_ix)}")
    return word_to_ix, tag_to_ix

# Task 2: Tạo PyTorch Dataset và DataLoader

class POSDataset(Dataset):
    """
    Dataset tùy chỉnh cho POS Tagging
    """
    def __init__(self, sentences, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        # Chuyển từ và nhãn sang chỉ số
        sentence_idx = [self.word_to_ix.get(w, self.word_to_ix["<UNK>"]) for w, t in sentence]
        tag_idx = [self.tag_to_ix[t] for w, t in sentence]
        return torch.tensor(sentence_idx, dtype=torch.long), torch.tensor(tag_idx, dtype=torch.long)

def collate_fn(batch):
    """
    Pad các câu và nhãn về cùng độ dài trong batch
    """
    sentences, tags = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=-100)  # ignore_index cho loss
    return sentences_padded, tags_padded

# Task 3: Xây dựng mô hình RNN
class SimpleRNNForTokenClassification(nn.Module):
    """
    Mô hình RNN đơn giản cho POS Tagging
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(SimpleRNNForTokenClassification, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        output: [batch_size, seq_len, num_classes]
        """
        embeds = self.embedding(x)                 # [batch_size, seq_len, embedding_dim]
        rnn_out, _ = self.rnn(embeds)             # [batch_size, seq_len, hidden_dim]
        logits = self.fc(rnn_out)                 # [batch_size, seq_len, num_classes]
        return logits

# Task 4: Huấn luyện mô hình

def train_model(model, train_loader, dev_loader, epochs=5, lr=0.001, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # bỏ qua padding

    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for sentences, tags in train_loader:
            sentences, tags = sentences.to(device), tags.to(device)
            optimizer.zero_grad()
            logits = model(sentences)               # [batch_size, seq_len, num_classes]
            loss = criterion(logits.view(-1, logits.shape[-1]), tags.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        dev_acc = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Dev Acc: {dev_acc:.4f}")

# Task 5: Evaluate

def evaluate(model, data_loader, device='cpu'):
    """
    Tính accuracy trên tập dev, bỏ qua padding
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for sentences, tags in data_loader:
            sentences, tags = sentences.to(device), tags.to(device)
            logits = model(sentences)
            preds = torch.argmax(logits, dim=-1)
            mask = tags != -100
            correct += (preds[mask] == tags[mask]).sum().item()
            total += mask.sum().item()
    return correct / total

def predict_sentence(model, sentence, word_to_ix, tag_to_ix, device='cpu'):
    """
    Dự đoán nhãn POS cho một câu mới
    """
    model.eval()
    idx_to_tag = {v: k for k, v in tag_to_ix.items()}
    tokens_idx = [word_to_ix.get(w, word_to_ix["<UNK>"]) for w in sentence.split()]
    inputs = torch.tensor(tokens_idx, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]
    with torch.no_grad():
        logits = model(inputs)
        preds = torch.argmax(logits, dim=-1).squeeze(0)
    return list(zip(sentence.split(), [idx_to_tag[i.item()] for i in preds]))

if __name__ == "__main__":
    # Load dữ liệu
    train_sentences = load_conllu("src/data/UD_English-EWT/en_ewt-ud-train.conllu")
    dev_sentences = load_conllu("src/data/UD_English-EWT/en_ewt-ud-dev.conllu")

    # Build vocab
    word_to_ix, tag_to_ix = build_vocab(train_sentences)

    # Dataset & DataLoader
    train_dataset = POSDataset(train_sentences, word_to_ix, tag_to_ix)
    dev_dataset = POSDataset(dev_sentences, word_to_ix, tag_to_ix)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Khởi tạo mô hình
    VOCAB_SIZE = len(word_to_ix)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    NUM_CLASSES = len(tag_to_ix)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleRNNForTokenClassification(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES)

    # Huấn luyện mô hình
    train_model(model, train_loader, dev_loader, epochs=5, lr=0.001, device=device)

    # Độ chính xác cuối cùng trên dev
    final_dev_acc = evaluate(model, dev_loader, device=device)
    print(f"Final Dev Accuracy: {final_dev_acc:.4f}")

    # Ví dụ dự đoán câu mới
    sentence = "I love NLP"
    prediction = predict_sentence(model, sentence, word_to_ix, tag_to_ix, device=device)
    print(f"Sentence: {sentence}")
    print("Prediction:", prediction)
