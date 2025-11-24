import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# Task 1: Load dữ liệu từ tfds
ds_train, ds_val = tfds.load('conll2003', split=['train', 'dev'], as_supervised=False)

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
# Task 2: Tạo vocab cho từ và nhãn
# Vocab từ
word_to_ix = {"<PAD>": 0, "<UNK>": 1}
for sent in train_sentences:
    for w in sent:
        if w not in word_to_ix:
            word_to_ix[w] = len(word_to_ix)

# Vocab nhãn
ner_feature = tfds.builder('conll2003').info.features['ner']
tag_names = ner_feature.names

tag_to_ix = {"<PAD>": 0}
for seq in train_tags:
    for t in seq:
        t_name = tag_names[t]
        if t_name not in tag_to_ix:
            tag_to_ix[t_name] = len(tag_to_ix)

# Chuyển nhãn số sang nhãn string
train_tags = [[tag_names[t] for t in seq] for seq in train_tags]
val_tags   = [[tag_names[t] for t in seq] for seq in val_tags]

print("Số lượng từ trong vocab:", len(word_to_ix))
print("Số lượng nhãn:", len(tag_to_ix))

# Task 2: Tạo PyTorch Dataset và DataLoader
class NERDataset(Dataset):
    def __init__(self, sentences, tags, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.tags = tags
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        tag_seq = self.tags[idx]

        # Chuyển mỗi từ => chỉ số, OOV thì dùng <UNK>
        sent_ids = torch.tensor([self.word_to_ix.get(w, self.word_to_ix["<UNK>"]) 
                                 for w in sent])

        # Chuyển nhãn => chỉ số
        tag_ids = torch.tensor([self.tag_to_ix[t] for t in tag_seq])

        return sent_ids, tag_ids

# Hàm collate_fn để padding cho batch
PAD_WORD = word_to_ix["<PAD>"]
PAD_TAG  = tag_to_ix["<PAD>"]    

def collate_fn(batch):
    sentences, tags = zip(*batch)

    # pad các chuỗi theo batch
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=PAD_WORD)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=PAD_TAG)

    return sentences_padded, tags_padded


# Tạo DataLoader
train_loader = DataLoader(
    NERDataset(train_sentences, train_tags, word_to_ix, tag_to_ix),
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    NERDataset(val_sentences, val_tags, word_to_ix, tag_to_ix),
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

# Task 3: Xây dựng mô hình RNN
class SimpleRNNForNER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super().__init__()

        # Lớp Embedding ánh xạ từ index → vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # RNN tầng 1 chiều
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False
        )

        # Lớp Linear ánh xạ hidden_state → số lượng nhãn
        self.fc = nn.Linear(hidden_dim, num_tags)

    def forward(self, x):
        emb = self.embedding(x)        # (batch, seq_len, emb_dim)
        rnn_out, _ = self.rnn(emb)     # (batch, seq_len, hidden_dim)
        logits = self.fc(rnn_out)      # (batch, seq_len, num_tags)
        return logits

# Task 4: Huấn luyện mô hình
# Khởi tạo mô hình
model = SimpleRNNForNER(
    vocab_size=len(word_to_ix),
    embedding_dim=128,
    hidden_dim=128,
    num_tags=len(tag_to_ix)
)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Loss function — bỏ qua padding tag
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TAG)
# Training Loop
for epoch in range(3):
    model.train()
    total_loss = 0

    for sent_batch, tag_batch in train_loader:
        optimizer.zero_grad()

        logits = model(sent_batch)             # (B, L, num_tags)
        logits = logits.view(-1, logits.size(-1))  
        tags = tag_batch.view(-1)              # (B*L)

        loss = criterion(logits, tags)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Task5: Đánh giá mô hình
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# Mapping index => tag
ix_to_tag = {v:k for k,v in tag_to_ix.items()}

def evaluate_ner(model, loader, ix_to_tag):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sent_batch, tag_batch in loader:
            logits = model(sent_batch)
            preds = logits.argmax(-1)

            # Tạo mask để bỏ qua padding
            mask = tag_batch != PAD_TAG

            # Accuracy token-level
            correct += ((preds == tag_batch) & mask).sum().item()
            total += mask.sum().item()

            # Chuẩn bị dữ liệu cho seqeval
            for i in range(sent_batch.size(0)):
                pred_seq = []
                label_seq = []
                for j in range(sent_batch.size(1)):
                    if mask[i, j]:
                        pred_seq.append(ix_to_tag[preds[i,j].item()])
                        label_seq.append(ix_to_tag[tag_batch[i,j].item()])
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    acc = correct / total
    print(f"Token-level Accuracy: {acc:.4f}")
    print("Classification report (seqeval):")
    print(classification_report(all_labels, all_preds))
    print("F1-score:", f1_score(all_labels, all_preds))
    print("Precision:", precision_score(all_labels, all_preds))
    print("Recall:", recall_score(all_labels, all_preds))

    return acc, all_preds, all_labels

# Gọi evaluate
accuracy, preds, labels = evaluate_ner(model, val_loader, ix_to_tag)


# Hàm dự đoán câu mới
def predict_sentence(model, sentence):
    model.eval()
    tokens = sentence.split()
    ids = torch.tensor([word_to_ix.get(w, word_to_ix["<UNK>"]) for w in tokens]).unsqueeze(0)

    with torch.no_grad():
        logits = model(ids)
        preds = logits.argmax(-1).squeeze().tolist()

    return [(token, ix_to_tag[pred]) for token, pred in zip(tokens, preds)]

example = "VNU University is located in Hanoi"
prediction = predict_sentence(model, example)
print("Ví dụ:", prediction)
