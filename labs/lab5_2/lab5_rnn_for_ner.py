from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn


# Task 1: Tải và tiền xử lý dữ liệu
# 1. Tải dữ liệu CoNLL2003
dataset = load_dataset("conll2003", trust_remote_code=True)

# Trích xuất tokens và nhãn (ở dạng số)
train_sentences = dataset["train"]["tokens"]
train_tags = dataset["train"]["ner_tags"]

val_sentences = dataset["validation"]["tokens"]
val_tags = dataset["validation"]["ner_tags"]

# 2. Chuyển nhãn số sang nhãn text
tag_names = dataset["train"].features["ner_tags"].feature.names

train_tags = [[tag_names[t] for t in seq] for seq in train_tags]
val_tags   = [[tag_names[t] for t in seq] for seq in val_tags]

# 3. Xây vocab cho từ & nhãn
# Vocab từ: chỉ lấy từ tập train
word_to_ix = {"<PAD>": 0, "<UNK>": 1}
for sent in train_sentences:
    for w in sent:
        if w not in word_to_ix:
            word_to_ix[w] = len(word_to_ix)

# Vocab nhãn
tag_to_ix = {"<PAD>": 0}
for seq in train_tags:
    for tag in seq:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

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
