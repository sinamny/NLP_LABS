import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, f1_score, log_loss
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping


# 1. Load dữ liệu
df_train = pd.read_csv('src/data/hwu/train.csv', quotechar='"')
df_val   = pd.read_csv('src/data/hwu/val.csv', quotechar='"')
df_test  = pd.read_csv('src/data/hwu/test.csv', quotechar='"')

df_train.columns = ['text', 'intent']
df_val.columns = ['text', 'intent']
df_test.columns = ['text', 'intent']

print("Train shape:", df_train.shape)
print("Validation shape:", df_val.shape)
print("Test shape:", df_test.shape)
df_train.head()

# 2. Encode Labels
le = LabelEncoder()
le.fit(pd.concat([df_train['intent'], df_val['intent'], df_test['intent']]))

y_train = le.transform(df_train['intent'])
y_val   = le.transform(df_val['intent'])
y_test  = le.transform(df_test['intent'])
num_classes = len(le.classes_)
print("Test missing classes:", set(df_test['intent']) - set(df_train['intent']))
# 3. Task 1. Pipeline TF-IDF + Logistic Regression
tfidf_lr_pipeline = make_pipeline(TfidfVectorizer(max_features=5000), LogisticRegression(max_iter=1000))
tfidf_lr_pipeline.fit(df_train['text'], y_train)
y_pred_tfidf = tfidf_lr_pipeline.predict(df_test['text'])
print("TF-IDF + Logistic Regression")
print(classification_report(y_test, y_pred_tfidf, target_names=le.classes_))


# 4. Task 2. Pipeline Word2Vec + DenseLayer

# 4.1. Train Word2Vec trên dữ liệu text
sentences = [text.split() for text in df_train['text']]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 4.2. Convert câu thành vector trung bình
def sentence_to_avg_vector(text, model):
    words = text.split()
    vecs = [model.wv[w] for w in words if w in model.wv]
    if len(vecs) > 0:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(model.vector_size)

# 4.3. Tạo dữ liệu train/val/test
X_train_avg = np.array([sentence_to_avg_vector(text, w2v_model) for text in df_train['text']])
X_val_avg   = np.array([sentence_to_avg_vector(text, w2v_model) for text in df_val['text']])
X_test_avg  = np.array([sentence_to_avg_vector(text, w2v_model) for text in df_test['text']])

# 4.4. Xây dựng model Dense
model_dense = Sequential([
    Dense(128, activation='relu', input_shape=(w2v_model.vector_size,)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model_dense.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_dense.fit(X_train_avg, y_train, validation_data=(X_val_avg, y_val), epochs=10, batch_size=32)
y_pred_dense = np.argmax(model_dense.predict(X_test_avg), axis=1)
print("Word2Vec Avg + Dense")
print(classification_report(y_test, y_pred_dense, target_names=le.classes_))

# 5. Task 3: Mô hình nâng cao (Embedding Pre-trained + LSTM)
# 5.1. Tokenizer + Padding
tokenizer = Tokenizer(num_words=5000, oov_token="<UNK>")
tokenizer.fit_on_texts(df_train['text'])
X_train_seq = tokenizer.texts_to_sequences(df_train['text'])
X_val_seq   = tokenizer.texts_to_sequences(df_val['text'])
X_test_seq  = tokenizer.texts_to_sequences(df_test['text'])
max_len = 50
X_train_pad = pad_sequences(X_train_seq, maxlen= max_len, padding='post')
X_val_pad   = pad_sequences(X_val_seq, maxlen=max_len, padding='post')
X_test_pad  = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# 5.2. Ma trận Embedding từ Word2Vec
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = w2v_model.vector_size
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# 5.3.Xây dựng model LSTM với pre-trained embeddings
lstm_pretrained = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(num_classes, activation='softmax')
])
lstm_pretrained.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lstm_pretrained.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val), epochs=10, batch_size=32, callbacks=[early_stop])
y_pred_lstm_pre = np.argmax(lstm_pretrained.predict(X_test_pad), axis=1)
print("Embedding Pre-trained + LSTM")
print(classification_report(y_test, y_pred_lstm_pre, target_names=le.classes_))

# 6. Task 4: Mô hình nang cao (Embedding học từ đầu + LSTM)
lstm_scratch = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_len, trainable=True),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(num_classes, activation='softmax')
])
lstm_scratch.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lstm_scratch.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val), epochs=10, batch_size=32, callbacks=[early_stop])
y_pred_lstm_scratch = np.argmax(lstm_scratch.predict(X_test_pad), axis=1)
print("Embedding Scratch + LSTM")
print(classification_report(y_test, y_pred_lstm_scratch, target_names=le.classes_))

# F1-Score Macro & Test_Loss
results = []

y_proba_tfidf = tfidf_lr_pipeline.predict_proba(df_test['text'])
f1_tfidf = f1_score(y_test, y_pred_tfidf, average='macro')
loss_tfidf = log_loss(y_test, y_proba_tfidf, labels=list(range(num_classes)))

results.append({
    'pipeline': 'TF-IDF + LR',
    'f1_macro': f1_tfidf,
    'test_loss': loss_tfidf
})

f1_w2v = f1_score(y_test, y_pred_dense, average='macro')
loss_w2v, acc_w2v = model_dense.evaluate(X_test_avg, y_test, verbose=0)
results.append({
    'pipeline': 'Word2Vec Avg + Dense',
    'f1_macro': f1_w2v,
    'test_loss': loss_w2v
})

f1_lstm_pre = f1_score(y_test, y_pred_lstm_pre, average='macro')
loss_lstm_pre, acc_lstm_pre = lstm_pretrained.evaluate(X_test_pad, y_test, verbose=0)
results.append({
    'pipeline': 'Embedding Pre-trained + LSTM',
    'f1_macro': f1_lstm_pre,
    'test_loss': loss_lstm_pre
})

f1_lstm_scratch = f1_score(y_test, y_pred_lstm_scratch, average='macro')
loss_lstm_scratch, acc_lstm_scratch = lstm_scratch.evaluate(X_test_pad, y_test, verbose=0)
results.append({
    'pipeline': 'Embedding Scratch + LSTM',
    'f1_macro': f1_lstm_scratch,
    'test_loss': loss_lstm_scratch
})

df_results = pd.DataFrame(results).sort_values('f1_macro', ascending=False)
print("\nBảng so sánh (F1 macro & Test Loss)")
print(df_results.to_string(index=False))

examples = [
    "sorry but i think you've got that not right.",
    "is starbucks stock up or down from last quarter",
   "find new trump articles but not from fox news",
]

def predict_all(texts):
    # TF-IDF LR
    pred_tfidf = tfidf_lr_pipeline.predict(texts)
    # Word2Vec avg
    X_ex_avg = np.array([sentence_to_avg_vector(t, w2v_model) for t in texts])
    pred_w2v = np.argmax(model_dense.predict(X_ex_avg), axis=1)
    # LSTM pre
    seqs = tokenizer.texts_to_sequences(texts)
    pad = pad_sequences(seqs, maxlen=max_len, padding='post')
    pred_lstm_pre = np.argmax(lstm_pretrained.predict(pad), axis=1)
    # LSTM scratch
    pred_lstm_scratch = np.argmax(lstm_scratch.predict(pad), axis=1)
    return pred_tfidf, pred_w2v, pred_lstm_pre, pred_lstm_scratch

preds = predict_all(examples)

for i, ex in enumerate(examples):
    # Lấy True label nếu có trong test
    true_label_arr = df_test.loc[df_test['text'] == ex, 'intent'].values
    true_label = true_label_arr[0] if len(true_label_arr) > 0 else "N/A (example)"

    print(f"\nExample: {ex}")
    print("True label:", true_label)
    print("TF-IDF + LR:", le.inverse_transform([preds[0][i]])[0])
    print("Word2Vec Avg + Dense:", le.inverse_transform([preds[1][i]])[0])
    print("Embedding Pre-trained + LSTM:", le.inverse_transform([preds[2][i]])[0])
    print("Embedding Scratch + LSTM:", le.inverse_transform([preds[3][i]])[0])
