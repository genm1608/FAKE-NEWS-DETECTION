import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import pickle
import time # Import the time module

nltk.download('stopwords')
nltk.download('wordnet')

# --- BƯỚC 1: TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU ---
true_df = pd.read_csv('true.csv')
fake_df = pd.read_csv('fake.csv')
true_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([true_df, fake_df], ignore_index=True)

# Kiểm tra và làm sạch dữ liệu
df.dropna(subset=['title', 'text'], inplace=True)
print(f"Tổng số bản ghi: {len(df)} (True: {len(true_df)}, Fake: {len(fake_df)})")

# Kết hợp tiêu đề và nội dung
df['content'] = df['title'] + ' ' + df['text']


# Tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Chuẩn hóa dấu cách
    text = ''.join(c for c in text if c.isalnum() or c.isspace())  # Loại bỏ ký tự đặc biệt
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
    return text


df['content'] = df['content'].apply(preprocess_text)

# Token hóa và đệm chuỗi
max_words = 10000
max_len = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['content'])
sequences = tokenizer.texts_to_sequences(df['content'])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Lưu tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

labels = df['label'].values
X_train, X_temp, y_train, y_temp = train_test_split(padded_sequences, labels, test_size=0.36, stratify=labels,
                                                     random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5556, stratify=y_temp, random_state=42)

print(f"Kích thước tập huấn luyện: {len(X_train)}")
print(f"Kích thước tập xác thực: {len(X_val)}")
print(f"Kích thước tập kiểm tra: {len(X_test)}")
print("Tỷ lệ nhãn trong tập huấn luyện:", np.bincount(y_train))
print("Tỷ lệ nhãn trong tập xác thực:", np.bincount(y_val))
print("Tỷ lệ nhãn trong tập kiểm tra:", np.bincount(y_test))

# --- BƯỚC 2: XÂY DỰNG CÁC MÔ HÌNH ---
embedding_dim = 40


def build_lstm_model():
    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_len),
        LSTM(100),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_bilstm_model():
    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_len),
        Bidirectional(LSTM(100)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_bilstm_model():
    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_len),
        Conv1D(32, 5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(100)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# --- BƯỚC 3: HUẤN LUYỆN VÀ ĐÁNH GIÁ ---
epochs = 10
batch_size = 64
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Lưu kết quả để tạo bảng tổng hợp
results = []


def train_and_evaluate(model, model_name):
    print(f"Đang huấn luyện {model_name}...")
    start_time = time.time() # Start time measurement
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)
    end_time = time.time() # End time measurement
    elapsed_time = end_time - start_time # Calculate elapsed time

    # Đánh giá trên tập kiểm tra
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Tính toán các chỉ số cho từng nhãn (0 và 1)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)

    # Tính AUC tổng thể
    auc = roc_auc_score(y_test, y_pred_prob)

    # Lưu kết quả cho từng nhãn
    for i, label in enumerate([0, 1]):
        support = np.sum(y_test == label)
        results.append({
            'Model': model_name,
            'Label': label,
            'Precision': precision[i],
            'Recall': recall[i],
            'F1-score': f1[i],
            'Support': support,
            'AUC': auc,
            'Time (s)': elapsed_time # Add elapsed time
        })

    model.save(f'{model_name}_model.h5')
    return history


# Huấn luyện và đánh giá các mô hình
lstm_model = build_lstm_model()
lstm_history = train_and_evaluate(lstm_model, "LSTM")

bilstm_model = build_bilstm_model()
bilstm_history = train_and_evaluate(bilstm_model, "Bi-LSTM")

cnn_bilstm_model = build_cnn_bilstm_model()
cnn_bilstm_history = train_and_evaluate(cnn_bilstm_model, "CNN-BiLSTM")

# --- BƯỚC 4: XUẤT BẢNG TỔNG HỢP ---
# Tạo DataFrame từ kết quả
results_df = pd.DataFrame(results)

# Hiển thị bảng tổng hợp giống hình ảnh

print(f"{'Models':<10} {'Label':<6} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'Support':<12} {'AUC':<12} {'Time (s)':<12}")
for index, row in results_df.iterrows():
    print(f"{row['Model']:<10} {int(row['Label']):<6} {row['Precision']:<12.3f} {row['Recall']:<12.3f} "
          f"{row['F1-score']:<12.3f} {int(row['Support']):<12d} {row['AUC']:<12.3f} {row['Time (s)']:<12.2f}")