import tensorflow as tf
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

# ===============================
# 1. Load Cleaned Dataset
# ===============================
print("📥 Loading cleaned Sentiment140 dataset...")

data = pd.read_csv(r"C:\Users\shree\OneDrive\Desktop\Original_twitter\dataset\cleaned_sentiment140.csv")

print("✅ Columns found:", data.columns.tolist())

# Expecting columns: ['clean_text', 'sentiment'] or similar
# Rename if needed
if 'clean_text' in data.columns and 'sentiment' in data.columns:
    data = data.rename(columns={'clean_text': 'text', 'sentiment': 'target'})
elif len(data.columns) == 2:
    data.columns = ['text', 'target']

# Remove NaNs
data.dropna(inplace=True)

# ===============================
# 2. Preprocessing
# ===============================
def clean_text(text):
    text = re.sub(r"http\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    return text.lower().strip()

data['text'] = data['text'].apply(clean_text)

# If sentiment labels are like 0, 4 — convert to 0, 1, 2 (negative, neutral, positive)
unique_labels = sorted(data['target'].unique())
label_map = {label: idx for idx, label in enumerate(unique_labels)}
data['target'] = data['target'].map(label_map)

print("✅ Label mapping:", label_map)

# ===============================
# 3. Tokenization & Padding
# ===============================
tokenizer = Tokenizer(num_words=50000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')

X = np.array(padded)
y = np.array(data['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 4. Build CNN Model
# ===============================
model = Sequential([
    Embedding(50000, 128, input_length=50),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(label_map), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ===============================
# 5. Train Model
# ===============================
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=512,
    validation_split=0.1,
    verbose=1
)

# ===============================
# 6. Evaluate and Save
# ===============================
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

model.save(r"C:\Users\shree\OneDrive\Desktop\Original_twitter\models\cnn_sentiment_model.keras")
print("💾 Model saved successfully!")
