# 🧠 Twitter Sentiment Analysis - CNN Model Training (Optimized + Fixed)

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Enable GPU memory growth (for smoother GPU usage)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("⚡ GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# Step 2: Load Cleaned Dataset
print("📥 Loading cleaned dataset...")
df = pd.read_csv("../dataset/cleaned_sentiment140.csv")
df.dropna(inplace=True)
texts = df["clean_text"].astype(str)
labels = df["sentiment"]

print(f"✅ Dataset Loaded — {len(df)} samples")

# Step 3: Tokenization and Padding
vocab_size = 20000
max_len = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')
y = np.array(labels)

# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split into training and testing sets")

# Step 5: Build CNN Model (Fixed vocab_size + 1)
model = Sequential([
    Embedding(vocab_size + 1, 128, input_length=max_len),  # ✅ Fix for OOV index
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Step 6: Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Step 7: Train Model (Faster training with prefetch)
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(512).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(512).prefetch(tf.data.AUTOTUNE)

history = model.fit(train_ds, epochs=5, validation_data=test_ds, verbose=1)

# Step 8: Save Model in New Keras Format
# Save model safely without optimizer state
model.save("../models/cnn_sentiment_model.keras", include_optimizer=False)

print("✅ Model saved successfully at ../models/cnn_sentiment_model.keras")

# Step 9: Evaluate Model
loss, acc = model.evaluate(test_ds)
print(f"\n📊 Test Accuracy: {acc*100:.2f}%")

# Step 10: Plot Accuracy & Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
