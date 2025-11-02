# predict_sentiment.py
import re
import os
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --- Configuration ---
MODEL_PATH = r"C:\Users\shree\OneDrive\Desktop\Original_twitter\dataset\cnn_sentiment_model1.keras"
TOKENIZER_PATH = r"C:\Users\shree\OneDrive\Desktop\Original_twitter\models\tokenizer.pkl"
CLEANED_DATASET = r"C:\Users\shree\OneDrive\Desktop\Original_twitter\dataset\cleaned_sentiment140.csv"
VOCAB_SIZE = 20000
MAX_LEN = 100
NEUTRAL_LOW = 0.35   # <= this => Negative
NEUTRAL_HIGH = 0.65  # >= this => Positive

# --- Ensure stopwords are available ---
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# --- Text cleaning (use same steps as training) ---
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"http\S+|www\S+|https\S+", "", s)       # remove URLs
    s = re.sub(r"@\w+|#\w+", "", s)                     # remove mentions & hashtags
    s = re.sub(r"[^A-Za-z\s]", " ", s)                  # keep only letters + spaces
    s = s.lower()
    s = " ".join([w for w in s.split() if w and w not in STOPWORDS])
    return s.strip()

# --- Load model safely ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print("📥 Loading model (compile=False)...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded")

# --- Load or build tokenizer ---
if os.path.exists(TOKENIZER_PATH):
    print(f"📥 Loading tokenizer from {TOKENIZER_PATH}")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    print("✅ Tokenizer loaded")
else:
    # Rebuild tokenizer from cleaned dataset as fallback, then save it
    if not os.path.exists(CLEANED_DATASET):
        raise FileNotFoundError(f"Tokenizer not found and dataset missing: {CLEANED_DATASET}")
    print("⚠️ Tokenizer not found. Rebuilding tokenizer from cleaned dataset (this may take a moment)...")
    df = pd.read_csv(CLEANED_DATASET)
    df.dropna(inplace=True)
    # choose clean_text if present, else text
    if "clean_text" in df.columns:
        texts = df["clean_text"].astype(str).tolist()
    elif "text" in df.columns:
        texts = df["text"].astype(str).tolist()
    else:
        raise KeyError("Cleaned dataset missing 'clean_text' or 'text' column.")
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Tokenizer rebuilt and saved at {TOKENIZER_PATH}")

# --- Prediction function with Neutral support ---
def predict_sentiment(text: str):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    # ensure tokens do not exceed vocab index
    seq = [[min(token, VOCAB_SIZE) if token is not None else 0 for token in s] for s in seq]
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    pred = float(model.predict(padded, verbose=0)[0][0])  # probability 0..1
    if pred >= NEUTRAL_HIGH:
        label = "😊 Positive"
    elif pred <= NEUTRAL_LOW:
        label = "😞 Negative"
    else:
        label = "😐 Neutral"
    return {"text": text, "cleaned": cleaned, "score": pred, "label": label}

# --- Interactive CLI ---
def main():
    print("\n💬 Twitter Sentiment Predictor (Positive / Neutral / Negative)")
    print("Type a tweet and press Enter. Type 'exit' to quit.")
    while True:
        s = input("\nYour Tweet: ").strip()
        if not s:
            continue
        if s.lower() in ("exit", "quit"):
            print("👋 Goodbye.")
            break
        try:
            out = predict_sentiment(s)
            print(f"\nCleaned: {out['cleaned']}")
            print(f"Prediction: {out['label']} (score: {out['score']:.4f})")
        except Exception as e:
            print("❌ Prediction error:", e)

if __name__ == "__main__":
    main()
  