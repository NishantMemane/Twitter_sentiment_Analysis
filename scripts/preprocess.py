import os
import pandas as pd
import re
import nltk
import multiprocessing as mp
from nltk.corpus import stopwords
from tqdm import tqdm

nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

# --- Load dataset ---
def load_data(path=r"C:\Users\shree\OneDrive\Desktop\Original_twitter\training.1600000.processed.noemoticon.csv"):

    print("📥 Loading dataset...")
    df = pd.read_csv(path, encoding="latin-1", header=None)
    df.columns = ["target", "ids", "date", "flag", "user", "text"]
    return df[["target", "text"]]

# --- Clean a single tweet ---
def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)
    tweet = re.sub(r"@\w+|#\w+", "", tweet)
    tweet = re.sub(r"[^A-Za-z\s]", "", tweet)
    tweet = tweet.lower()
    tweet = " ".join([w for w in tweet.split() if w not in STOPWORDS])
    return tweet

# --- Clean dataset using multiprocessing ---
def preprocess_dataset(df):
    print("🧹 Cleaning tweets using multiprocessing...")
    tweets = df["text"].tolist()

    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        cleaned = list(tqdm(pool.imap(clean_tweet, tweets), total=len(tweets)))

    df["clean_text"] = cleaned
    df["sentiment"] = df["target"].apply(lambda x: 1 if x == 4 else 0)
    return df[["clean_text", "sentiment"]]

# --- Save cleaned data ---
def save_clean_data(df, path="../dataset/cleaned_sentiment140.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Cleaned dataset saved at: {path}")

# --- Main Execution ---
if __name__ == "__main__":
    raw_df = load_data()
    clean_df = preprocess_dataset(raw_df)
    save_clean_data(clean_df)
