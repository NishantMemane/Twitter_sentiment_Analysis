import os

# Define the folder structure
folders = [
    "dataset",
    "models",
    "notebooks",
    "scripts",
    "webapp"
]

files = {
    "requirements.txt": "",
    "README.md": "# Twitter Sentiment Analysis (CNN-based)\n",
    ".gitignore": "*.pyc\n__pycache__/\nmodels/\n",
    "webapp/app.py": "",
    "webapp/style.css": "",
    "scripts/fetch_tweets.py": "",
    "scripts/predict_sentiment.py": "",
    "scripts/preprocess.py": "",
    "notebooks/data_preprocessing.ipynb": "",
    "notebooks/model_training.ipynb": "",
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for path, content in files.items():
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    with open(path, "w") as f:
        f.write(content)

print("✅ Project folder structure created successfully!")
