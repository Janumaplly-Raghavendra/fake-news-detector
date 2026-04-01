"""
train_model.py
==============
Trains a TF-IDF + Logistic Regression classifier on the ISOT Fake News dataset.

Dataset files required (place in  ../dataset/):
  - Fake.csv   (columns: title, text, subject, date)
  - True.csv   (columns: title, text, subject, date)

Outputs (saved in ./  i.e. backend/):
  - model.pkl
  - vectorizer.pkl

Usage:
  cd backend
  python train_model.py
"""

import os
import re
import string
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ─────────────────────────────────────────────
# 1. File Paths
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset')

FAKE_PATH  = os.path.join(DATASET_DIR, 'Fake.csv')
TRUE_PATH  = os.path.join(DATASET_DIR, 'True.csv')

MODEL_OUT  = os.path.join(BASE_DIR, 'model.pkl')
VECT_OUT   = os.path.join(BASE_DIR, 'vectorizer.pkl')


# ─────────────────────────────────────────────
# 2. Text Preprocessing
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    Lowercase → remove URLs → remove punctuation → remove digits → strip whitespace.
    """
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────
# 3. Load & Merge Dataset
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("📂  Loading dataset…")

    if not os.path.exists(FAKE_PATH) or not os.path.exists(TRUE_PATH):
        raise FileNotFoundError(
            f"Dataset files not found.\n"
            f"Expected:\n  {FAKE_PATH}\n  {TRUE_PATH}\n"
            "Download the ISOT Fake News Dataset from Kaggle and place "
            "Fake.csv and True.csv in the dataset/ folder."
        )

    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    # Assign labels:  Fake = 0,  Real = 1
    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Combine title + text for richer features
    df['combined'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  ✔ Loaded {len(df):,} articles  "
          f"({fake_df.shape[0]:,} fake / {true_df.shape[0]:,} real)")
    return df


# ─────────────────────────────────────────────
# 4. Feature Engineering
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame):
    print("🔧  Preprocessing text…")
    df['clean'] = df['combined'].apply(preprocess)
    print("  ✔ Text preprocessing complete")

    print("📐  Fitting TF-IDF vectoriser…")
    vectorizer = TfidfVectorizer(
        max_features=50_000,   # vocabulary size
        ngram_range=(1, 2),    # unigrams + bigrams
        sublinear_tf=True,     # log-scale TF
        min_df=2,              # ignore very rare terms
        max_df=0.95,           # ignore near-universal terms
        stop_words='english',
    )
    X = vectorizer.fit_transform(df['clean'])
    y = df['label'].values
    print(f"  ✔ Feature matrix: {X.shape[0]:,} samples × {X.shape[1]:,} features")
    return X, y, vectorizer


# ─────────────────────────────────────────────
# 5. Train
# ─────────────────────────────────────────────
def train(X, y):
    print("✂️   Splitting into train / test (80/20)…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  ✔ Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    print("🧠  Training Logistic Regression…")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,           # regularisation
        solver='lbfgs',
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("  ✔ Training complete")

    # ── Evaluation ──
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊  Test Accuracy : {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")

    return model


# ─────────────────────────────────────────────
# 6. Save
# ─────────────────────────────────────────────
def save_artifacts(model, vectorizer):
    print("\n💾  Saving model and vectoriser…")
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(model, f)
    with open(VECT_OUT, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  ✔ model.pkl     → {MODEL_OUT}")
    print(f"  ✔ vectorizer.pkl→ {VECT_OUT}")


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 52)
    print("  TruthLens  ·  Model Training Script")
    print("=" * 52)

    df = load_data()
    X, y, vectorizer = build_features(df)
    model = train(X, y)
    save_artifacts(model, vectorizer)

    print("\n✅  Done! Backend is ready to serve predictions.")
    print("   Run:  python app.py")
