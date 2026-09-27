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

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ─────────────────────────────────────────────
# 1. File Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check ./dataset first, then fallback to ../dataset
if os.path.isdir(os.path.join(BASE_DIR, 'dataset')):
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
else:
    DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset')

FAKE_PATH  = os.path.join(DATASET_DIR, 'Fake.csv')
TRUE_PATH  = os.path.join(DATASET_DIR, 'True.csv')

MODEL_OUT  = os.path.join(BASE_DIR, 'model.pkl')
VECT_OUT   = os.path.join(BASE_DIR, 'vectorizer.pkl')


# ─────────────────────────────────────────────
# 2. Text Preprocessing & Publisher Leakage Scrubbing
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Robust text cleaning to prevent data leakage and false positives:
    - Strips wire service prefixes (e.g. 'WASHINGTON (Reuters) - ')
    - Removes publisher watermarks ('reuters', 'reuters.com', 'via twitter', etc.)
    - Removes HTML tags, URLs, brackets and bracketed captions
    - Removes punctuation and digits
    - Normalizes whitespace and lowercases
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ''

    # Strip wire service prefixes / datelines at the beginning of the text
    # e.g., "WASHINGTON (Reuters) - ", "LONDON (Reuters) — ", "BEIJING (AP) - "
    text = re.sub(r'^.*?\([A-Za-z\s]+\)\s*[-–—]\s*', '', text)

    # Strip HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Strip URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove brackets and editorial captions inside brackets (e.g. [IMAGE], [Reuters], etc.)
    text = re.sub(r'\[.*?\]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove all occurrences of publisher watermarks and wire references
    watermarks = [
        r'\breuters(?:\.com)?\b',
        r'\bvia twitter\b',
        r'\bfeatured image via\b',
        r'\bgetty images\b',
        r'\bassociated press\b',
        r'\bap\b',
        r'\bphoto by\b',
        r'\bimage via\b',
    ]
    for wm in watermarks:
        text = re.sub(wm, ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # Remove digits
    text = re.sub(r'\d+', ' ', text)

    # Collapse multiple whitespaces and strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Backwards compatibility alias
preprocess = clean_text


# ─────────────────────────────────────────────
# 3. Load & Merge Dataset
# ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("[*] Loading dataset...")

    if not os.path.exists(FAKE_PATH) or not os.path.exists(TRUE_PATH):
        raise FileNotFoundError(
            f"Dataset files not found.\n"
            f"Expected:\n  {FAKE_PATH}\n  {TRUE_PATH}\n"
            "Download the ISOT Fake News Dataset from Kaggle and place "
            "Fake.csv and True.csv in the dataset/ folder."
        )

    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    print(f"    Raw counts: {len(fake_df):,} fake / {len(true_df):,} real")

    # Clean text and title separately so start-of-text wire heads are cleanly removed
    print("[*] Scrubbing publisher watermarks and wire headers from True articles...")
    true_cleaned_titles = true_df['title'].fillna('').apply(clean_text)
    true_cleaned_texts = true_df['text'].fillna('').apply(clean_text)
    true_df['clean'] = (true_cleaned_titles + ' ' + true_cleaned_texts).str.strip()

    print("[*] Scrubbing publisher watermarks from Fake articles...")
    fake_cleaned_titles = fake_df['title'].fillna('').apply(clean_text)
    fake_cleaned_texts = fake_df['text'].fillna('').apply(clean_text)
    fake_df['clean'] = (fake_cleaned_titles + ' ' + fake_cleaned_texts).str.strip()

    # Assign labels:  Fake = 0,  Real = 1
    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([
        fake_df[['clean', 'label']],
        true_df[['clean', 'label']]
    ], ignore_index=True)

    # Drop any empty cleaned texts
    df = df[df['clean'].str.len() > 10].copy()

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[+] Prepared {len(df):,} valid articles")
    return df


# ─────────────────────────────────────────────
# 4. Feature Engineering
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame):
    print("[*] Fitting TF-IDF vectorizer (tuned pipeline)...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        stop_words='english',
    )
    X = vectorizer.fit_transform(df['clean'])
    y = df['label'].values
    print(f"[+] Feature matrix: {X.shape[0]:,} samples x {X.shape[1]:,} features")
    return X, y, vectorizer


# ─────────────────────────────────────────────
# 5. Train
# ─────────────────────────────────────────────
def train(X, y):
    print("[*] Splitting into train / test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[+] Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    print("[*] Training Logistic Regression (balanced, C=1.0, fit_intercept=False)...")
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight='balanced',
        fit_intercept=False,
        solver='lbfgs',
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("[+] Training complete")

    # Verify class indexing
    print(f"[+] Model classes: {model.classes_} (0=Fake, 1=Real)")
    assert list(model.classes_) == [0, 1], f"Unexpected model.classes_: {model.classes_}"

    # ── Evaluation ──
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[+] Test Accuracy : {acc * 100:.2f}%")
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
    print("\n[*] Saving model and vectorizer...")
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(model, f)
    with open(VECT_OUT, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"[+] model.pkl     -> {MODEL_OUT}")
    print(f"[+] vectorizer.pkl-> {VECT_OUT}")


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 52)
    print("  TruthLens - Model Training Script")
    print("=" * 52)

    df = load_data()
    X, y, vectorizer = build_features(df)
    model = train(X, y)
    save_artifacts(model, vectorizer)

    print("\n[+] Done! Backend is ready to serve predictions.")
    print("    Run:  python app.py")

