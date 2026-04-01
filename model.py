"""
model.py
========
Loads the trained TF-IDF vectoriser + Logistic Regression model
and exposes a single public function: predict_news(text).

Returns a dict with:
  classification : "Real" | "Fake" | "Misleading"
  score          : int (0-100)  credibility score
  explanation    : str          human-readable reason
  keywords       : list[str]    top indicative words
"""

import os
import re
import pickle
import string
import numpy as np
from typing import Optional

# ─────────────────────────────────────────────
# Paths (model files live alongside this script)
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
VECT_PATH  = os.path.join(BASE_DIR, 'vectorizer.pkl')

# ─────────────────────────────────────────────
# Lazy-loaded globals (loaded once on first use)
# ─────────────────────────────────────────────
_model       = None
_vectorizer  = None


def _load_artifacts():
    """Load model and vectorizer from disk if not already cached."""
    global _model, _vectorizer

    if _model is not None and _vectorizer is not None:
        return  # already loaded

    for path, label in [(MODEL_PATH, 'model.pkl'), (VECT_PATH, 'vectorizer.pkl')]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{label}' not found at {path}.\n"
                "Please run:  python train_model.py  to generate it first."
            )

    with open(MODEL_PATH, 'rb') as f:
        _model = pickle.load(f)
    with open(VECT_PATH, 'rb') as f:
        _vectorizer = pickle.load(f)


# ─────────────────────────────────────────────
# Text Preprocessing (must match train_model.py)
# ─────────────────────────────────────────────
def _preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────
# Keyword Extraction
# ─────────────────────────────────────────────
def _extract_keywords(text: str, vectorizer, n: int = 8) -> list:
    """
    Return the top-N TF-IDF weighted terms from the input text.
    These give the user a hint about what drove the model's decision.
    """
    try:
        tfidf_matrix = vectorizer.transform([_preprocess(text)])
        feature_names = vectorizer.get_feature_names_out()

        # non-zero feature indices sorted by score
        scores = tfidf_matrix.toarray()[0]
        top_indices = scores.argsort()[::-1][:n]
        keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
        return keywords
    except Exception:
        return []


# ─────────────────────────────────────────────
# Score → Classification + Explanation
# ─────────────────────────────────────────────

# Phrases commonly found in unreliable / sensational content
FAKE_SIGNALS = [
    'breaking', 'exclusive', 'shocking', 'unbelievable', 'scandal',
    'conspiracy', 'hoax', 'fake', 'secret', 'cover up', 'exposed',
    'they don', 'mainstream media', 'deep state', 'wake up',
]

# Phrases more common in reliable journalism
REAL_SIGNALS = [
    'according to', 'officials said', 'statement', 'confirmed',
    'research shows', 'data shows', 'study finds', 'report says',
    'sources say', 'percent', 'analysis', 'evidence', 'published',
    'LIVE', 'warns', 'reported', 'updates', 'headline', 'breaking',
]


def _build_explanation(text: str, classification: str, score: int) -> str:
    """Generate a human-readable explanation for the prediction."""
    lower = text.lower()

    fake_hits = [s for s in FAKE_SIGNALS if s in lower]
    real_hits = [s for s in REAL_SIGNALS if s in lower]

    if classification == 'Real':
        base = (
            f"The model assigned a high credibility score of {score}/100. "
        )
        if real_hits:
            signals = ', '.join(f'"{h}"' for h in real_hits[:3])
            base += (
                f"The text contains language typical of credible reporting "
                f"({signals}), suggesting it follows journalistic standards."
            )
        else:
            base += (
                "The writing style, vocabulary distribution, and structural "
                "patterns closely match verified news articles in the training data."
            )

    elif classification == 'Fake':
        base = (
            f"The model assigned a low credibility score of {score}/100. "
        )
        if fake_hits:
            signals = ', '.join(f'"{h}"' for h in fake_hits[:3])
            base += (
                f"The text contains sensationalist or alarmist language "
                f"({signals}) often associated with misinformation."
            )
        else:
            base += (
                "The vocabulary patterns, phrase structures, and statistical "
                "features resemble articles labelled as false in the training dataset."
            )

    else:  # Misleading
        base = (
            f"The model is uncertain and assigned a borderline score of {score}/100. "
            "The article contains a mix of credible and potentially misleading "
            "signals. It may be partially factual but could lack context, "
            "use selective quoting, or present information in a misleading framing."
        )

    base += (
        " Note: This is an automated ML prediction. "
        "Always cross-check with trusted news sources."
    )
    return base


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def predict_news(text: str) -> dict:
    """
    Predict whether a news article is Real, Fake, or Misleading.

    Parameters
    ----------
    text : str
        Raw news article / headline text.

    Returns
    -------
    dict with keys:
        classification : str   "Real" | "Fake" | "Misleading"
        score          : int   0-100 (credibility percentage)
        explanation    : str
        keywords       : list[str]
    """
    if not text or not text.strip():
        return {
            'classification': 'Unknown',
            'score': 0,
            'explanation': 'No text provided.',
            'keywords': [],
        }

    # Ensure model is loaded
    _load_artifacts()

    # Preprocess
    clean = _preprocess(text)

    # Vectorise
    X = _vectorizer.transform([clean])

    # Predict probability for class 1 (Real)
    proba = _model.predict_proba(X)[0]   # [P(fake), P(real)]
    real_prob = float(proba[1])
    fake_prob = float(proba[0])

    # Convert to a 0-100 credibility score
    score = round(real_prob * 100)

    # Classification thresholds
    if score >= 50:
        classification = 'Real'
    elif score >= 25:
        classification = 'Misleading'
    else:
        classification = 'Fake'

    # Keywords
    keywords = _extract_keywords(text, _vectorizer, n=8)

    # Explanation
    explanation = _build_explanation(text, classification, score)

    return {
        'classification': classification,
        'score': score,
        'explanation': explanation,
        'keywords': keywords,
    }


# ─────────────────────────────────────────────
# Quick manual test
# ─────────────────────────────────────────────
if __name__ == '__main__':
    samples = [
        "Scientists confirm that COVID-19 vaccines are safe and effective, "
        "according to data published by the CDC and WHO.",

        "SHOCKING: Government has been secretly putting mind-control chemicals "
        "in the water supply. Wake up sheeple! They don't want you to know!",

        "The president signed a new trade bill yesterday, officials confirmed. "
        "The legislation is expected to affect tariffs on imported goods.",
    ]

    for s in samples:
        result = predict_news(s)
        print(f"\n{'─'*60}")
        print(f"TEXT   : {s[:70]}…")
        print(f"CLASS  : {result['classification']}")
        print(f"SCORE  : {result['score']}/100")
        print(f"EXPLAIN: {result['explanation'][:120]}…")
        print(f"KEYS   : {result['keywords']}")
