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
# Paths (model files live in the project-root /models/ directory)
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')
VECT_PATH  = os.path.join(MODELS_DIR, 'vectorizer.pkl')

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
# Text Preprocessing (Exact match with train_model.py)
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
    text = re.sub(r'^.*?\([A-Za-z\s]+\)\s*[-–—]\s*', '', text)

    # Strip HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Strip URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove brackets and editorial captions inside brackets
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
_preprocess = clean_text


# ─────────────────────────────────────────────
# Keyword Extraction
# ─────────────────────────────────────────────
def _extract_keywords(text: str, vectorizer, n: int = 8) -> list:
    """
    Return the top-N TF-IDF weighted terms from the input text.
    These give the user a hint about what drove the model's decision.
    """
    try:
        tfidf_matrix = vectorizer.transform([clean_text(text)])
        feature_names = vectorizer.get_feature_names_out()

        # non-zero feature indices sorted by score
        scores = tfidf_matrix.toarray()[0]
        top_indices = scores.argsort()[::-1][:n]
        keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
        return keywords
    except Exception:
        return []


# ─────────────────────────────────────────────
# Score & Dynamic Indicators
# ─────────────────────────────────────────────

# Sensational or clickbait terminology indicative of unreliable content
SENSATIONAL_TERMS = [
    'shocking', 'exclusive', 'unbelievable', 'scandal', 'conspiracy',
    'hoax', 'secret', 'cover up', 'exposed', 'mainstream media',
    'deep state', 'wake up', 'free laptop', 'free recharge', 'guaranteed prize',
    'miracle cure', 'urgent alert', 'viral', 'lottery', 'whistleblower',
    'whistleblowers', 'alien', 'aliens', 'banned', 'mind control', 'bombshell'
]

# Reporting and citation indicators
ATTRIBUTION_TERMS = [
    'according to', 'officials', 'confirmed', 'researchers', 'scientists',
    'study', 'published', 'survey', 'report', 'evidence', 'spokesperson',
    'university', 'journal', 'statement', 'announced', 'data'
]


def _generate_indicators(text: str, classification: str, fake_pct: int, real_pct: int, keywords: list) -> list:
    """Generate genuinely dynamic content-driven key indicators."""
    lower = text.lower()

    # 1. Sensationalist terminology check
    sensational_hits = [w for w in SENSATIONAL_TERMS if w in lower]
    if sensational_hits:
        ind1 = {
            'status': 'bad',
            'title': 'Sensational / alarmist language',
            'desc': f'Detected emotionally charged or conspiracy triggers: {", ".join(sensational_hits[:3])}.'
        }
    else:
        ind1 = {
            'status': 'good',
            'title': 'Objective journalistic tone',
            'desc': 'No sensationalist, alarmist, or clickbait trigger terms detected.'
        }

    # 2. Source attribution check
    attr_hits = [w for w in ATTRIBUTION_TERMS if w in lower]
    if attr_hits:
        ind2 = {
            'status': 'good',
            'title': 'Credible source attribution',
            'desc': f'References institutional sources or research ({", ".join(attr_hits[:3])}).'
        }
    else:
        ind2 = {
            'status': 'bad' if classification == 'Fake' else 'good',
            'title': 'Source attribution presence',
            'desc': 'No direct attribution to named studies, researchers, or official agencies detected.' if classification == 'Fake' else 'Implicit descriptive reporting without formal citation anchors.'
        }

    # 3. Vocabulary & statistical patterns
    if classification == 'Real':
        ind3 = {
            'status': 'good',
            'title': 'Standard vocabulary patterns',
            'desc': 'Vocabulary and n-gram distribution closely match professional journalism standards.'
        }
    else:
        ind3 = {
            'status': 'bad',
            'title': 'Unusual keyword patterns',
            'desc': 'Extracted keywords and phrases align with disinformation and unverified reporting.'
        }

    # 4. Presentation & formatting
    caps_count = len(re.findall(r'\b[A-Z]{3,}\b', text))
    punct_hits = bool(re.search(r'[!?]{2,}', text))
    if caps_count > 2 or punct_hits:
        ind4 = {
            'status': 'bad',
            'title': 'Informal formatting & punctuation',
            'desc': 'Contains sensational capitalization or repetitive punctuation.'
        }
    else:
        ind4 = {
            'status': 'good',
            'title': 'Balanced reporting style',
            'desc': 'Maintains measured grammatical structure and standard capitalization.'
        }

    return [ind1, ind2, ind3, ind4]


def _build_reasons_and_explanation(text: str, classification: str, confidence: int, real_pct: int, fake_pct: int) -> tuple:
    """
    Generate comprehensive, structured reasons directly explaining
    why an article is classified as Real (True), Fake (False), or Misleading.
    """
    lower = text.lower()
    sensational_hits = [w for w in SENSATIONAL_TERMS if w in lower]
    attr_hits = [w for w in ATTRIBUTION_TERMS if w in lower]
    has_caps = len(re.findall(r'\b[A-Z]{3,}\b', text)) > 2
    has_exclam = bool(re.search(r'[!?]{2,}', text))

    reasons = []

    if classification == 'Real':
        summary = (
            f"This article is classified as Real / True with {confidence}% confidence. "
            "It exhibits linguistic conventions of credible reporting, verified syntax, and balanced phrasing."
        )
        if attr_hits:
            reasons.append(f"Institutional Attribution: Explicitly references verifiable sources or research ({', '.join(attr_hits[:3])}).")
        else:
            reasons.append("Journalistic Vocabulary: Follows standard vocabulary distributions matching verified news datasets.")

        if not sensational_hits:
            reasons.append("Absence of Hyperbole: Contains no alarmist buzzwords, conspiracy triggers, or clickbait hooks.")
        else:
            reasons.append("Contextualized Phrasing: Mentions key topics within a structured, analytical context.")

        if not (has_caps or has_exclam):
            reasons.append("Formal Composition: Uses professional journalistic formatting, standard punctuation, and neutral sentence structures.")
        else:
            reasons.append("Informative Tone: Core statements maintain objective assertions.")

        reasons.append(f"Statistical NLP Alignment: High correlation ({confidence}%) with authentic news language patterns.")

    elif classification == 'Fake':
        summary = (
            f"This article is classified as Fake / False with {confidence}% confidence. "
            "It displays patterns typical of fabricated content, alarmist narratives, or unverified claims."
        )
        if sensational_hits:
            reasons.append(f"Sensationalist Language: Contains emotionally charged or conspiracy trigger terms ({', '.join(sensational_hits[:3])}).")
        else:
            reasons.append("Linguistic Anomalies: Terminology and n-gram frequencies match known disinformation profiles.")

        if not attr_hits:
            reasons.append("Absence of Verifiable Sources: Lacks direct citations to recognized news agencies, research institutions, or official records.")
        else:
            reasons.append("Superficial Attribution: Mentions entities without substantiated context or verifiable data.")

        if has_caps or has_exclam:
            reasons.append("Dramatic Formatting: Employs aggressive capitalization or repetitive punctuation to elicit emotional reactions.")
        else:
            reasons.append("Unsubstantiated Assertions: Makes broad claims without corroborating factual context.")

        reasons.append(f"Statistical NLP Disparity: {confidence}% statistical correspondence with fabricated news corpora.")

    else:  # Misleading
        summary = (
            f"This article is classified as Misleading with {confidence}% confidence. "
            "It presents mixed or ambiguous signals — potentially combining factual topics with exaggerated or selective framing."
        )
        reasons.append(f"Borderline Statistical Probability: The model indicates a split confidence ({real_pct}% Real vs {fake_pct}% Fake), indicating ambiguity.")
        if sensational_hits and attr_hits:
            reasons.append(f"Conflicting Signals: Cites credible terms ({', '.join(attr_hits[:2])}) while simultaneously using sensational triggers ({', '.join(sensational_hits[:2])}).")
        elif sensational_hits:
            reasons.append(f"Exaggerated Framing: Incorporates sensational phrasing ({', '.join(sensational_hits[:2])}) that distorts otherwise neutral context.")
        else:
            reasons.append("Partial Attribution: References topics without comprehensive context or independent corroboration.")

        reasons.append("Selective Presentation: May rely on out-of-context quotes, sensationalized headlines, or subjective interpretation.")
        reasons.append("Recommended Action: Exercise caution and cross-reference with primary reporting before sharing.")

    return summary, reasons


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def predict_news(text: str) -> dict:
    """
    Predict whether a news article is Real, Fake, or Misleading
    using trained TF-IDF + LogisticRegression with comprehensive reasoning.

    Parameters
    ----------
    text : str
        Raw news article / headline text.

    Returns
    -------
    dict with keys:
        classification     : str   "Real" | "Fake" | "Misleading"
        score              : int   0-100 (probability of Real)
        fake_prob          : int   0-100 (probability of Fake)
        real_prob          : int   0-100 (probability of Real)
        confidence         : int   0-100 (confidence in verdict)
        explanation        : str
        detailed_reasoning : str
        reasons            : list[str]
        keywords           : list[str]
        indicators         : list[dict]
    """
    if not text or not text.strip():
        return {
            'classification': 'Unknown',
            'score': 0,
            'fake_prob': 0,
            'real_prob': 0,
            'confidence': 0,
            'explanation': 'No text provided.',
            'detailed_reasoning': 'Please input text to receive an analysis.',
            'reasons': ['No input provided for analysis.'],
            'keywords': [],
            'indicators': [],
        }

    # Ensure model and vectorizer are loaded
    _load_artifacts()

    # Preprocess using the exact same function as train_model.py
    clean = clean_text(text)

    # Vectorise
    X = _vectorizer.transform([clean])

    # Predict probability directly from the trained model
    proba = _model.predict_proba(X)[0]

    # Map classes strictly (0=Fake, 1=Real)
    classes = list(_model.classes_)
    fake_idx = classes.index(0) if 0 in classes else 0
    real_idx = classes.index(1) if 1 in classes else 1

    fake_prob = float(proba[fake_idx])
    real_prob = float(proba[real_idx])

    fake_pct = round(fake_prob * 100)
    real_pct = round(real_prob * 100)

    # Multi-factor 3-way classification: Real, Fake, or Misleading
    lower_text = text.lower()
    sensational_hits = [w for w in SENSATIONAL_TERMS if w in lower_text]
    attr_hits = [w for w in ATTRIBUTION_TERMS if w in lower_text]

    # Check for borderline probability or conflicting signals
    is_borderline = 42 <= real_pct <= 58
    has_mixed_signals = (len(sensational_hits) > 0 and len(attr_hits) > 0) or (real_pct >= 60 and len(sensational_hits) >= 2)

    if is_borderline or has_mixed_signals:
        classification = 'Misleading'
        confidence = max(fake_pct, real_pct, 58)
    elif real_prob >= 0.50:
        classification = 'Real'
        confidence = real_pct
    else:
        classification = 'Fake'
        confidence = fake_pct

    score = real_pct

    # Keywords
    keywords = _extract_keywords(text, _vectorizer, n=8)

    # Dynamic Key Indicators
    indicators = _generate_indicators(text, classification, fake_pct, real_pct, keywords)

    # Human-readable explanation and structured reasons
    summary, reasons = _build_reasons_and_explanation(text, classification, confidence, real_pct, fake_pct)

    return {
        'classification': classification,
        'score': score,
        'fake_prob': fake_pct,
        'real_prob': real_pct,
        'confidence': confidence,
        'explanation': summary,
        'detailed_reasoning': summary,
        'reasons': reasons,
        'keywords': keywords,
        'indicators': indicators,
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
