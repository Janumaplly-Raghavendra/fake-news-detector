<div align="center">

# 🔍 TruthLens — AI Fake News Detector

**A full-stack misinformation detection system powered by NLP and Machine Learning.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> Paste any news article, URL, or headline — get an instant **Real / Fake / Misleading** verdict with transparent AI reasoning.

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **AI Classification** | TF-IDF + Logistic Regression trained on 44,898 real-world news articles |
| 🎯 **3-Way Verdict** | Classifies as **Real**, **Fake**, or **Misleading** with confidence score |
| 💬 **Transparent Reasoning** | Bullet-point explanations for *why* an article is true, false, or misleading |
| 🔑 **Key Indicators** | Shows linguistic signals — source attribution, sensationalism, formatting |
| 🌐 **URL Fetch** | Auto-scrapes and analyses article text directly from a news URL |
| 📁 **File Upload** | Upload a `.txt` file and analyse its contents |
| 🌙 **Dark / Light Mode** | Toggleable theme with `localStorage` persistence |
| 📐 **Resizable Results** | Drag or use `+` / `−` buttons to resize the result panel |
| 🕓 **Session History** | Recent analyses stored per-session with one-click re-analysis |
| ⌨️ **Keyboard Shortcut** | `Ctrl + Enter` to submit analysis |

---

## 📁 Project Structure

```
fake-news-detector/
│
├── 📁 frontend/                  # Browser UI (no build step required)
│   ├── index.html                ← Single-page app structure
│   ├── style.css                 ← Dark/light theme, animations, layout
│   └── script.js                 ← UI logic, theme toggle, result rendering
│
├── 📁 backend/                   # Python REST API
│   ├── app.py                    ← Flask server — /detect, /fetch-url, /health
│   ├── model.py                  ← predict_news(), reasoning, indicators
│   ├── train_model.py            ← Full training on ISOT dataset (~92–99% accuracy)
│   └── mock_train.py             ← Quick fallback training (no dataset needed)
│
├── 📁 models/                    # Serialized ML artifacts
│   ├── model.pkl                 ← Trained Logistic Regression model
│   └── vectorizer.pkl            ← Fitted TF-IDF vectorizer
│
├── 📁 dataset/                   # ISOT dataset (download separately — see below)
│   ├── Fake.csv                  ← 23,481 fake news articles
│   └── True.csv                  ← 21,417 real news articles
│
├── 📁 tests/                     # Utility & sanity test scripts
│   ├── verify_test.py            ← Automated prediction sanity checks
│   ├── check_score.py            ← Quick confidence score tester
│   └── test_headline.py         ← Single headline quick test
│
├── requirements.txt              ← Python dependencies
├── .gitignore
└── README.md
```

---

## 🗂 Dataset

This project uses the **ISOT Fake News Dataset** from Kaggle.

1. Go to: [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Download and extract the archive
3. Place `Fake.csv` and `True.csv` inside the `dataset/` folder

> ⚠️ The dataset files are excluded from this repository via `.gitignore` due to their size (~100MB).

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Janumaplly-Raghavendra/fake-news-detector.git
cd fake-news-detector
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate — Windows:
venv\Scripts\activate

# Activate — macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

### Option A — Full Training (recommended, requires Kaggle dataset)
```bash
cd backend
python train_model.py
```

Expected output:
```
====================================================
  TruthLens  ·  Model Training Script
====================================================
📂  Loading dataset…
  ✔ Loaded 44,898 articles  (23,481 fake / 21,417 real)
🔧  Preprocessing text…
  ✔ Text preprocessing complete
📐  Fitting TF-IDF vectoriser…
  ✔ Feature matrix: 44,898 samples × 50,000 features
🧠  Training Logistic Regression…
  ✔ Training complete

📊  Test Accuracy: ~92–99%
💾  model.pkl  ✔
💾  vectorizer.pkl  ✔

✅  Done! Backend is ready to serve predictions.
```

### Option B — Quick Mock Training (no dataset needed)
```bash
cd backend
python mock_train.py
```
> Trains on a tiny 16-sample dummy dataset. Useful for testing the app instantly without downloading the dataset. Accuracy will be limited.

Both options save `model.pkl` and `vectorizer.pkl` to the `models/` folder.

---

## 🚀 Run the Backend

```bash
cd backend
python app.py
```

The Flask API starts at: **`http://127.0.0.1:5000`**

Verify it's alive:
```bash
curl http://127.0.0.1:5000/health
```

---

## 🌐 Open the Frontend

With the backend running, open the UI in your browser:

```bash
# Windows
start frontend/index.html

# macOS
open frontend/index.html

# Linux
xdg-open frontend/index.html
```

Or just navigate to **`http://127.0.0.1:5000`** directly — Flask also serves the frontend.

---

## 🔌 API Reference

### `POST /detect`
Analyse a news article or headline.

**Request:**
```json
{ "text": "NASA confirmed water was found on Mars according to official research." }
```

**Response:**
```json
{
  "classification": "Real",
  "confidence": 87,
  "real_prob": 87,
  "fake_prob": 13,
  "score": 87,
  "reasons": [
    "Institutional Attribution: References verifiable sources (NASA, research).",
    "Absence of Hyperbole: No alarmist buzzwords detected.",
    "Formal Composition: Standard journalistic formatting."
  ],
  "detailed_reasoning": "Classified as Real with 87% confidence...",
  "indicators": [...],
  "keywords": ["nasa", "confirmed", "research", "water", "mars"]
}
```

---

### `POST /fetch-url`
Scrape article text from a news URL.

**Request:**
```json
{ "url": "https://example.com/some-news-article" }
```

**Response:**
```json
{ "text": "Extracted article body text..." }
```

---

### `GET /health`
Returns server and model status.

### `GET /history`
Returns in-memory analysis history for the current session.

### `DELETE /history`
Clears the analysis history.

---

## 🧪 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3 (custom dark/light theme), Vanilla JavaScript |
| **Backend** | Python 3.9+, Flask 2.3+, Flask-CORS |
| **ML** | scikit-learn — TF-IDF Vectorizer + Logistic Regression |
| **NLP** | Publisher-watermark scrubbing, heuristic indicator extraction |
| **Dataset** | ISOT Fake News Dataset (Kaggle) — 44,898 articles |
| **Scraping** | requests + BeautifulSoup4 + lxml |

---

## 🧬 How It Works

```
User Input (text / URL / file)
        │
        ▼
 [ Publisher Tag Scrubbing ]     ← strips Reuters/AP wire headers
        │
        ▼
 [ TF-IDF Vectorization ]        ← converts text to 50,000-feature vector
        │
        ▼
 [ Logistic Regression ]         ← predicts Real / Fake probabilities
        │
        ▼
 [ Heuristic Signal Analysis ]   ← checks sensationalism, attribution, formatting
        │
        ▼
 [ 3-Way Classification ]        ← Real | Fake | Misleading (mixed signals)
        │
        ▼
 [ Reason Generation ]           ← bullet-point explanation returned to UI
```

**Misleading** is triggered when:
- Confidence is borderline (`42% ≤ real% ≤ 58%`), **OR**
- Credible source terms appear alongside sensationalist triggers (conflicting signals)

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**. The ML model is not infallible — it may misclassify satire, opinion pieces, or highly technical content. Always verify important information against multiple trusted primary sources before forming conclusions or sharing.

---

## 📄 License

MIT License — free to use, modify, and distribute.
