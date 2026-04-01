# 🔍 TruthLens — Fake News & Misinformation Detector

A complete full-stack misinformation detection system powered by **TF-IDF + Logistic Regression**.

---

## 📁 Project Structure

```
fake-news-detector/
│
├── frontend/
│   ├── index.html       ← Main UI
│   ├── style.css        ← Dark editorial theme
│   └── script.js        ← Fetch API + history + URL detection
│
├── backend/
│   ├── app.py           ← Flask REST API
│   ├── model.py         ← Predict function + keyword extraction
│   ├── train_model.py   ← TF-IDF + LogReg training script
│   └── requirements.txt
│
└── dataset/
    ├── Fake.csv         ← Download from Kaggle (see below)
    └── True.csv         ← Download from Kaggle (see below)
```

---

## 🗂 Dataset

This project uses the **ISOT Fake News Dataset**.

1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. Download and extract
3. Place `Fake.csv` and `True.csv` inside the `dataset/` folder

---

## ⚙️ Setup & Installation

### 1. Clone / Download the project
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install Python dependencies
```bash
cd backend
pip install -r requirements.txt
```

---

## 🧠 Train the Model

Make sure `Fake.csv` and `True.csv` are in `dataset/` before running.

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
✂️   Splitting into train / test (80/20)…
  ✔ Train: 35,918  |  Test: 8,980
🧠  Training Logistic Regression…
  ✔ Training complete

📊  Test Accuracy : ~98-99%
...
💾  Saving model and vectoriser…
  ✔ model.pkl
  ✔ vectorizer.pkl

✅  Done! Backend is ready to serve predictions.
```

This creates `backend/model.pkl` and `backend/vectorizer.pkl`.

---

## 🚀 Run the Backend

```bash
cd backend
python app.py
```

Server will start at: **http://127.0.0.1:5000**

Verify it's running:
```bash
curl http://127.0.0.1:5000/health
```

---

## 🌐 Open the Frontend

Simply open the HTML file in your browser:

```bash
# macOS
open frontend/index.html

# Windows
start frontend/index.html

# Linux
xdg-open frontend/index.html
```

Or serve it with Python's built-in server:
```bash
cd frontend
python -m http.server 3000
# Open: http://localhost:3000
```

---

## 🔌 API Reference

### `POST /detect`
Analyse a news article.

**Request:**
```json
{ "text": "Your news article or headline here..." }
```

**Response:**
```json
{
  "classification": "Real",
  "score": 87,
  "explanation": "The model assigned a high credibility score...",
  "keywords": ["study", "confirmed", "according", "officials", "data"]
}
```

---

### `POST /fetch-url`
Extract article text from a URL (requires `requests` + `beautifulsoup4`).

**Request:**
```json
{ "url": "https://www.bbc.com/news/example-article" }
```

**Response:**
```json
{ "text": "Extracted article text..." }
```

---

### `GET /health`
Health check.

```json
{
  "status": "ok",
  "model_ready": true,
  "history_count": 5,
  "timestamp": "2024-01-15T10:30:00"
}
```

---

### `GET /history`
Get last N detections (default 20).

### `DELETE /history`
Clear server-side history.

---

## 🎯 Features

| Feature | Description |
|---|---|
| ✅ Text analysis | Paste any news article or headline |
| ✅ Credibility score | 0–100 percentage bar |
| ✅ Classification | Real / Fake / Misleading |
| ✅ Explanation | Human-readable reasoning |
| ✅ Key indicators | Top TF-IDF weighted terms |
| ✅ URL detection | Auto-fetch article from URL |
| ✅ History panel | In-memory per-session history |
| ✅ Keyboard shortcut | Ctrl+Enter to submit |

---

## 🧪 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3 (custom dark theme), Vanilla JS |
| Backend | Python 3.9+, Flask, Flask-CORS |
| ML | scikit-learn (TF-IDF + Logistic Regression) |
| Dataset | ISOT Fake News Dataset (Kaggle) |

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**. The ML model can make mistakes — always verify information with multiple trusted news sources before forming conclusions.

---

## 📄 License

MIT License — free to use and modify.
