"""
app.py
======
Flask REST API for the TruthLens Fake News Detector.

Endpoints:
  POST /detect     → Analyse news text
  POST /fetch-url  → Scrape article text from a URL (bonus)
  GET  /health     → Server health check
  GET  /history    → Return in-memory analysis history
  DELETE /history  → Clear history

Run:
  cd backend
  python app.py
"""

import os
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from model import predict_news

# ─────────────────────────────────────────────
# App Initialisation
# ─────────────────────────────────────────────
app = Flask(__name__)

# Allow all origins during development so the plain HTML frontend can call us.
# In production, restrict to your domain: CORS(app, origins=["https://yourdomain.com"])
CORS(app)

# In-memory history (cleared on server restart)
_history = []


# ─────────────────────────────────────────────
# Helper: validate & extract JSON body
# ─────────────────────────────────────────────
def _get_json_or_400(required_keys: list):
    """Parse request JSON and validate required keys."""
    data = request.get_json(silent=True)
    if not data:
        return None, jsonify({'error': 'Request body must be JSON.'}), 400
    for key in required_keys:
        if key not in data or not str(data[key]).strip():
            return None, jsonify({'error': f'Missing required field: "{key}"'}), 400
    return data, None, None


# ─────────────────────────────────────────────
# POST /detect
# ─────────────────────────────────────────────
@app.route('/detect', methods=['POST'])
def detect():
    """
    Accepts:
        { "text": "<article text>" }

    Returns:
        {
            "classification": "Real" | "Fake" | "Misleading",
            "score": 0-100,
            "explanation": "...",
            "keywords": ["word1", "word2", ...]
        }
    """
    data, err_response, status = _get_json_or_400(['text'])
    if err_response:
        return err_response, status

    text = str(data['text']).strip()

    if len(text) < 10:
        return jsonify({'error': 'Text is too short. Please provide more content.'}), 422

    try:
        result = predict_news(text)

        # Append to server-side history
        _history.append({
            'timestamp': datetime.now().isoformat(),
            'text_snippet': text[:120],
            **result
        })
        # Keep last 100 entries
        if len(_history) > 100:
            _history.pop(0)

        return jsonify(result), 200

    except FileNotFoundError as e:
        # Model not trained yet
        return jsonify({
            'error': (
                'Model files not found. '
                'Please run: python train_model.py  to train the model first.'
            ),
            'detail': str(e)
        }), 503

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Internal server error.', 'detail': str(e)}), 500


# ─────────────────────────────────────────────
# POST /fetch-url  (Bonus: URL-based detection)
# ─────────────────────────────────────────────
@app.route('/fetch-url', methods=['POST'])
def fetch_url():
    """
    Accepts:
        { "url": "https://example.com/article" }

    Returns:
        { "text": "<extracted article text>" }
    or
        { "error": "..." }

    Requires: pip install requests beautifulsoup4
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return jsonify({
            'error': (
                'URL fetching requires extra packages. '
                'Run: pip install requests beautifulsoup4'
            )
        }), 503

    data, err_response, status = _get_json_or_400(['url'])
    if err_response:
        return err_response, status

    url = str(data['url']).strip()

    try:
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 Chrome/120.0 Safari/537.36'
            )
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Remove scripts, styles, nav, footer
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
            tag.decompose()

        # Try common article containers first
        article = (
            soup.find('article') or
            soup.find('main') or
            soup.find(class_=lambda c: c and 'article' in c.lower()) or
            soup.find('body')
        )

        if article:
            # Extract paragraphs
            paragraphs = article.find_all('p')
            text = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)
        else:
            text = soup.get_text(separator=' ', strip=True)

        # Trim to first 3000 chars to stay within reasonable request size
        text = ' '.join(text.split())[:3000]

        if not text:
            return jsonify({'error': 'Could not extract text from the provided URL.'}), 422

        return jsonify({'text': text}), 200

    except requests.exceptions.Timeout:
        return jsonify({'error': 'The request timed out. The URL may be unreachable.'}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Could not fetch the URL: {str(e)}'}), 422
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Failed to parse page content.', 'detail': str(e)}), 500


# ─────────────────────────────────────────────
# GET /history
# ─────────────────────────────────────────────
@app.route('/history', methods=['GET'])
def get_history():
    """Return the last N detection results (most recent first)."""
    limit = request.args.get('limit', default=20, type=int)
    return jsonify({
        'count': len(_history),
        'history': list(reversed(_history))[:limit]
    }), 200


# ─────────────────────────────────────────────
# DELETE /history
# ─────────────────────────────────────────────
@app.route('/history', methods=['DELETE'])
def clear_history():
    """Clear the server-side history."""
    global _history
    _history = []
    return jsonify({'message': 'History cleared.'}), 200


# ─────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    """Quick server health check."""
    import os
    model_ready = (
        os.path.exists(os.path.join(os.path.dirname(__file__), 'model.pkl')) and
        os.path.exists(os.path.join(os.path.dirname(__file__), 'vectorizer.pkl'))
    )
    return jsonify({
        'status': 'ok',
        'model_ready': model_ready,
        'history_count': len(_history),
        'timestamp': datetime.now().isoformat(),
    }), 200


# ─────────────────────────────────────────────
# Static File Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    """Serve the main UI."""
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    """Serve other static assets."""
    return send_from_directory('.', path)


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  TruthLens Backend  ·  Flask API")
    print("  http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
