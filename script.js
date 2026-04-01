/**
 * TruthLens — Fake News Detector
 * script.js  ·  Vanilla JavaScript
 *
 * Responsibilities:
 *  - Form validation & UX feedback
 *  - POST to Flask /detect endpoint
 *  - Render result (classification, score, explanation, keywords)
 *  - In-memory history management
 *  - URL-to-text fetching (bonus)
 *  - Animated loading steps
 */

'use strict';

/* ─────────────────────────────────────────────
   Configuration
───────────────────────────────────────────── */
const API_BASE   = 'http://127.0.0.1:5000';
const DETECT_URL = `${API_BASE}/detect`;
const FETCH_URL  = `${API_BASE}/fetch-url`;  // bonus endpoint

/* ─────────────────────────────────────────────
   DOM References
───────────────────────────────────────────── */
const newsInput      = document.getElementById('newsInput');
const urlInput       = document.getElementById('urlInput');
const fetchUrlBtn    = document.getElementById('fetchUrlBtn');
const checkBtn       = document.getElementById('checkBtn');
const clearBtn       = document.getElementById('clearBtn');
const historyBtn     = document.getElementById('historyBtn');
const closeHistory   = document.getElementById('closeHistory');
const clearHistoryBtn= document.getElementById('clearHistoryBtn');

const charCount      = document.getElementById('charCount');
const errorBanner    = document.getElementById('errorBanner');
const errorText      = document.getElementById('errorText');
const loadingState   = document.getElementById('loadingState');
const resultSection  = document.getElementById('resultSection');

// Result elements
const classificationBadge = document.getElementById('classificationBadge');
const classIcon      = document.getElementById('classIcon');
const classLabel     = document.getElementById('classLabel');
const resultTimestamp= document.getElementById('resultTimestamp');
const scoreValue     = document.getElementById('scoreValue');
const scoreBar       = document.getElementById('scoreBar');
const explanationText= document.getElementById('explanationText');
const keywordTags    = document.getElementById('keywordTags');

// History
const historyPanel   = document.getElementById('historyPanel');
const historyList    = document.getElementById('historyList');
const historyCount   = document.getElementById('historyCount');

// Loading step indicators
const step1 = document.getElementById('step1');
const step2 = document.getElementById('step2');
const step3 = document.getElementById('step3');

/* ─────────────────────────────────────────────
   In-Memory History Store
───────────────────────────────────────────── */
let detectionHistory = [];

/* ─────────────────────────────────────────────
   Utility Helpers
───────────────────────────────────────────── */

/** Show an error message in the banner */
function showError(msg) {
  errorText.textContent = msg;
  errorBanner.classList.remove('hidden');
}

/** Hide error banner */
function hideError() {
  errorBanner.classList.add('hidden');
}

/** Toggle loading state visibility */
function setLoading(visible) {
  if (visible) {
    loadingState.classList.remove('hidden');
    resultSection.classList.add('hidden');
    checkBtn.disabled = true;
    checkBtn.innerHTML = `
      <svg class="btn-icon" viewBox="0 0 20 20" fill="currentColor" width="18">
        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd"/>
      </svg> Analysing…`;
  } else {
    loadingState.classList.add('hidden');
    checkBtn.disabled = false;
    checkBtn.innerHTML = `
      <svg class="btn-icon" viewBox="0 0 20 20" fill="currentColor" width="18">
        <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414L8.414 15 3.293 9.879a1 1 0 011.414-1.414L8.414 12.172l6.879-6.879a1 1 0 011.414 0z" clip-rule="evenodd"/>
      </svg> Check News`;
  }
}

/** Reset all loading step indicators */
function resetSteps() {
  [step1, step2, step3].forEach(s => {
    s.classList.remove('active', 'done');
  });
  step1.classList.add('active');
}

/** Advance the animated loading steps */
function advanceStep(stepNum) {
  if (stepNum === 2) {
    step1.classList.remove('active');
    step1.classList.add('done');
    step2.classList.add('active');
  } else if (stepNum === 3) {
    step2.classList.remove('active');
    step2.classList.add('done');
    step3.classList.add('active');
  }
}

/** Format a Date to a readable timestamp */
function formatTimestamp(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** Classify a score into real / misleading / fake */
function scoreToClass(score) {
  if (score >= 50) return 'real';
  if (score >= 25) return 'misleading';
  return 'fake';
}

/** Pick an emoji icon for each classification */
function classToIcon(cls) {
  const icons = { real: '✅', fake: '❌', misleading: '⚠️' };
  return icons[cls] || '❓';
}

/** Pick a display label */
function classToLabel(cls) {
  const labels = { real: 'Real News', fake: 'Fake News', misleading: 'Misleading News' };
  return labels[cls] || cls;
}

/* ─────────────────────────────────────────────
   Character Counter
───────────────────────────────────────────── */
newsInput.addEventListener('input', () => {
  const len = newsInput.value.length;
  charCount.textContent = `${len.toLocaleString()} character${len !== 1 ? 's' : ''}`;
});

/* ─────────────────────────────────────────────
   Clear Button
───────────────────────────────────────────── */
clearBtn.addEventListener('click', () => {
  newsInput.value = '';
  urlInput.value  = '';
  charCount.textContent = '0 characters';
  hideError();
  resultSection.classList.add('hidden');
  newsInput.focus();
});

/* ─────────────────────────────────────────────
   URL Fetch (Bonus Feature)
   Sends URL to backend /fetch-url which returns
   the scraped article text.
───────────────────────────────────────────── */
fetchUrlBtn.addEventListener('click', async () => {
  const url = urlInput.value.trim();
  if (!url) { showError('Please enter a URL first.'); return; }

  // Validate URL format
  try { new URL(url); } catch {
    showError('Invalid URL format. Please include https://'); return;
  }

  hideError();
  fetchUrlBtn.textContent = 'Fetching…';
  fetchUrlBtn.disabled = true;

  try {
    const res = await fetch(FETCH_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();

    if (data.text) {
      newsInput.value = data.text;
      // Trigger character count update
      newsInput.dispatchEvent(new Event('input'));
    } else {
      showError(data.error || 'Could not extract text from the URL.');
    }
  } catch (err) {
    showError('Could not reach backend. Is the Flask server running on port 5000?');
  } finally {
    fetchUrlBtn.textContent = 'Fetch';
    fetchUrlBtn.disabled = false;
  }
});

/* ─────────────────────────────────────────────
   Render Result
───────────────────────────────────────────── */
function renderResult(data, inputText) {
  const cls = scoreToClass(data.score);

  // ── Classification badge
  classificationBadge.className = `classification-badge ${cls}`;
  classIcon.textContent  = classToIcon(cls);
  classLabel.textContent = classToLabel(cls);

  // ── Timestamp
  resultTimestamp.textContent = formatTimestamp(new Date());

  // ── Score
  scoreValue.textContent = `${data.score}`;
  scoreValue.style.color = cls === 'real'
    ? 'var(--real-color)'
    : cls === 'fake'
    ? 'var(--fake-color)'
    : 'var(--mislead-color)';

  // Animate the score bar (slight delay so the element is visible first)
  scoreBar.style.width = '0%';
  scoreBar.className = `score-bar ${cls}`;
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      scoreBar.style.width = `${data.score}%`;
    });
  });

  // ── Explanation
  explanationText.textContent = data.explanation || 'No explanation provided.';

  // ── Keywords (bonus)
  keywordTags.innerHTML = '';
  if (data.keywords && data.keywords.length > 0) {
    data.keywords.forEach((kw, i) => {
      const tag = document.createElement('span');
      tag.className = `keyword-tag${i < 3 ? ' high-weight' : ''}`;
      tag.textContent = kw;
      keywordTags.appendChild(tag);
    });
  } else {
    keywordTags.innerHTML = '<span class="keyword-tag">No keywords extracted</span>';
  }

  // ── Show result section
  resultSection.classList.remove('hidden');

  // ── Scroll to result smoothly
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // ── Add to history
  addToHistory({ classification: cls, score: data.score, text: inputText, timestamp: new Date() });
}

/* ─────────────────────────────────────────────
   Main Detect Function
───────────────────────────────────────────── */
checkBtn.addEventListener('click', async () => {
  const text = newsInput.value.trim();

  // Validation
  if (!text) {
    showError('Please enter some news text before checking.');
    newsInput.focus();
    return;
  }
  if (text.length < 20) {
    showError('Text too short. Please enter at least 20 characters for accurate results.');
    newsInput.focus();
    return;
  }

  hideError();
  setLoading(true);
  resetSteps();

  // Animate steps while waiting
  const step2Timer = setTimeout(() => advanceStep(2), 600);
  const step3Timer = setTimeout(() => advanceStep(3), 1300);

  try {
    const response = await fetch(DETECT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    clearTimeout(step2Timer);
    clearTimeout(step3Timer);

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.error || `Server returned ${response.status}`);
    }

    const data = await response.json();
    setLoading(false);
    renderResult(data, text);

  } catch (err) {
    clearTimeout(step2Timer);
    clearTimeout(step3Timer);
    setLoading(false);

    if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
      showError('Cannot connect to the backend. Make sure the Flask server is running on http://127.0.0.1:5000');
    } else {
      showError(`Error: ${err.message}`);
    }
  }
});

/* ─────────────────────────────────────────────
   History Management
───────────────────────────────────────────── */

/** Add a detection result to in-memory history */
function addToHistory(entry) {
  detectionHistory.unshift(entry); // newest first
  updateHistoryUI();
}

/** Rebuild the history panel DOM */
function updateHistoryUI() {
  const count = detectionHistory.length;
  historyCount.textContent = count;

  // Show / hide badge
  historyCount.style.display = count > 0 ? 'inline-flex' : 'none';

  if (count === 0) {
    historyList.innerHTML = '<p class="history-empty">No checks yet. Analyse an article to get started.</p>';
    return;
  }

  historyList.innerHTML = '';
  detectionHistory.forEach((entry, idx) => {
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `
      <div class="history-item-header">
        <span class="history-class ${entry.classification}">${classToLabel(entry.classification)}</span>
        <span class="history-score">Score: ${entry.score}/100</span>
      </div>
      <div class="history-snippet">${entry.text.substring(0, 80)}${entry.text.length > 80 ? '…' : ''}</div>
      <div class="history-time">${formatTimestamp(entry.timestamp)}</div>
    `;
    // Clicking a history item re-populates the textarea
    item.addEventListener('click', () => {
      newsInput.value = entry.text;
      newsInput.dispatchEvent(new Event('input'));
      historyPanel.classList.remove('open');
      newsInput.scrollIntoView({ behavior: 'smooth' });
    });
    historyList.appendChild(item);
  });
}

/** Open / close history panel */
historyBtn.addEventListener('click', () => {
  historyPanel.classList.toggle('open');
});
closeHistory.addEventListener('click', () => {
  historyPanel.classList.remove('open');
});

/** Clear all history */
clearHistoryBtn.addEventListener('click', () => {
  detectionHistory = [];
  updateHistoryUI();
});

/* Close history panel when clicking outside */
document.addEventListener('click', (e) => {
  if (!historyPanel.contains(e.target) && e.target !== historyBtn && !historyBtn.contains(e.target)) {
    historyPanel.classList.remove('open');
  }
});

/* ─────────────────────────────────────────────
   Keyboard Shortcut: Ctrl+Enter to submit
───────────────────────────────────────────── */
newsInput.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    checkBtn.click();
  }
});

/* ─────────────────────────────────────────────
   Initialize
───────────────────────────────────────────── */
updateHistoryUI();
newsInput.focus();

console.log(
  '%cTruthLens 🔍 loaded',
  'color:#e74c3c; font-weight:700; font-size:1.1em;',
  '\nBackend expected at:', API_BASE
);
