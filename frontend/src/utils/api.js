/**
 * API utilities for REMIX-FND backend
 * Updated to match backend v3.0 API
 */

/**
 * In `npm run dev`, use same-origin `/api` so Vite proxies to the backend (avoids CORS / localhost issues).
 * Production / preview: full URL. Override anytime: VITE_API_BASE=https://api.example.com npm run build
 */
export function getApiBase() {
  const env = import.meta.env.VITE_API_BASE
  if (env) return env.replace(/\/$/, '')
  if (import.meta.env.DEV) return '/api'
  return 'http://127.0.0.1:8000'
}

const API_BASE = getApiBase()

/** Short label for footer (hide long URLs) */
export function getApiBaseLabel() {
  const b = getApiBase()
  if (b === '/api') return 'local (Vite → :8000)'
  try {
    const u = new URL(b.startsWith('http') ? b : `https://${b}`)
    return u.host || b
  } catch {
    return b
  }
}

/**
 * Infer backend capabilities from GET /health (run.py vs run_lite).
 */
export function parseHealthCapabilities(health) {
  if (!health || typeof health !== 'object') {
    return { isLite: false, isFull: false, pipelineLoading: false, raw: health }
  }
  if (health.mode === 'lite') {
    return { isLite: true, isFull: false, pipelineLoading: false, raw: health }
  }
  const ready = health.ready === true
  const loading = health.loading === true || health.pipeline === 'loading'
  return {
    isLite: false,
    isFull: ready,
    pipelineLoading: loading,
    raw: health,
  }
}

/**
 * Map run_lite /detect JSON into shapes the UI already understands.
 */
export function normalizeDetectResponse(data) {
  if (!data || typeof data !== 'object') return data
  const isLite =
    data.mode === 'rule-based-lite' ||
    (typeof data.mode === 'string' && data.mode.includes('lite'))

  if (!isLite) return data

  const ind = data.indicators_found || {}
  const ruleScore = Math.min(
    100,
    (Number(ind.suspicious_words) || 0) * 12 +
      (Number(ind.clickbait_patterns) || 0) * 18 +
      (ind.excessive_caps ? 15 : 0) +
      (ind.excessive_exclamation ? 10 : 0)
  )

  let explanation = data.explanation
  if (explanation && typeof explanation === 'object') {
    explanation = {
      ...explanation,
      verdict_explanation:
        explanation.verdict_explanation || explanation.confidence_reason || null,
    }
  }

  return {
    ...data,
    feature_scores: data.feature_scores || { rule_based_heuristics: ruleScore },
    explanation,
  }
}

async function readErrorDetail(response) {
  try {
    const t = await response.text()
    if (!t) return response.statusText || `HTTP ${response.status}`
    try {
      const j = JSON.parse(t)
      if (j?.detail) return typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)
      if (typeof j?.message === 'string') return j.message
    } catch {
      if (t.length < 500) return t.trim()
    }
  } catch {
    /* ignore */
  }
  return response.statusText || `HTTP ${response.status}`
}

function backendHintForStatus(status) {
  if (status === 502 || status === 503 || status === 504) {
    return ' Is the API running on port 8000? From repo root: cd backend && python3 run.py'
  }
  return ''
}

/**
 * Detect fake news with all available features
 * @param {string} text - News text to analyze
 * @param {object} options - Detection options
 * @param {boolean} options.includeExplanation - Include explanation in response
 * @param {string} options.explanationLevel - 'novice' | 'intermediate' | 'expert'
 * @param {boolean} options.checkAI - Check for AI-generated content
 * @param {boolean} options.checkEvidence - Check evidence/fact-check
 * @param {string} options.imageBase64 - Base64 encoded image (optional)
 * @param {boolean} options.enableEarlyExit - Enable early exit optimization
 */
export async function detectFakeNews(text, options = {}) {
  const {
    includeExplanation = true,
    explanationLevel = 'intermediate',
    checkAI = false,
    checkEvidence = false,
    imageBase64 = null,
    enableEarlyExit = true
  } = options

  const body = {
    text,
    include_explanation: includeExplanation,
    explanation_level: explanationLevel,
    check_ai_generated: checkAI,
    check_evidence: checkEvidence,
    enable_early_exit: enableEarlyExit
  }

  // Only include image if provided
  if (imageBase64) {
    body.image_base64 = imageBase64
  }

  let response
  try {
    response = await fetch(`${API_BASE}/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
  } catch (e) {
    const base = getApiBase()
    const proxyHint =
      base === '/api'
        ? ' Vite proxies /api → http://127.0.0.1:8000 — start the backend first.'
        : ''
    throw new Error((e?.message || 'Network error') + proxyHint)
  }

  if (!response.ok) {
    const detail = await readErrorDetail(response)
    throw new Error(
      `Detection failed (${response.status}): ${detail}.${backendHintForStatus(response.status)}`
    )
  }

  const raw = await response.text()
  let parsed
  try {
    parsed = JSON.parse(raw)
  } catch {
    throw new Error(
      'Invalid JSON from /detect. If you use npm run dev, ensure the backend is up on port 8000 (502 pages are HTML).'
    )
  }
  return normalizeDetectResponse(parsed)
}

/**
 * Detect AI-generated content
 */
export async function detectAI(text) {
  const response = await fetch(`${API_BASE}/ai-detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  })

  if (!response.ok) {
    throw new Error(`AI detection failed: ${await readErrorDetail(response)}`)
  }

  return response.json()
}

/**
 * Get evidence for fact-checking
 */
export async function getEvidence(text, maxResults = 10, uncertainty = 0.5) {
  const response = await fetch(`${API_BASE}/evidence`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      max_results: maxResults,
      uncertainty
    })
  })

  if (!response.ok) {
    throw new Error(`Evidence retrieval failed: ${await readErrorDetail(response)}`)
  }

  return response.json()
}

/**
 * Get detailed explanation
 */
export async function getExplanation(text, level = 'intermediate') {
  const response = await fetch(`${API_BASE}/explain`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      level
    })
  })

  if (!response.ok) {
    throw new Error(`Explanation failed: ${await readErrorDetail(response)}`)
  }

  return response.json()
}

/**
 * Analyze image for manipulation
 */
export async function analyzeImage(imageBase64, text = null) {
  const response = await fetch(`${API_BASE}/image-analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_base64: imageBase64,
      text
    })
  })

  if (!response.ok) {
    throw new Error(`Image analysis failed: ${await readErrorDetail(response)}`)
  }

  return response.json()
}

/**
 * Health check
 */
export async function healthCheck() {
  const response = await fetch(`${API_BASE}/health`)
  return response.json()
}
