/**
 * API utilities for REMIX-FND backend
 * Updated to match backend v3.0 API
 */

/**
 * In `npm run dev`, use same-origin `/api` so Vite proxies to the backend (avoids CORS / localhost issues).
 * Production / preview: full URL. Override anytime: VITE_API_BASE=https://api.example.com npm run build
 */
function getApiBase() {
  const env = import.meta.env.VITE_API_BASE
  if (env) return env.replace(/\/$/, '')
  if (import.meta.env.DEV) return '/api'
  return 'http://127.0.0.1:8000'
}

const API_BASE = getApiBase()

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

  const response = await fetch(`${API_BASE}/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  })

  if (!response.ok) {
    throw new Error(`Detection failed: ${response.statusText}`)
  }

  return response.json()
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
    throw new Error(`AI detection failed: ${response.statusText}`)
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
    throw new Error(`Evidence retrieval failed: ${response.statusText}`)
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
    throw new Error(`Explanation failed: ${response.statusText}`)
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
    throw new Error(`Image analysis failed: ${response.statusText}`)
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
