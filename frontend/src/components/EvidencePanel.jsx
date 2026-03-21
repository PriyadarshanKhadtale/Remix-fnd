import React from 'react'

function EvidencePanel({ result }) {
  if (!result) return null

  // Map backend verdict to display class
  // Backend returns: likely_true, likely_false, mixed_evidence, insufficient_evidence
  const getVerdictClass = () => {
    switch (result.verdict) {
      case 'likely_true':
      case 'supported':
        return 'verdict-supported'
      case 'likely_false':
      case 'contradicted':
        return 'verdict-contradicted'
      case 'mixed_evidence':
        return 'verdict-mixed'
      case 'insufficient_evidence':
      default:
        return 'verdict-inconclusive'
    }
  }

  const getVerdictIcon = () => {
    switch (result.verdict) {
      case 'likely_true':
      case 'supported':
        return '✅'
      case 'likely_false':
      case 'contradicted':
        return '❌'
      case 'mixed_evidence':
        return '⚖️'
      case 'insufficient_evidence':
      default:
        return '❓'
    }
  }

  const getVerdictText = () => {
    switch (result.verdict) {
      case 'likely_true':
        return 'Claim Likely TRUE'
      case 'likely_false':
        return 'Claim Likely FALSE'
      case 'mixed_evidence':
        return 'Mixed Evidence'
      case 'insufficient_evidence':
        return 'Insufficient Evidence'
      case 'supported':
        return 'Claim Supported'
      case 'contradicted':
        return 'Claim Contradicted'
      default:
        return 'Inconclusive'
    }
  }

  const getVerdictDescription = () => {
    switch (result.verdict) {
      case 'likely_true':
        return 'Evidence from our knowledge base supports this claim'
      case 'likely_false':
        return 'Evidence from our knowledge base contradicts this claim'
      case 'mixed_evidence':
        return 'Some evidence supports and some contradicts this claim'
      case 'insufficient_evidence':
        return 'Not enough evidence found to verify this claim'
      default:
        return 'Unable to determine claim validity'
    }
  }

  return (
    <div className="evidence-panel">
      {/* Query */}
      <div className="evidence-query">
        <span className="query-label">📝 Claim analyzed:</span>
        <p className="query-text">"{result.query}"</p>
      </div>

      {/* Verdict */}
      <div className={`evidence-verdict ${getVerdictClass()}`}>
        <span className="verdict-icon">{getVerdictIcon()}</span>
        <div className="verdict-info">
          <span className="verdict-text">{getVerdictText()}</span>
          <span className="verdict-confidence">
            {result.confidence?.toFixed(0) || 0}% confidence
          </span>
          <span className="verdict-description">{getVerdictDescription()}</span>
        </div>
      </div>

      {/* Search Info */}
      {(result.search_method || result.retrieval_depth) && (
        <div className="evidence-search-info">
          <span className="search-badge">
            🔍 {result.search_method || 'keyword search'}
          </span>
          {result.retrieval_depth && (
            <span className="depth-badge">
              📚 Searched {result.retrieval_depth} documents
            </span>
          )}
        </div>
      )}

      {/* Keywords */}
      {result.claim_keywords && result.claim_keywords.length > 0 && (
        <div className="evidence-keywords">
          <span className="keywords-label">🔑 Key terms searched:</span>
          <div className="keywords-list">
            {result.claim_keywords.slice(0, 10).map((keyword, index) => (
              <span key={index} className="keyword-chip">{keyword}</span>
            ))}
          </div>
        </div>
      )}

      {/* Evidence Summary */}
      {result.evidence_summary && (
        <div className="evidence-summary-box">
          <span className="summary-icon">📊</span>
          <p>{result.evidence_summary}</p>
        </div>
      )}

      {/* Evidence Items */}
      {result.evidence && result.evidence.length > 0 && (
        <div className="evidence-items">
          <h4>📚 Related Evidence Found ({result.evidence.length})</h4>
          {result.evidence.map((item, index) => (
            <div key={index} className="evidence-item">
              <div className="evidence-item-header">
                <span className={`support-indicator ${
                  item.supports_claim === true ? 'supports' : 
                  item.supports_claim === false ? 'contradicts' : 'neutral'
                }`}>
                  {item.supports_claim === true ? '✓ Supports' : 
                   item.supports_claim === false ? '✗ Contradicts' : '○ Related'}
                </span>
                <span className="relevance-score">
                  {((item.relevance_score || 0) * 100).toFixed(0)}% relevant
                </span>
              </div>
              <h5 className="evidence-title">{item.title}</h5>
              <p className="evidence-snippet">{item.snippet}</p>
              <div className="evidence-meta">
                <span className="evidence-source">📰 {item.source}</span>
                {item.category && (
                  <span className="evidence-category">🏷️ {item.category}</span>
                )}
                {item.stance && (
                  <span className={`evidence-stance stance-${item.stance}`}>
                    {item.stance === 'supports' ? '👍' : 
                     item.stance === 'refutes' ? '👎' : '➖'} {item.stance}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* No Evidence */}
      {(!result.evidence || result.evidence.length === 0) && (
        <div className="no-evidence">
          <span className="no-evidence-icon">📭</span>
          <p>No relevant evidence found in our knowledge base.</p>
          <p className="no-evidence-suggestion">
            Try searching fact-checking websites like Snopes, PolitiFact, or FactCheck.org
          </p>
        </div>
      )}

      {/* Recommendation */}
      {result.recommendation && (
        <div className={`evidence-recommendation ${getVerdictClass()}`}>
          <p>{result.recommendation}</p>
        </div>
      )}
    </div>
  )
}

export default EvidencePanel

