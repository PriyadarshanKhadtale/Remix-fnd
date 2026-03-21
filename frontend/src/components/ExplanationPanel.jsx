import React, { useState } from 'react'

function ExplanationPanel({ explanation }) {
  const [expanded, setExpanded] = useState(true)

  if (!explanation) return null

  return (
    <div className="explanation-panel">
      <button 
        className="explanation-header"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="explanation-title">
          <span className="explanation-icon">💡</span>
          Why this verdict?
        </span>
        <span className={`expand-icon ${expanded ? 'expanded' : ''}`}>▼</span>
      </button>
      
      {expanded && (
        <div className="explanation-content">
          {/* Features Used Section */}
          {explanation.features_used && explanation.features_used.length > 0 && (
            <div className="explanation-section features-section">
              <h4>🧠 Analysis Methods Used</h4>
              <div className="features-grid">
                {explanation.features_used.map((feature, index) => {
                  const score = Number(feature?.score ?? 0)
                  return (
                  <div 
                    key={index} 
                    className={`feature-card ${feature.status === 'primary' ? 'primary' : ''}`}
                  >
                    <div className="feature-header">
                      <span className="feature-icon">{feature.icon}</span>
                      <div className="feature-info">
                        <span className="feature-name">{feature.name}</span>
                        <span className="feature-desc">{feature.description}</span>
                      </div>
                    </div>
                    <div className="feature-score-section">
                      <div className="feature-score-bar">
                        <div 
                          className="feature-score-fill"
                          style={{ width: `${Math.min(score, 100)}%` }}
                        ></div>
                      </div>
                      <span className="feature-score-value">
                        {Number.isFinite(score) ? score.toFixed(0) : '0'}%
                      </span>
                    </div>
                    {feature.details && (
                      <span className="feature-details">{feature.details}</span>
                    )}
                  </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Summary */}
          {explanation.summary && (
            <div className="explanation-section">
              <h4>Summary</h4>
              <p className="summary-text">{explanation.summary}</p>
            </div>
          )}

          {/* Key Factors */}
          {explanation.key_factors && explanation.key_factors.length > 0 && (
            <div className="explanation-section">
              <h4>Key Factors Detected</h4>
              <ul className="factors-list">
                {explanation.key_factors.map((factor, index) => (
                  <li key={index}>{factor}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Detailed Signals */}
          {explanation.detailed_signals && (
            <div className="explanation-section">
              <h4>Detailed Signal Analysis</h4>
              
              {/* Fake Indicators */}
              {explanation.detailed_signals.fake_indicators && 
               explanation.detailed_signals.fake_indicators.length > 0 && (
                <div className="signal-group">
                  <span className="signal-group-label">⚠️ Warning Signals:</span>
                  <div className="signal-items">
                    {explanation.detailed_signals.fake_indicators.map((signal, index) => (
                      <div key={index} className="signal-item-compact fake">
                        <span className="signal-category">{signal.category}</span>
                        <span className="signal-match">"{signal.match}"</span>
                        <span className="signal-severity">{(signal.severity * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Credibility Indicators */}
              {explanation.detailed_signals.credibility_indicators && 
               explanation.detailed_signals.credibility_indicators.length > 0 && (
                <div className="signal-group">
                  <span className="signal-group-label">✅ Credibility Signals:</span>
                  <div className="signal-items">
                    {explanation.detailed_signals.credibility_indicators.map((signal, index) => (
                      <div key={index} className="signal-item-compact credible">
                        <span className="signal-category">{signal.category}</span>
                        {signal.match && <span className="signal-match">"{signal.match}"</span>}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Highlighted Words */}
          {explanation.highlighted_words && explanation.highlighted_words.length > 0 && (
            <div className="explanation-section">
              <h4>Suspicious/Important Words</h4>
              <div className="highlighted-words">
                {explanation.highlighted_words.map((item, index) => (
                  <span 
                    key={index}
                    className={`word-chip ${item.is_suspicious ? 'suspicious' : 'credible'}`}
                    title={item.reason || `Importance: ${(item.importance * 100).toFixed(0)}%`}
                  >
                    {item.word}
                    {item.reason && <span className="word-reason">{item.reason}</span>}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Confidence Breakdown */}
          {explanation.confidence_breakdown && (
            <div className="explanation-section">
              <h4>Confidence Breakdown</h4>
              <div className="confidence-grid">
                <div className="confidence-item">
                  <span className="conf-label">Real Probability</span>
                  <span className="conf-value real">{explanation.confidence_breakdown.real_probability?.toFixed(1)}%</span>
                </div>
                <div className="confidence-item">
                  <span className="conf-label">Fake Probability</span>
                  <span className="conf-value fake">{explanation.confidence_breakdown.fake_probability?.toFixed(1)}%</span>
                </div>
                {explanation.confidence_breakdown.fake_signal_score !== undefined && (
                  <div className="confidence-item">
                    <span className="conf-label">Fake Signal Score</span>
                    <span className="conf-value">{explanation.confidence_breakdown.fake_signal_score?.toFixed(2)}</span>
                  </div>
                )}
                {explanation.confidence_breakdown.credibility_score !== undefined && (
                  <div className="confidence-item">
                    <span className="conf-label">Credibility Score</span>
                    <span className="conf-value">{explanation.confidence_breakdown.credibility_score?.toFixed(2)}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Verdict Explanation */}
          {explanation.verdict_explanation && (
            <div className="explanation-section">
              <h4>Detailed Analysis</h4>
              <p>{explanation.verdict_explanation}</p>
            </div>
          )}

          {/* Suggestions */}
          {explanation.suggestions && explanation.suggestions.length > 0 && (
            <div className="explanation-section suggestions-section">
              <h4>💡 What You Should Do</h4>
              <ul className="suggestions-list">
                {explanation.suggestions.map((suggestion, index) => (
                  <li key={index}>{suggestion}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default ExplanationPanel
