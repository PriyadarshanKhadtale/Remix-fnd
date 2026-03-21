import React, { useState } from 'react'

function AIDetectionPanel({ result, standalone = false }) {
  const [expanded, setExpanded] = useState(standalone)

  if (!result) return null

  const isAI = result.is_ai_generated === true
  const isUncertain = result.is_ai_generated === null || 
    (result.probability && result.probability >= 40 && result.probability <= 60)

  const getStatusClass = () => {
    if (isAI) return 'ai-detected'
    if (isUncertain) return 'ai-uncertain'
    return 'ai-human'
  }

  const getStatusIcon = () => {
    if (isAI) return '🤖'
    if (isUncertain) return '🤔'
    return '👤'
  }

  // Get confidence value - backend returns it as 0-100
  const confidence = result.confidence || result.probability || 50

  // Backend returns 'detectors' array, map it to display format
  const detectors = result.detectors || result.signals || []

  return (
    <div className={`ai-panel ${getStatusClass()} ${standalone ? 'standalone' : ''}`}>
      {!standalone && (
        <button 
          className="ai-header"
          onClick={() => setExpanded(!expanded)}
        >
          <span className="ai-title">
            <span className="ai-icon">{getStatusIcon()}</span>
            AI Content Analysis
          </span>
          <span className={`expand-icon ${expanded ? 'expanded' : ''}`}>▼</span>
        </button>
      )}
      
      {(expanded || standalone) && (
        <div className="ai-content">
          {/* Main Verdict */}
          <div className="ai-verdict-section">
            <div className={`ai-verdict-badge ${getStatusClass()}`}>
              <span className="verdict-icon">{getStatusIcon()}</span>
              <div className="verdict-text-group">
                <span className="verdict-main">{result.verdict}</span>
                <span className="verdict-confidence">{confidence.toFixed(0)}% confidence</span>
              </div>
            </div>
          </div>

          {/* Confidence Bar - shows AI probability */}
          <div className="ai-confidence-bar">
            <div className="bar-labels">
              <span>👤 Human</span>
              <span>🤖 AI</span>
            </div>
            <div className="bar-track">
              <div 
                className="bar-fill"
                style={{ width: `${result.probability || confidence}%` }}
              ></div>
              <div 
                className="bar-marker"
                style={{ left: `${result.probability || confidence}%` }}
              ></div>
            </div>
            <div className="bar-value">
              {(result.probability || confidence).toFixed(0)}% AI likelihood
            </div>
          </div>

          {/* Detectors - Backend returns 'detectors' array */}
          {detectors.length > 0 && (
            <div className="ai-signals">
              <h4>🔬 Detection Methods ({detectors.length} detectors)</h4>
              <div className="signals-list">
                {detectors.map((detector, index) => (
                  <div key={index} className="signal-item">
                    <div className="signal-header">
                      <span className="signal-name">
                        {detector.score > 50 ? '🤖' : '👤'} {detector.name}
                      </span>
                      <span className="signal-score">
                        {typeof detector.score === 'number' ? detector.score.toFixed(0) : detector.score}%
                      </span>
                    </div>
                    <p className="signal-desc">
                      {detector.details || detector.description || 'Analysis complete'}
                    </p>
                    <div className="signal-bar">
                      <div 
                        className="signal-fill"
                        style={{ 
                          width: `${detector.score}%`,
                          backgroundColor: detector.score > 60 ? '#ef4444' : 
                                          detector.score > 40 ? '#eab308' : '#22c55e'
                        }}
                      ></div>
                    </div>
                    {detector.weight && (
                      <span className="signal-weight">Weight: {(detector.weight * 100).toFixed(0)}%</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Summary */}
          {result.summary && (
            <div className="ai-summary">
              <h4>📋 Summary</h4>
              <p>{result.summary}</p>
            </div>
          )}

          {/* Explanation */}
          {result.explanation && (
            <div className="ai-explanation">
              <h4>📝 Detailed Analysis</h4>
              <pre className="explanation-text">{result.explanation}</pre>
            </div>
          )}

          {/* Recommendation */}
          {result.recommendation && (
            <div className="ai-recommendation">
              <p>{result.recommendation}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default AIDetectionPanel
