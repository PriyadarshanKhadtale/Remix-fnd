import React from 'react'

function ResultCard({ result, newsText }) {
  const isFake = result.prediction === 'FAKE'
  
  // Format processing time
  const formatTime = (ms) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`
    return `${(ms / 1000).toFixed(2)}s`
  }

  // Get feature score color
  const getScoreColor = (score) => {
    if (score >= 80) return '#22c55e'  // green
    if (score >= 60) return '#eab308'  // yellow
    if (score >= 40) return '#f97316'  // orange
    return '#ef4444'  // red
  }

  return (
    <div className={`result-card ${isFake ? 'result-fake' : 'result-real'}`}>
      {/* Verdict Badge */}
      <div className="verdict-section">
        <div className={`verdict-badge ${isFake ? 'badge-fake' : 'badge-real'}`}>
          <span className="verdict-icon">{isFake ? '🚨' : '✅'}</span>
          <span className="verdict-text">{result.prediction}</span>
          <span className="verdict-subtext">NEWS</span>
        </div>
        <div className="confidence">
          <span className="confidence-value">{result.confidence.toFixed(1)}%</span>
          <span className="confidence-label">confidence</span>
        </div>
      </div>

      {/* Processing Stats Badge */}
      {(result.processing_time_ms || result.early_exit || result.stages_run) && (
        <div className="processing-stats">
          {result.processing_time_ms && (
            <span className="stat-badge time-badge" title="Processing time">
              ⚡ {formatTime(result.processing_time_ms)}
            </span>
          )}
          {result.stages_run && (
            <span className="stat-badge stages-badge" title="Analysis stages completed">
              📊 {result.stages_run} stage{result.stages_run > 1 ? 's' : ''}
            </span>
          )}
          {result.early_exit && (
            <span className="stat-badge early-exit-badge" title="High confidence - early exit">
              🎯 Early Exit
            </span>
          )}
        </div>
      )}

      {/* News Text Preview */}
      <div className="news-preview">
        <p className="preview-label">Analyzed Text:</p>
        <p className="preview-text">"{newsText.length > 200 ? newsText.substring(0, 200) + '...' : newsText}"</p>
      </div>

      {/* Probability Bars */}
      <div className="probability-section">
        <div className="prob-bar-container">
          <div className="prob-label">
            <span>✅ Real</span>
            <span>{result.real_probability.toFixed(1)}%</span>
          </div>
          <div className="prob-bar prob-bar-real">
            <div 
              className="prob-fill" 
              style={{ width: `${result.real_probability}%` }}
            ></div>
          </div>
        </div>
        
        <div className="prob-bar-container">
          <div className="prob-label">
            <span>🚨 Fake</span>
            <span>{result.fake_probability.toFixed(1)}%</span>
          </div>
          <div className="prob-bar prob-bar-fake">
            <div 
              className="prob-fill" 
              style={{ width: `${result.fake_probability}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Feature Scores */}
      {result.feature_scores && Object.keys(result.feature_scores).length > 0 && (
        <div className="feature-scores-section">
          <h4 className="section-title">🧠 Analysis Methods Used</h4>
          <div className="feature-scores-grid">
            {Object.entries(result.feature_scores).map(([feature, score]) => (
              <div key={feature} className="feature-score-item">
                <div className="feature-score-header">
                  <span className="feature-score-name">
                    {feature === 'text_analysis' && '📝'}
                    {feature === 'ai_detection' && '🤖'}
                    {feature === 'evidence_retrieval' && '📚'}
                    {feature === 'image_analysis' && '🖼️'}
                    {' '}
                    {feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </span>
                  <span 
                    className="feature-score-value"
                    style={{ color: getScoreColor(score) }}
                  >
                    {typeof score === 'number' ? score.toFixed(0) : score}%
                  </span>
                </div>
                <div className="feature-score-bar">
                  <div 
                    className="feature-score-fill"
                    style={{ 
                      width: `${Math.min(score, 100)}%`,
                      backgroundColor: getScoreColor(score)
                    }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Processing Details */}
      {result.processing_details && result.processing_details.length > 0 && (
        <div className="processing-details">
          <h4 className="section-title">⚙️ Processing Pipeline</h4>
          <div className="pipeline-stages">
            {result.processing_details.map((stage, index) => (
              <div key={index} className="pipeline-stage">
                <span className="stage-number">{index + 1}</span>
                <div className="stage-info">
                  <span className="stage-name">
                    {stage.stage.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </span>
                  <span className="stage-time">{formatTime(stage.time_ms)}</span>
                </div>
                {index < result.processing_details.length - 1 && (
                  <span className="stage-arrow">→</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Image Analysis Results */}
      {result.image_analysis && (
        <div className="image-analysis-section">
          <h4 className="section-title">🖼️ Image Analysis</h4>
          <div className="image-analysis-grid">
            <div className="image-stat">
              <span className="stat-label">Manipulation Score</span>
              <span 
                className="stat-value"
                style={{ color: result.image_analysis.manipulation_score > 50 ? '#ef4444' : '#22c55e' }}
              >
                {result.image_analysis.manipulation_score?.toFixed(0) || 0}%
              </span>
            </div>
            <div className="image-stat">
              <span className="stat-label">Quality Score</span>
              <span className="stat-value">
                {result.image_analysis.quality_score?.toFixed(0) || 0}%
              </span>
            </div>
            <div className="image-stat">
              <span className="stat-label">Consistency</span>
              <span className="stat-value">
                {result.image_analysis.consistency_score?.toFixed(0) || 0}%
              </span>
            </div>
          </div>
          {result.image_analysis.suspicious_indicators && 
           result.image_analysis.suspicious_indicators.length > 0 && (
            <div className="suspicious-indicators">
              <span className="indicators-label">⚠️ Suspicious indicators:</span>
              <ul>
                {result.image_analysis.suspicious_indicators.map((indicator, i) => (
                  <li key={i}>{indicator}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default ResultCard
