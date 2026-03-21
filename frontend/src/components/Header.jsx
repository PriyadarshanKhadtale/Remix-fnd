import React from 'react'

function Header({ backendHint = null }) {
  return (
    <header className="header">
      <div className="logo">
        <span className="logo-icon">🔍</span>
        <span className="logo-text">REMIX-FND</span>
        <span className="version-badge">v3</span>
      </div>
      <p className="tagline">AI-Powered Fake News Detection System</p>
      {backendHint && (
        <p className="tagline" style={{ fontSize: '0.85rem', opacity: 0.85, marginTop: '-0.25rem' }}>
          {backendHint}
        </p>
      )}
      <div className="feature-badges">
        <span className="feature-badge">📝 Text Analysis</span>
        <span className="feature-badge">🤖 AI Detection</span>
        <span className="feature-badge">📚 Fact Check</span>
        <span className="feature-badge">💡 Explainability</span>
      </div>
    </header>
  )
}

export default Header

