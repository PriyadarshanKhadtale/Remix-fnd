import React from 'react'

function Header() {
  return (
    <header className="header">
      <div className="logo">
        <span className="logo-icon">🔍</span>
        <span className="logo-text">REMIX-FND</span>
        <span className="version-badge">v2</span>
      </div>
      <p className="tagline">AI-Powered Fake News Detection System</p>
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

