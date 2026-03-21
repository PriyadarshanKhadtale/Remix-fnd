import React, { useState } from 'react'
import NewsInput from './components/NewsInput'
import ResultCard from './components/ResultCard'
import ExplanationPanel from './components/ExplanationPanel'
import AIDetectionPanel from './components/AIDetectionPanel'
import EvidencePanel from './components/EvidencePanel'
import Header from './components/Header'
import { detectFakeNews, detectAI, getEvidence } from './utils/api'
import './styles/main.css'

function App() {
  const [result, setResult] = useState(null)
  const [aiResult, setAiResult] = useState(null)
  const [evidenceResult, setEvidenceResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [newsText, setNewsText] = useState('')
  const [activeTab, setActiveTab] = useState('detect') // detect, ai, evidence

  const handleAnalyze = async (text, options = {}) => {
    setLoading(true)
    setError(null)
    setNewsText(text)
    setResult(null)
    setAiResult(null)
    setEvidenceResult(null)
    
    try {
      if (activeTab === 'detect') {
        // Full detection with all options
        const response = await detectFakeNews(text, {
          includeExplanation: true,
          explanationLevel: options.explanationLevel || 'intermediate',
          checkAI: options.checkAI || false,
          checkEvidence: options.checkEvidence || false,
          imageBase64: options.imageBase64 || null,
          enableEarlyExit: true
        })
        
        setResult(response)
        
        // If AI analysis is included in response, show it
        if (response.ai_analysis) {
          setAiResult(response.ai_analysis)
        }
        
        // If evidence is included in response, show it
        if (response.evidence) {
          setEvidenceResult(response.evidence)
        }
        
      } else if (activeTab === 'ai') {
        // Standalone AI detection
        const response = await detectAI(text)
        setAiResult(response)
        
      } else if (activeTab === 'evidence') {
        // Standalone evidence retrieval
        const response = await getEvidence(text)
        setEvidenceResult(response)
      }
    } catch (err) {
      const msg = err?.message || 'Request failed.'
      const net =
        /failed to fetch|networkerror|load failed/i.test(msg)
          ? ' Start the backend: cd backend && python3 run.py (listens on port 8000), then restart npm run dev.'
          : ''
      setError(msg + net)
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setResult(null)
    setAiResult(null)
    setEvidenceResult(null)
    setError(null)
    setNewsText('')
  }

  return (
    <div className="app">
      {/* Background Effects */}
      <div className="background-grid"></div>
      <div className="background-glow"></div>
      
      {/* Header */}
      <Header />

      {/* Tab Navigation */}
      <nav className="tab-nav">
        <button 
          className={`tab-btn ${activeTab === 'detect' ? 'active' : ''}`}
          onClick={() => { setActiveTab('detect'); handleClear(); }}
        >
          <span className="tab-icon">🔍</span>
          Fake News Detection
        </button>
        <button 
          className={`tab-btn ${activeTab === 'ai' ? 'active' : ''}`}
          onClick={() => { setActiveTab('ai'); handleClear(); }}
        >
          <span className="tab-icon">🤖</span>
          AI Content Check
        </button>
        <button 
          className={`tab-btn ${activeTab === 'evidence' ? 'active' : ''}`}
          onClick={() => { setActiveTab('evidence'); handleClear(); }}
        >
          <span className="tab-icon">📚</span>
          Fact Check
        </button>
      </nav>

      {/* Main Content */}
      <main className="main">
        <NewsInput 
          onAnalyze={handleAnalyze} 
          onClear={handleClear}
          loading={loading}
          mode={activeTab}
        />
        
        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}
        
        {/* Fake News Detection Results */}
        {result && activeTab === 'detect' && (
          <div className="results-section">
            <ResultCard result={result} newsText={newsText} />
            
            {/* Explanation Panel */}
            {result.explanation && (
              <ExplanationPanel explanation={result.explanation} />
            )}
            
            {/* Evidence Panel (if checked) */}
            {result.evidence && (
              <div className="evidence-in-detect">
                <EvidencePanel result={result.evidence} />
              </div>
            )}
            
            {/* AI Analysis Panel */}
            {result.ai_analysis && (
              <AIDetectionPanel result={result.ai_analysis} />
            )}
          </div>
        )}

        {/* Standalone AI Detection Results */}
        {aiResult && activeTab === 'ai' && (
          <div className="results-section">
            <AIDetectionPanel result={aiResult} standalone />
          </div>
        )}

        {/* Standalone Evidence Results */}
        {evidenceResult && activeTab === 'evidence' && (
          <div className="results-section">
            <EvidencePanel result={evidenceResult} />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>Built with 🧠 AI · Always verify news from multiple sources</p>
        <p className="footer-sub">REMIX-FND v3.0 · Backend running on localhost:8000</p>
      </footer>
    </div>
  )
}

export default App
