import React, { useState, useRef } from 'react'

function NewsInput({ onAnalyze, onClear, loading, mode }) {
  const [text, setText] = useState('')
  const [checkAI, setCheckAI] = useState(false)
  const [checkEvidence, setCheckEvidence] = useState(false)
  const [explanationLevel, setExplanationLevel] = useState('intermediate')
  const [imageBase64, setImageBase64] = useState(null)
  const [imageName, setImageName] = useState('')
  const fileInputRef = useRef(null)

  const handleSubmit = (e) => {
    e.preventDefault()
    if (text.trim()) {
      onAnalyze(text.trim(), { 
        checkAI, 
        checkEvidence,
        explanationLevel,
        imageBase64
      })
    }
  }

  const handleClear = () => {
    setText('')
    setImageBase64(null)
    setImageName('')
    onClear()
  }

  const handleImageUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      if (file.size > 5 * 1024 * 1024) {
        alert('Image too large. Please select an image under 5MB.')
        return
      }
      
      const reader = new FileReader()
      reader.onload = (event) => {
        // Remove the data:image/...;base64, prefix
        const base64 = event.target.result.split(',')[1]
        setImageBase64(base64)
        setImageName(file.name)
      }
      reader.readAsDataURL(file)
    }
  }

  const removeImage = () => {
    setImageBase64(null)
    setImageName('')
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const placeholders = {
    detect: "Paste a news headline or article to check if it's fake...",
    ai: "Paste text to check if it was written by AI (ChatGPT, etc.)...",
    evidence: "Enter a claim to fact-check (e.g., 'COVID vaccines are dangerous')..."
  }

  const buttonLabels = {
    detect: { icon: '🔬', text: 'Analyze News' },
    ai: { icon: '🤖', text: 'Check for AI' },
    evidence: { icon: '📚', text: 'Find Evidence' }
  }

  const examples = {
    detect: [
      "Scientists discover breakthrough treatment for cancer at Harvard Medical Center",
      "SHOCKING: Secret documents reveal government weather control conspiracy!",
      "Federal Reserve announces 0.25% interest rate increase amid inflation concerns",
      "You won't BELIEVE what this celebrity did! Doctors HATE this trick!"
    ],
    ai: [
      "It is important to note that this comprehensive analysis demonstrates the crucial factors involved. Additionally, furthermore, moreover, this highlights the essential aspects.",
      "I went to the store yesterday and bought some groceries. The weather was nice so I walked instead of driving.",
      "In today's rapidly evolving landscape, it is essential to leverage innovative solutions that facilitate robust outcomes and enhance stakeholder engagement.",
    ],
    evidence: [
      "COVID-19 vaccines cause more harm than good",
      "5G technology spreads coronavirus",
      "Climate change is a hoax invented by scientists",
      "The moon landing was faked by NASA"
    ]
  }

  return (
    <div className="news-input-container">
      <form onSubmit={handleSubmit} className="news-form">
        <div className="input-wrapper">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={placeholders[mode]}
            className="news-textarea"
            rows={5}
          />
          
          {/* Options Bar */}
          <div className="input-footer">
            <span className="char-count">{text.length} characters</span>
            
            {mode === 'detect' && (
              <div className="detection-options">
                {/* AI Check */}
                <label className="checkbox-label" title="Check if the text was written by AI">
                  <input 
                    type="checkbox" 
                    checked={checkAI}
                    onChange={(e) => setCheckAI(e.target.checked)}
                  />
                  <span className="checkbox-text">🤖 Check AI</span>
                </label>
                
                {/* Evidence Check */}
                <label className="checkbox-label" title="Fact-check against knowledge base">
                  <input 
                    type="checkbox" 
                    checked={checkEvidence}
                    onChange={(e) => setCheckEvidence(e.target.checked)}
                  />
                  <span className="checkbox-text">📚 Fact-check</span>
                </label>
                
                {/* Explanation Level */}
                <div className="select-wrapper" title="Explanation detail level">
                  <span className="select-label">💡</span>
                  <select 
                    value={explanationLevel}
                    onChange={(e) => setExplanationLevel(e.target.value)}
                    className="level-select"
                  >
                    <option value="novice">Simple</option>
                    <option value="intermediate">Standard</option>
                    <option value="expert">Expert</option>
                  </select>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Image Upload (for detect mode) */}
        {mode === 'detect' && (
          <div className="image-upload-section">
            <input
              type="file"
              ref={fileInputRef}
              accept="image/*"
              onChange={handleImageUpload}
              className="file-input"
              id="image-upload"
            />
            
            {!imageBase64 ? (
              <label htmlFor="image-upload" className="upload-label">
                <span className="upload-icon">🖼️</span>
                <span>Add image (optional)</span>
              </label>
            ) : (
              <div className="image-preview">
                <span className="preview-icon">✓</span>
                <span className="preview-name">{imageName}</span>
                <button 
                  type="button" 
                  className="remove-image"
                  onClick={removeImage}
                >
                  ✕
                </button>
              </div>
            )}
          </div>
        )}
        
        <div className="button-group">
          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={!text.trim() || loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Analyzing...
              </>
            ) : (
              <>
                <span className="btn-icon">{buttonLabels[mode].icon}</span>
                {buttonLabels[mode].text}
              </>
            )}
          </button>
          
          <button 
            type="button" 
            className="btn btn-secondary"
            onClick={handleClear}
            disabled={loading}
          >
            Clear
          </button>
        </div>
      </form>

      <div className="examples-section">
        <p className="examples-label">Try an example:</p>
        <div className="examples-grid">
          {(examples[mode] || []).map((example, index) => (
            <button
              key={index}
              className="example-btn"
              onClick={() => setText(example)}
              disabled={loading}
            >
              {example.length > 70 ? example.substring(0, 70) + '...' : example}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

export default NewsInput
