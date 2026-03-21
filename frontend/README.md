# REMIX-FND Frontend v3.0

Modern React frontend for the REMIX-FND fake news detection system.

## 🆕 What's New in v3.0

### New Features
- **Evidence Check** - Fact-check against 12.8K claims from LIAR dataset
- **Explanation Levels** - Choose Simple, Standard, or Expert explanations
- **Image Upload** - Analyze images for manipulation (optional)
- **Processing Stats** - View processing time, stages, and early exit status
- **Feature Scores** - See individual scores for each analysis method
- **5-Detector AI Analysis** - Detailed breakdown of AI detection methods

### Fixed Issues
- ✅ AI Detection panel now correctly maps `detectors` from backend
- ✅ Evidence panel handles all backend verdict types
- ✅ API calls include all available options

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **CSS** - Custom styling (no frameworks)

## Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── NewsInput.jsx      # Text input + options
│   │   ├── ResultCard.jsx     # Detection result + stats
│   │   ├── ExplanationPanel.jsx   # 3-tier explanations
│   │   ├── AIDetectionPanel.jsx   # 5-detector AI analysis
│   │   ├── EvidencePanel.jsx      # Fact-checking results
│   │   └── Header.jsx
│   ├── styles/
│   │   └── main.css           # All styles
│   ├── utils/
│   │   └── api.js             # Backend API calls
│   ├── App.jsx                # Main app
│   └── index.jsx              # Entry point
├── package.json
└── vite.config.js
```

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## API Integration

### Detection Request Options

```javascript
detectFakeNews(text, {
  includeExplanation: true,        // Include explanation
  explanationLevel: 'intermediate', // novice | intermediate | expert
  checkAI: false,                   // Check for AI-generated
  checkEvidence: false,             // Fact-check against KB
  imageBase64: null,                // Base64 image (optional)
  enableEarlyExit: true            // Early exit on high confidence
})
```

### Response Features

| Field | Description |
|-------|-------------|
| `prediction` | REAL or FAKE |
| `confidence` | 0-100% |
| `processing_time_ms` | Time taken |
| `stages_run` | Number of stages executed |
| `early_exit` | Whether early exit was triggered |
| `feature_scores` | Scores for each analysis method |
| `ai_analysis` | 5-detector ensemble results |
| `evidence` | Fact-checking results |
| `explanation` | 3-tier explanation |

## UI Features

### Detection Tab
- Text input with character count
- **🤖 Check AI** checkbox - detect AI-generated content
- **📚 Fact-check** checkbox - verify against knowledge base
- **💡 Level selector** - Simple/Standard/Expert explanations
- **🖼️ Image upload** - analyze attached images

### Result Display
- Verdict badge (REAL/FAKE) with confidence
- Processing stats (time, stages, early exit)
- Feature scores with progress bars
- Probability breakdown
- Processing pipeline visualization

### Explanation Panel
- Collapsible with "Why this verdict?"
- Features used with individual scores
- Summary and key factors
- Highlighted suspicious/credible words
- Actionable suggestions

### AI Detection Panel
- Human ↔ AI probability bar
- 5 detector results with details
- Summary and explanation

### Evidence Panel
- Verdict (Likely TRUE/FALSE/Mixed/Insufficient)
- Search method and depth info
- Evidence items with stance indicators
- Recommendations

## Design

- **Theme**: Dark futuristic with cyan/purple accents
- **Font**: Plus Jakarta Sans (headers) + JetBrains Mono (code)
- **Features**: Animated backgrounds, glassmorphism, smooth transitions

## Connecting to Backend

Default backend: `http://localhost:8000`

Start the backend first:
```bash
cd ../backend
python run.py
```

Then start the frontend:
```bash
npm run dev
```

Visit http://localhost:3000
