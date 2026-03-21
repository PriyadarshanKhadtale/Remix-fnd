# API Reference

## Base URL

```
http://localhost:8000
```

## Endpoints

### GET /
Returns API information and available endpoints.

**Response:**
```json
{
  "name": "REMIX-FND",
  "version": "2.0.0",
  "status": "running",
  "endpoints": {...},
  "features": {...}
}
```

---

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "app": "REMIX-FND",
  "version": "2.0.0"
}
```

---

### POST /detect
Main endpoint for fake news detection.

**Request Body:**
```json
{
  "text": "News headline or article text",
  "include_explanation": true,
  "check_ai_generated": false
}
```

**Response:**
```json
{
  "prediction": "FAKE",
  "confidence": 92.5,
  "fake_probability": 92.5,
  "real_probability": 7.5,
  "explanation": {...}
}
```

---

### POST /explain
Get detailed explanation for a prediction.

**Request Body:**
```json
{
  "text": "News text to explain",
  "detail_level": "detailed"
}
```

**Detail Levels:**
- `simple` - One sentence summary
- `detailed` - Summary + key factors + highlighted words
- `expert` - Full technical analysis

**Response:**
```json
{
  "summary": "This content is likely fake news...",
  "key_factors": ["Contains sensational language", ...],
  "highlighted_words": [
    {"word": "shocking", "importance": 0.8, "is_suspicious": true}
  ],
  "confidence_breakdown": {...},
  "suggestions": [...]
}
```

---

### POST /evidence
Retrieve evidence for fact-checking.

**Request Body:**
```json
{
  "text": "Claim to fact-check",
  "max_results": 5
}
```

**Response:**
```json
{
  "query": "...",
  "evidence": [...],
  "verdict": "supported|contradicted|inconclusive",
  "confidence": 85.0
}
```

---

## Error Responses

All errors return:
```json
{
  "detail": "Error message"
}
```

Common status codes:
- `400` - Bad request (invalid input)
- `503` - Feature disabled
- `500` - Internal server error

