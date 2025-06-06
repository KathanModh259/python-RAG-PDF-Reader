# FastAPI PDF Legal Q&A API Documentation

## Overview

The FastAPI PDF Legal Q&A API provides REST endpoints for uploading PDF documents and asking questions about their content using RAG (Retrieval-Augmented Generation) with local Ollama models.

## Features

- **PDF Upload & Processing**: Upload PDF files and extract text content
- **Question Answering**: Ask questions about the loaded PDF using enhanced legal reasoning
- **Health Monitoring**: Check API status, Ollama connectivity, and PDF loading status
- **Interactive Documentation**: Auto-generated Swagger UI at `/docs`
- **Cross-Origin Support**: CORS enabled for web applications

## Quick Start

### 1. Start the Server

```bash
# Option 1: Direct Python
python fastapi_app.py

# Option 2: Using batch file (Windows)
run_fastapi.bat

# Option 3: Using uvicorn directly
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Access the API

- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Home Page**: http://localhost:8000/

### 3. Test the API

```bash
# Option 1: Use test script
python test_api.py

# Option 2: Use batch file (Windows)
test_api.bat
```

## API Endpoints

### Health & Status

#### `GET /health`
Check API health and dependencies.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running properly",
  "ollama_available": true,
  "pdf_loaded": false
}
```

#### `GET /pdf-info`
Get information about the currently loaded PDF.

**Response:**
```json
{
  "filename": "document.pdf",
  "upload_time": "2024-01-15T10:30:00",
  "chunks_count": 45,
  "total_characters": 15420
}
```

### PDF Management

#### `POST /upload-pdf`
Upload and process a PDF file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: PDF file

**Response:**
```json
{
  "status": "success",
  "message": "PDF 'document.pdf' processed successfully",
  "pdf_info": {
    "filename": "document.pdf",
    "upload_time": "2024-01-15T10:30:00",
    "chunks_count": 45,
    "total_characters": 15420
  }
}
```

#### `DELETE /clear-pdf`
Clear the currently loaded PDF from memory.

**Response:**
```json
{
  "status": "success",
  "message": "PDF cleared successfully"
}
```

### Question Answering

#### `POST /ask`
Ask a question about the loaded PDF.

**Request:**
```json
{
  "question": "What are the main legal requirements?",
  "use_enhanced_reasoning": true
}
```

**Response:**
```json
{
  "question": "What are the main legal requirements?",
  "answer": "Based on the document, the main legal requirements are...",
  "confidence": null,
  "sources": [],
  "timestamp": "2024-01-15T10:35:00",
  "processing_time": 2.34
}
```

### Utility Endpoints

#### `GET /models`
List available Ollama models.

**Response:**
```json
{
  "models": {
    "models": [
      {
        "name": "llama3.2:1b",
        "model": "llama3.2:1b",
        "size": 1234567890
      }
    ]
  }
}
```

#### `POST /test-ollama`
Test Ollama connection and functionality.

**Response:**
```json
{
  "status": "success",
  "message": "Ollama is working",
  "response": "OK"
}
```

## Usage Examples

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Check health
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Upload PDF
with open("document.pdf", "rb") as f:
    files = {"file": ("document.pdf", f, "application/pdf")}
    response = requests.post(f"{BASE_URL}/upload-pdf", files=files)
    print(response.json())

# Ask question
question_data = {
    "question": "What is this document about?",
    "use_enhanced_reasoning": True
}
response = requests.post(f"{BASE_URL}/ask", json=question_data)
print(response.json())
```

### cURL Examples

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Upload PDF
curl -X POST "http://localhost:8000/upload-pdf" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"

# Ask question
curl -X POST "http://localhost:8000/ask" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is this document about?",
    "use_enhanced_reasoning": true
  }'
```

### JavaScript/Fetch Example

```javascript
// Upload PDF
const formData = new FormData();
formData.append('file', pdfFile);

fetch('http://localhost:8000/upload-pdf', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));

// Ask question
fetch('http://localhost:8000/ask', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'What is this document about?',
    use_enhanced_reasoning: true
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Error Handling

The API returns appropriate HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid file, empty question, etc.)
- **404**: Not Found (no PDF loaded)
- **500**: Internal Server Error

Error responses include details:
```json
{
  "detail": "No PDF loaded. Please upload a PDF first."
}
```

## Configuration

### Environment Variables

You can configure the API using environment variables:

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `OLLAMA_MODEL`: Default Ollama model (default: llama3.2:1b)

### Model Configuration

The API uses the Enhanced Legal RAG system which can be configured in `enhanced_legal_rag.py`:

- Embedding model: `all-MiniLM-L6-v2`
- Chunking strategy: Legal structure-based
- Ollama model: `llama3.2:1b`

## Dependencies

- FastAPI: Web framework
- Uvicorn: ASGI server
- Enhanced Legal RAG: Custom PDF analysis system
- Ollama: Local LLM inference
- FAISS: Vector similarity search
- SentenceTransformers: Text embeddings

## Troubleshooting

### Common Issues

1. **Ollama not available**
   - Ensure Ollama is installed and running
   - Check if the model is downloaded: `ollama pull llama3.2:1b`

2. **PDF upload fails**
   - Check file format (must be PDF)
   - Verify file is not corrupted
   - Check file size limits

3. **Questions not working**
   - Ensure PDF is uploaded first
   - Check if question is not empty
   - Verify Ollama model is available

### Logs

The API provides detailed logging. Check the console output for error messages and processing information.

## Development

### Running in Development Mode

```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

### Testing

Use the provided test script to validate all endpoints:

```bash
python test_api.py
```

## Production Deployment

For production deployment, consider:

1. **Security**: Configure CORS properly, add authentication
2. **Performance**: Use production ASGI server, load balancing
3. **Monitoring**: Add logging, metrics, health checks
4. **Storage**: Implement persistent storage for PDFs
5. **Scaling**: Consider containerization with Docker
