"""
FastAPI Web API for PDF Legal Q&A System
Provides REST endpoints for PDF upload, processing, and question answering
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import os
import shutil
from datetime import datetime
import asyncio
import uvicorn

# Import our enhanced legal RAG system
from enhanced_legal_rag import EnhancedLegalRAG

# Initialize FastAPI app
app = FastAPI(
    title="PDF Legal Q&A API",
    description="REST API for analyzing PDF documents and answering legal questions using RAG with local Ollama",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
rag_system = None
current_pdf_info = None

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    use_enhanced_reasoning: Optional[bool] = True

class QuestionResponse(BaseModel):
    question: str
    answer: str
    confidence: Optional[float] = None
    sources: Optional[List[Dict[str, Any]]] = None
    timestamp: str
    processing_time: Optional[float] = None

class ProcessingStatus(BaseModel):
    status: str
    message: str
    pdf_info: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    ollama_available: bool
    pdf_loaded: bool

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF Legal Q&A API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 10px 0; }
            .endpoint { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .method { color: #007bff; font-weight: bold; }
            h1 { color: #333; }
            h2 { color: #555; }
        </style>
    </head>
    <body>
        <h1>📄 PDF Legal Q&A API</h1>
        <p>Welcome to the PDF Legal Question & Answer API. This service allows you to upload PDF documents and ask questions about their content using AI-powered analysis.</p>
        
        <div class="container">
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <p><span class="method">GET</span> <code>/health</code> - Check API health and status</p>
            </div>
            
            <div class="endpoint">
                <p><span class="method">POST</span> <code>/upload-pdf</code> - Upload and process a PDF file</p>
            </div>
            
            <div class="endpoint">
                <p><span class="method">POST</span> <code>/ask</code> - Ask a question about the loaded PDF</p>
            </div>
            
            <div class="endpoint">
                <p><span class="method">GET</span> <code>/pdf-info</code> - Get information about the currently loaded PDF</p>
            </div>
            
            <div class="endpoint">
                <p><span class="method">GET</span> <code>/docs</code> - Interactive API documentation (Swagger UI)</p>
            </div>
        </div>
        
        <div class="container">
            <h2>Quick Start:</h2>
            <ol>
                <li>Check API health: <a href="/health">GET /health</a></li>
                <li>Upload a PDF file: <code>POST /upload-pdf</code></li>
                <li>Ask questions: <code>POST /ask</code></li>
            </ol>
        </div>
        
        <p><a href="/docs">📚 View Interactive API Documentation</a></p>
    </body>
    </html>
    """
    return html_content

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and dependencies"""
    try:
        # Check if Ollama is available
        import ollama
        ollama_available = True
        try:
            # Try to list models to verify Ollama connection
            models = ollama.list()
            ollama_available = True
        except:
            ollama_available = False
        
        # Check if PDF is loaded
        pdf_loaded = rag_system is not None and len(rag_system.chunks) > 0
        
        return HealthResponse(
            status="healthy",
            message="API is running properly",
            ollama_available=ollama_available,
            pdf_loaded=pdf_loaded
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/upload-pdf", response_model=ProcessingStatus)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    global rag_system, current_pdf_info
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        # Initialize RAG system
        rag_system = EnhancedLegalRAG()
        
        # Save current working directory and change to temp directory
        original_cwd = os.getcwd()
        temp_dir = os.path.dirname(tmp_path)
        temp_filename = os.path.basename(tmp_path)
        
        try:
            os.chdir(temp_dir)
            # Rename temp file to match expected pattern
            final_path = os.path.join(temp_dir, file.filename)
            shutil.move(tmp_path, final_path)
            
            # Load and process PDF
            text = rag_system.load_pdf()
            if not text:
                raise HTTPException(status_code=400, detail="Failed to extract text from PDF")
            
            # Process the document
            rag_system.process_document(text)
            
            # Store PDF info
            current_pdf_info = {
                "filename": file.filename,
                "upload_time": datetime.now().isoformat(),
                "chunks_count": len(rag_system.chunks),
                "total_characters": len(text)
            }
            
        finally:
            # Restore working directory and cleanup
            os.chdir(original_cwd)
            try:
                os.unlink(final_path)
            except:
                pass
        
        return ProcessingStatus(
            status="success",
            message=f"PDF '{file.filename}' processed successfully",
            pdf_info=current_pdf_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the loaded PDF"""
    if rag_system is None or len(rag_system.chunks) == 0:
        raise HTTPException(status_code=400, detail="No PDF loaded. Please upload a PDF first.")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        start_time = datetime.now()
        
        # Ask the question using enhanced reasoning if requested
        if request.use_enhanced_reasoning:
            answer = rag_system.ask_question(request.question)
        else:
            # Use basic RAG approach (you can implement a simpler version if needed)
            answer = rag_system.ask_question(request.question)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Try to extract confidence and sources (if available in your RAG system)
        confidence = None  # You can implement confidence scoring
        sources = []  # You can implement source tracking
        
        return QuestionResponse(
            question=request.question,
            answer=answer,
            confidence=confidence,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")

@app.get("/pdf-info")
async def get_pdf_info():
    """Get information about the currently loaded PDF"""
    if current_pdf_info is None:
        raise HTTPException(status_code=404, detail="No PDF currently loaded")
    
    return JSONResponse(content=current_pdf_info)

@app.delete("/clear-pdf")
async def clear_pdf():
    """Clear the currently loaded PDF"""
    global rag_system, current_pdf_info
    
    rag_system = None
    current_pdf_info = None
    
    return JSONResponse(content={"status": "success", "message": "PDF cleared successfully"})

# Additional utility endpoints

@app.get("/models")
async def list_ollama_models():
    """List available Ollama models"""
    try:
        import ollama
        models = ollama.list()
        return JSONResponse(content={"models": models})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.post("/test-ollama")
async def test_ollama():
    """Test Ollama connection"""
    try:
        import ollama
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": "Hello, please respond with 'OK' if you're working."}]
        )
        return JSONResponse(content={
            "status": "success",
            "message": "Ollama is working",
            "response": response["message"]["content"]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama test failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting PDF Legal Q&A API...")
    print("📚 Interactive docs available at: http://localhost:8000/docs")
    print("🏠 Home page available at: http://localhost:8000/")
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )