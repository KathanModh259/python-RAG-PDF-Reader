from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.di.container import ApplicationContainer
from app.infra.logging import logger


class QueryRequest(BaseModel):
    question: str
    mode: str = "standard"


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    model_loaded: bool


def create_app(container: ApplicationContainer) -> FastAPI:
    app = FastAPI(title="Legal AI API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount UI static assets
    workspace_root = Path(__file__).resolve().parent.parent.parent.parent
    assets_dir = workspace_root / "fe" / "dist" / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")



    @app.get("/health", response_model=HealthResponse)
    async def health():
        rag = container.rag_orchestrator()
        return HealthResponse(
            status="ok",
            documents_indexed=rag.vector_store.count,
            model_loaded=rag.llm.is_loaded,
        )

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        try:
            rag = container.rag_orchestrator()
            result = rag.query(request.question, mode=request.mode)
            return QueryResponse(**result)
        except Exception as e:
            logger.error("Query failed: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        ext = Path(file.filename).suffix.lower()
        allowed = container.config().allowed_extensions
        if ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {ext}. Allowed: {', '.join(allowed)}",
            )
        try:
            content = await file.read()
            if len(content) > container.config().max_file_size_mb * 1024 * 1024:
                raise HTTPException(status_code=400, detail="File too large")

            tmp = Path("data/raw/uploads") / file.filename
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(content)

            loader = container.document_loader()
            text = loader.load(tmp)
            if not text:
                raise HTTPException(status_code=400, detail="No text could be extracted")

            cleaner = container.cleaner()
            cleaned = cleaner.clean(text)

            rag = container.rag_orchestrator()
            chunk_count = rag.index_document(cleaned, source=file.filename)

            return {
                "status": "ok",
                "filename": file.filename,
                "chunks": chunk_count,
                "characters": len(cleaned),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Upload failed: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stats")
    async def stats():
        rag = container.rag_orchestrator()
        return {
            "documents_indexed": rag.vector_store.count,
            "model_loaded": rag.llm.is_loaded,
            "vector_store_type": container.config().vector_store_type,
        }

    @app.get("/")
    async def index():
        from fastapi.responses import FileResponse
        workspace_root = Path(__file__).resolve().parent.parent.parent.parent
        dist_index = workspace_root / "fe" / "dist" / "index.html"
        if dist_index.exists():
            return FileResponse(dist_index)
        return FileResponse(workspace_root / "fe" / "web" / "index.html")


    return app
