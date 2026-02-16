from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os 
from typing import Any, Optional, Dict, List

from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator, 
    ChatIngestor, 
    FaissManager
)

from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from utils.document_ops import read_pdf_via_handler , FastAPIFileADapter
from logger import GLOBAL_LOGGER as log

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")

app = FastAPI(title="DeepDoc AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_class=JSONResponse)
async def health_check():
    """
    Health check endpoint.
    """
    log.info("Health check passed")
    return JSONResponse(content={"status": "ok"}, status_code=200)

def _read_pdf_via_handler(handler: DocHandler, pdf_path: str) -> str:
    """
    Helper function to read PDF using DocHandler
    """
    try:
        text = handler.read_pdf(pdf_path)
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read PDF at {pdf_path}") from e
    
@app.post("/analyze", response_class=JSONResponse)
async def analyse_documents(file: UploadFile = File(...)) -> Any: 
    """
    Endpoint to analyze a single PDF document
    """
    try: 
        log.info("Received file for analysis", filename=file.filename)
        dh = DocHandler() 
        save_path = dh.save_pdf(FastAPIFileADapter(file))
        text = read_pdf_via_handler(dh, save_path)
        analyzer = DocumentAnalyzer()
        analysis_result = analyzer.analyze_document(text)
        log.info("Document analysis completed", filename=file.filename)
        return JSONResponse(content=analysis_result, status_code=200)
    except HTTPException as http_exc:
        log.error("HTTP exception during document analysis", error=str(http_exc))
        raise http_exc
    except Exception as e:
        log.error("Exception during document analysis", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed") from e

@app.post("/compare", response_class=JSONResponse)
async def compare_documents(reference_file: UploadFile = File(...), actual_file: UploadFile = File(...)) -> Any:
    """ 
    Endpoint to compare two PDF documents
    """
    try:
        log.info("Received files for comparison",
                 reference_filename=reference_file.filename,
                 actual_filename=actual_file.filename)
        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(FastAPIFileADapter(reference_file), FastAPIFileADapter(actual_file))
        _ = ref_path, act_path
        combined_text = dc.combine_documents()
        comparator = DocumentComparatorLLM()
        df = comparator.compare_documents(combined_text)
        log.info("Document comparison completed", session_id=dc.session_id)
        return JSONResponse(
            content={
                "records": df.to_dict(orient="records"),
                "session_id": dc.session_id
            }
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail = f"Comparison failed") from e

@app.post('/chat/index')
async def build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5)
) -> Any:
    """
    Endpoint to ingest documents and build FAISS index for chat
    """
    try:
        wrapped = [FastAPIFileADapter(f) for f in files]
        ci = ChatIngestor(
            temp_base = UPLOAD_BASE, 
            faiss_base = FAISS_BASE,
            use_session_dirs = use_session_dirs,
            session_id = session_id or None
        )
        ci.build_retriever(uploaded_files=wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k)
        return {
            "session_id" : ci.session_id, 
            "k" : k,
            "use_session_dirs" : use_session_dirs,
            "engine" : "LCEL-RAG"
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        log.error("Failed ", error=str(e))
        raise HTTPException(status_code=500, detail = f"Document chat failed") from e

@app.post('/chat/query')
async def chat_with_query(
    user_query: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True)
) -> Any:
    """
    Endpoint to chat with ingested documents using a query
    """
    try:
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="Session ID must be provided when use_session_dirs is True.")

        #Prepare Faiss index path 
        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE # type : ignore
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index directory not found: {index_dir}")

        rag = ConversationalRAG(session_id = session_id) #type : ignore 
        rag.load_retriever_from_faiss(index_dir)

        response = rag.invoke(user_query, chat_history=[])
        result = {
            "answer" : response, 
            "session_id" : session_id, 
            "engine" : "LCEL-RAG"
        }
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:  
        raise HTTPException(status_code=500, detail=f"Chat with query failed") from e
    
