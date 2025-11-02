from __future__ import annotations
import os
import sys 
import json
import uuid 
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Iterable

import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception_archive import DeepDocException

from utils.file_io import generate_session_id, save_uploaded_files
from utils.document_ops import load_documents, concat_for_analysis, concat_for_comparison

from logger import GLOBAL_LOGGER as log

SUPPORTED_FILE_TYPES = {'.pdf', '.docx', '.txt', '.md'}

class FaissManager: 

    def __init__(self, index_dir: Path , model_loader: Optional[ModelLoader] = None):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.index_dir / "ingested_metadata.json"
        self._meta:Dict[str, Any] = {"rows" : {}}

        if self.metadata_file.exists():
            try: 
                self._meta = json.loads(self.metadata_file.read_text(encoding = 'utf-8')) or {"rows" : {}}
            except Exception as e:
                log.info("No existing metadata found, starting fresh.")
                raise DeepDocException("Error loading existing metadata", sys) from e
            
        self.model_loader = model_loader or ModelLoader()
        self.embedding_model = self.model_loader.load_embeddings()
        self.vector_store: Optional[FAISS] = None

    def _exists(self) -> bool:
        log.info("Checking for existing FAISS index and metadata")
        return (self.metadata_file / "index.pk1").exists() and (self.index_dir / "index.faiss").exists()
    
    @staticmethod
    def _fingerprint(text: str, md:Dict[str, Any]) -> str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None: 
            return f"{src} :: {'' if rid is None else rid}"
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _save_metadata(self):
        self.metadata_path.write_text(json.dumps(self._meta, ensure_ascii = False, indent=4), encoding = 'utf-8')

    def load_or_create(self,texts:Optional[List[str]]=None, metadatas: Optional[List[dict]] = None):
        ## if we running first time then it will not go in this block
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True,
            )
            return self.vs
        
        
        if not texts:
            raise DeepDocException("No existing FAISS index and no data to create one", sys)
        self.vs = FAISS.from_texts(texts=texts, embedding=self.embedding_model, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs
    
    def add_documents(self, docs: List[Document]):
        if self.vs is None:
            raise DeepDocException("Vector store is not initialized", sys)
        new_docs: List[Document] = []
        for doc in new_docs:
            fp = self._fingerprint(doc.page_content, doc.metadata)
            if fp not in self._meta["rows"]:
                new_docs.append(doc)
                self._meta["rows"][fp] = doc.metadata
        if new_docs:
            self.vector_store.add_documents(new_docs)
            self._save_metadata()
            self.vector_store.save_local(str(self.index_dir))
            self.log.info(f"Added {len(new_docs)} new documents to FAISS index", session_id=self.session_id)
        return len(new_docs)

class DocHandler: 
    """
    PDF save + read (page-wise) for analysis.
    """
    def __init__(self, data_dir: Optional[str] = None, session_id: Optional[str] = None):
        self.log=CustomLogger().get_logger(__name__)
        self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis"))
        self.session_id = session_id or generate_session_id("session")
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        log.info("DocHandler initialized", session_id=self.session_id, session_path=self.session_path)

    def save_pdf(self, uploaded_file) -> str:
        try:
            filename = os.path.basename(uploaded_file.name)
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Invalid file type. Only PDFs are allowed.")
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            log.info("PDF saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            log.error("Failed to save PDF", error=str(e), session_id=self.session_id)
            raise DeepDocException(f"Failed to save PDF: {str(e)}", e) from e

    def read_pdf(self, pdf_path: str) -> str:
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_chunks.append(f"\n--- Page {page_num + 1} ---\n{page.get_text()}")  # type: ignore
            text = "\n".join(text_chunks)
            log.info("PDF read successfully", pdf_path=pdf_path, session_id=self.session_id, pages=len(text_chunks))
            return text
        except Exception as e:
            log.error("Failed to read PDF", error=str(e), pdf_path=pdf_path, session_id=self.session_id)
            raise DeepDocException(f"Could not process PDF: {pdf_path}", e) from e

class DocumentComparator: 
    """
    Save, read & combine PDFs for comparison with session-based versioning.
    """
    def __init__(self, base_dir: str = "data/document_compare", session_id: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.session_id = session_id or generate_session_id()
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        log.info("DocumentComparator initialized", session_path=str(self.session_path))

    def save_uploaded_files(self, reference_file, actual_file):
        try:
            ref_path = self.session_path / reference_file.name
            act_path = self.session_path / actual_file.name
            for fobj, out in ((reference_file, ref_path), (actual_file, act_path)):
                if not fobj.name.lower().endswith(".pdf"):
                    raise ValueError("Only PDF files are allowed.")
                with open(out, "wb") as f:
                    if hasattr(fobj, "read"):
                        f.write(fobj.read())
                    else:
                        f.write(fobj.getbuffer())
            log.info("Files saved", reference=str(ref_path), actual=str(act_path), session=self.session_id)
            return ref_path, act_path
        except Exception as e:
            log.error("Error saving PDF files", error=str(e), session=self.session_id)
            raise DeepDocException("Error saving files", e) from e

    def read_pdf(self, pdf_path: Path) -> str:
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted: {pdf_path.name}")
                parts = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()  # type: ignore
                    if text.strip():
                        parts.append(f"\n --- Page {page_num + 1} --- \n{text}")
            log.info("PDF read successfully", file=str(pdf_path), pages=len(parts))
            return "\n".join(parts)
        except Exception as e:
            log.error("Error reading PDF", file=str(pdf_path), error=str(e))
            raise DeepDocException("Error reading PDF", e) from e

    def combine_documents(self) -> str:
        try:
            doc_parts = []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() == ".pdf":
                    content = self.read_pdf(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")
            combined_text = "\n\n".join(doc_parts)
            log.info("Documents combined", count=len(doc_parts), session=self.session_id)
            return combined_text
        except Exception as e:
            log.error("Error combining documents", error=str(e), session=self.session_id)
            raise DeepDocException("Error combining documents", e) from e

    def clean_old_sessions(self, keep_latest: int = 3):
        try:
            sessions = sorted([f for f in self.base_dir.iterdir() if f.is_dir()], reverse=True)
            for folder in sessions[keep_latest:]:
                shutil.rmtree(folder, ignore_errors=True)
                self.log.info("Old session folder deleted", path=str(folder))
        except Exception as e:
            log.error("Error cleaning old sessions", error=str(e))
            raise DeepDocException("Error cleaning old sessions", e) from e
        
class ChatIngestor: 
    def __init__(
        self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.model_loader = ModelLoader()
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()
            self.temp_base = Path(temp_base)
            self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base)
            self.faiss_base.mkdir(parents=True, exist_ok=True)
            self.temp_dir = self._resolve_dir(temp_base)
            self.faiss_dir = self._resolve_dir(faiss_base)

            self.log.info(
                "ChatIngestor initialized",
                temp_dir=str(self.temp_dir),
                faiss_dir=str(self.faiss_dir),
                session_id=self.session_id,
                sessionized = self.use_session 
                )

        except Exception as e:
            self.log.error(f"Error initializing ChatIngestor: {e}")
            raise DeepDocException("Error initializing ChatIngestor", sys) from e

    def _resolve_dir(self, base_dir: str) -> Path:
        if self.use_session:
            dir_path = Path(base_dir) / self.session_id
        else:
            dir_path = Path(base_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.log.info("Text splitter initialized", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(text)

    def build_retriever(
    self, 
    *,
    uploaded_files: Iterable,
    chunk_size: int = 100, 
    chunk_overlap: int = 200, 
    k: int = 5
    ):
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")
            
            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            ## FAISS manager important class for the docchat
            fm = FaissManager(self.faiss_dir, self.model_loader)
            
            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]
            
            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
                
            added = fm.add_documents(chunks)
            self.log.info("FAISS index updated", added=added, index=str(self.faiss_dir))
            
            return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
            
        except Exception as e:
            # log.error("Failed to build retriever", error=str(e))
            raise DeepDocException("Failed to build retriever", e) from e
        
