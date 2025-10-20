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
                raise DeepDocException("Error loading existing metadata", sys) from e
            
        self.model_loader = model_loader or ModelLoader()
        self.embedding_model = self.model_loader.load_embeddings()
        self.vector_store: Optional[FAISS] = None

    def _exists(self) -> bool:
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

    def load_or_create(self, docs: List[Document]):
        



    def add_documents(self):
        if self.vs is None:
            raise DeepDocException("Vector store is not initialized", sys)
        


class DocHandler: 
    def __init__(self):
        pass 

    def save_pdf(self):
        pass

    def read_pdf(self):
        pass

class DocumentComparator: 
    def __init__(self):
        pass

    def save_uploaded_files(self):
        pass

    def read_pdf(self):
        pass

    def combine_documents(self):
        pass

    def clean_old_sessions(self):
        pass

class ChatIngestor: 
    def __init__(self):
        pass 

    def _resolve_dir(self):
        pass

    def _split(self):
        pass

    def ingest_files(self):
        pass 

    def build_retriever(self):
        pass