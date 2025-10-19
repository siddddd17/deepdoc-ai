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

class FaissManager: 
    def __init__(self):
        pass 

    def _exists(self):
        pass 

    @staticmethod
    def _fingerprint():
        pass 

    def _save_metadata(self):
        pass

    def load_or_create(self):
        pass

    def add_documents(self):
        pass


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