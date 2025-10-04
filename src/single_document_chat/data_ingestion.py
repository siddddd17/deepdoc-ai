import uuid
from pathlib import Path
import sys
from langchain_community.document_loaders import PyMuPDFLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DeepDocException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.model_loader import ModelLoader
from langchain.community.vectorstores import FAISS

class DataIngestion:  
    def __init__(self):
        try:
            self.log = CustomLogger().get_logger(__name__)
        except Exception as e:
            raise DeepDocException("Error initializing DataIngestion", sys)

    def ingest_pdf(self):
        try:
            pass
        except Exception as e:
            self.log.error(f"Error ingesting PDF: {e}")
            raise DeepDocException("Error ingesting PDF", sys)
    
    def create_retriever(self):
        try:
            pass
        except Exception as e:
            self.log.error(f"Error creating retriever: {e}")
            raise DeepDocException("Error creating retriever", sys) 
    

