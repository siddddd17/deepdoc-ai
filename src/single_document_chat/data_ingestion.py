import uuid
from pathlib import Path
import sys
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DeepDocException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.model_loader import ModelLoader
from langchain_community.vectorstores import FAISS
from datetime import datetime

class DataIngestion:  
    def __init__(self, data_dir: str = "data/single_document_chat", faiss_dir: str = "faiss_index",session_id: str = None):
        try:
            self.log = CustomLogger().get_logger(__name__)
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.faiss_dir = Path(faiss_dir)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)
            self.session_id = session_id or self._generate_session_id()
            self.session_path = self.data_dir / self.session_id
            self.session_path.mkdir(parents=True, exist_ok=True)
            self.model_loader = ModelLoader()

            self.log.info("DataIngestion initialized", data_dir=str(self.data_dir), faiss_dir=str(self.faiss_dir), session_id=self.session_id)
        except Exception as e:
            raise DeepDocException("Error initializing DataIngestion", sys)
        
    def _generate_session_id(self) -> str:
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def ingest_files(self, uploaded_file: Path) -> str:
        try:
            documents = []
            for file in uploaded_file:
                if not file.name.lower().endswith('.pdf'):
                    raise DeepDocException("Invalid file type. Only PDFs are allowed.")
                unique_filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{Path(file.name).name}";
                save_path = self.session_path / unique_filename
                with open(save_path, "wb") as f:
                    f.write(file.read())
                self.log.info("File saved for ingestion", file=file.name, save_path=str(save_path), session_id=self.session_id)
                
                loader=PyPDFLoader(str(save_path))
                docs=loader.load()
                documents.extend(docs)
            self.log.info("File loaded", count=len(documents))
            return self._create_retriever(documents)
        except Exception as e:
            self.log.error(f"Error ingesting PDF: {e}")
            raise DeepDocException("Error ingesting PDF", sys)
    
    def _create_retriever(self, documents: list):
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
            chunks = splitter.split_documents(documents)
            self.log.info("Documents split into chunks", chunk_count=len(chunks))
            embeddings = self.model_loader.load_embeddings()
            vector_store = FAISS.from_documents(chunks, embeddings)

            #save the vector store
            vector_store.save_local(str(self.faiss_dir))
            self.log.info("FAISS vector store created and saved", faiss_dir=str(self.faiss_dir))

            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":5})
            self.log.info("Retriever created", retriever_type="FAISS", top_k=5)
            return retriever

        except Exception as e:
            self.log.error(f"Error creating retriever: {e}")
            raise DeepDocException("Error creating retriever", sys) 
    

