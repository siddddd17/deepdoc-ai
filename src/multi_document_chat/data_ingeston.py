import uuid
from pathlib import Path
import sys
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, Docx2txtLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DeepDocException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.model_loader import ModelLoader
from langchain_community.vectorstores import FAISS
from datetime import datetime

class document_ingestor: 
    SUPPORTED_FILE_TYPES = {'.pdf', '.docx', '.txt', '.md'}
    def __init__(self, temp_dir: str= 'data/multi_document_chat', faiss_dir: str= 'faiss_index', session_id: str = None):
        try:
            self.log = CustomLogger().get_logger(__name__)

            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.faiss_dir = Path(faiss_dir)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)

            self.session_id = session_id or self._generate_session_id()
            self.session_temp_path = self.temp_dir / self.session_id
            self.session_temp_path.mkdir(parents=True, exist_ok=True)
            self.sesssion_faiss_path = self.faiss_dir / self.session_id
            self.sesssion_faiss_path.mkdir(parents=True, exist_ok=True)

            self.model_loader = ModelLoader()

            self.log.info("document_ingestor initialized",
                           temp_dir=str(self.temp_dir),
                             faiss_dir=str(self.faiss_dir),
                               session_id=self.session_id)
        except Exception as e:
            self.log.error(f"Error initializing document_ingestor: {e}")
            raise DeepDocException("Error initializing document_ingestor", sys)

    def ingest_file(self, uploaded_files) -> str: 
        try:
            documents = []
            for file in uploaded_files:
                if not any(file.name.lower().endswith(ext) for ext in self.SUPPORTED_FILE_TYPES):
                    self.log.error(f"Unsupported file type: {file.name}")
                    raise DeepDocException("Invalid file type. Supported types are: PDF, DOCX, TXT, MD.")
                
                unique_filename = f"{uuid.uuid4().hex[:8]}_{Path(file.name).name}"
                save_path = self.session_temp_path / unique_filename

                with open(save_path, "wb") as f:
                    f.write(file.read())
                self.log.info("File saved for ingestion", file=file.name, save_path=str(save_path), session_id=self.session_id)
                
                suffix = Path(file.name).suffix.lower()
                match suffix:
                    case '.pdf':
                        loader = PyPDFLoader(str(save_path))
                    case '.docx':
                        loader = Docx2txtLoader(str(save_path))
                    case '.md':
                        loader = UnstructuredMarkdownLoader(str(save_path))
                    case '.txt':
                        loader = TextLoader(str(save_path), encoding = 'utf-8')
                    case _:
                        self.log.error(f"Unsupported file type in match: {file.name}")
                        raise DeepDocException("Invalid file type. Supported types are: PDF, DOCX, TXT, MD.")
                
                docs = loader.load()
                documents.extend(docs)

                if not documents:
                    self.log.error(f"No documents loaded from file: {file.name}")
                    raise DeepDocException("No documents could be loaded from the uploaded files.", sys)

            self.log.info("Files loaded", count=len(documents), session_id=self.session_id)
            return self._create_retriever(documents)
        except Exception as e:
            self.log.error(f"Error ingesting file: {e}")
            raise DeepDocException("Error ingesting file", sys)
        
    def _generate_session_id(self) -> str:
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _create_retriever(self, documents: list):
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
            chunks = splitter.split_documents(documents)
            self.log.info("Documents split into chunks", chunk_count=len(chunks), session_id=self.session_id)
            embeddings = self.model_loader.load_embeddings()
            vector_store = FAISS.from_documents(chunks, embeddings)

            #save the vector store
            vector_store.save_local(str(self.sesssion_faiss_path))
            self.log.info("FAISS vector store created and saved", faiss_dir=str(self.sesssion_faiss_path), session_id=self.session_id)

            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":5})
            self.log.info("Retriever created", retriever_type="FAISS", top_k=5, session_id=self.session_id)
            return retriever
        
        except Exception as e:
            self.log.error(f"Error creating retriever: {e}")
            raise DeepDocException("Error creating retriever", sys)


