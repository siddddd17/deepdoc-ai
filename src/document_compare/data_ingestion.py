import sys
from pathlib import Path
import fitz
from datetime import datetime
import uuid
from logger.custom_logger import CustomLogger
from exception.custom_exception import DeepDocException

#TODO: Use factory pattern 
class DocumentIngestion:
    def __init__(self,base_dir:str="data/document_compare", session_id: Optional[str]= None):
        self.log = CustomLogger().get_logger(__name__)
        self.base_dir = Path(base_dir)
        self.session_id = session_id or _generate_session_id()
        self.base_dir = self.base_dir / self.session_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        log.info("DocumentIngestion initialized", base_dir=str(self.base_dir), session_id=self.session_id)

    def _generate_session_id() -> str:
        return f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def delete_existing_files(self):
        """
        Deletes existing files at the specified paths.
        """
        try:
            if self.base_dir.exists() and self.base_dir.is_dir():
                for file in self.base_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                        self.log.info("File deleted", path=str(file))
                self.log.info("Directory cleaned", directory=str(self.base_dir))
        except Exception as e:
            self.log.error(f"Error deleting existing files: {e}")
            raise DeepDocException("An error occurred while deleting existing files.", sys)
     
    def save_uploaded_files(self,reference_file, actual_file):
        """
        Saves uploaded files to a specific directory.
        """
        try:
            ref_path = self.base_dir / reference_file.name
            act_path = self.base_dir / actual_file.name
            for file_object, output_path in ((reference_file, ref_path), (actual_file, act_path)):
                if not file_object.name.lower().endswith('.pdf'):
                    raise ValueError("Invalide File Type. Only PDFs are allowed.")
                if file_object.hasAttr("getbuffer") :
                    with open(output_path, "wb") as f: 
                        f.write(file_object.getbuffer())
                else : 
                    with open(output_path, "wb") as f: 
                        f.write(file_object.read())
            self.log.info("Files saved", reference=str(ref_path), actual=str(act_path))
            return ref_path, act_path
        except Exception as e:
            self.log.error(f"Error saving uploaded files: {e}", error=str(e), 
                           reference_file=reference_file.name, actual_file=actual_file.name, session_id=self.session_id)
            raise DeepDocException("An error occurred while saving uploaded files.", sys)

    def read_pdf(self,pdf_path: Path)->str:
        """
        Reads a PDF file and extracts text from each page.
        """
        try:
             with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted: {pdf_path.name}")
                all_text = []
                for page_num in range(doc.page_count):
                    page=doc.load_page(page_num)
                    text = page.get_text() #type: ignore
                    if text.strip():
                        all_text.append(f"\n --- Page {page_num + 1} --- \n{text}")
                self.log.info("PDF read successfully", file=str(pdf_path), pages=len(all_text))
                return "\n".join(all_text)
        except Exception as e:
            self.log.error(f"Error reading PDF: {e}")
            raise DeepDocException("An error occurred while reading the PDF.", sys)
        
    def combine_documents(self)->str:
        try:
            content_dict = {}
            doc_parts = []

            for filename in sorted(self.base_dir.iterdir()):
                if filename.is_file() and filename.suffix == ".pdf":
                    content_dict[filename.name] = self.read_pdf(filename)

            for filename, content in content_dict.items():
                doc_parts.append(f"Document: {filename}\n{content}")

            combined_text = "\n\n".join(doc_parts)
            self.log.info("Documents combined", count=len(doc_parts))
            return combined_text

        except Exception as e:
            self.log.error(f"Error combining documents: {e}")
            raise DeepDocException("An error occurred while combining documents.", sys)
    
    def clean_old_sessions(self, keep_latest = 3):
        """
        Cleans up old session directories, keeping only the latest 'keep_latest' sessions.
        """
        try: 
            session_dirs = sorted([ d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("session_")] , reverse=True) 
            for old_dir in session_dirs[keep_latest:]:
                shutil.rmtree(old_dir)
                self.log.info("Old session directory deleted", path=str(old_dir))
        except Exception as e:
            self.log.error(f"Error cleaning old sessions: {e}")
            raise DeepDocException("An error occurred while cleaning old sessions.", sys)

