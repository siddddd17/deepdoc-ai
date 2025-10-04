import sys
import os
from docenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHisotory
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import craete_history_aware_retreiver, create_retriever_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils.model_loader import ModelLoader
from exception.custom_exception import DeepDocException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import PromptLibrary
from model.models import PromptType

class ConversationalRag: 
    def __init__(self) -> None:
        try: 
            self.log = CustomLogger().get_logger(__name__)
        except Exception as e:
            self.log.error(f"Error initializing ConversationalRag: {e}")
            raise DeepDocException("Error initializing ConversationalRag", sys)
    
    def _load_llm(self):
        try: 
            self.llm = ModelLoader().load_model()
        except Exception as e:
            self.log.error(f"Error loading LLM: {e}")
            raise DeepDocException("Error loading LLM", sys)
    
    def _get_session_history(self, session_id:str):
        try:
            pass
        except Exception as e:
            self.log.error(f"Error getting session history: {e}")
            raise DeepDocException("Error getting session history", sys)
        
    def _load_retriever(self):
        try:
            pass
        except Exception as e:
            self.log.error(f"Error loading retriever: {e}")
            raise DeepDocException("Error loading retriever", sys)
        
