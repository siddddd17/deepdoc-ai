import os
import sys
from pathlib import Path
from operator import itemgetter
from typing import List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_community.vectorstores import FAISS
from utils.model_loader import ModelLoader
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType
from logger.custom_logger import CustomLogger
from exception.custom_exception import DeepDocException

class ConversationalRAG:
    def __init__(self, session_id: str, retriever) -> None:
        try: 
            self.log = CustomLogger().get_logger(__name__)
            self.session_id = session_id
            if retriever is None:
                raise DeepDocException("Retriever cannot be None", sys)
            self.retriever = retriever
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_ANSWER.value]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
            self.history_aware_retriever = None
            self._build_lcel_chain()
            self.log.info("ConversationalRAG initialized", session_id=self.session_id)

        except Exception as e:
            self.log.error(f"Error initializing ConversationalRAG: {e}")
            raise DeepDocException("Error initializing ConversationalRAG", sys)
    
    def load_retriever_from_faiss(self, index_path: str = None):
        """
        Load FAISS retriever from the specified index path.
        """
        try:
            embeddings = ModelLoader().load_embeddings()
            faiss_index_path = index_path or os.path.join("faiss_index", self.session_id)
            if not os.path.exists(faiss_index_path):
                raise DeepDocException(f"FAISS index path does not exist: {faiss_index_path}", sys)
            self.retriever = FAISS.load_local(faiss_index_path,
                                             embeddings,
                                             allow_dangerous_serialization = True).as_retriever(search_type = "similarity", search_kwargs={"k":5}) # load retriever from faiss vector store
            self.log.info("FAISS retriever loaded successfully", index_path=faiss_index_path, session_id=self.session_id)
            self._build_lcel_chain()
            return self.retriever
        except Exception as e:
            self.log.error(f"Error loading FAISS retriever: {e}")
            raise DeepDocException("Error loading FAISS retriever", sys)
        
    def invoke(self, question: str, chat_history:Optional[List[BaseMessage]] = None ) -> str:
        """
        Invoke the Conversational RAG chain with the given question.
        """
        try:
            chat_history = chat_history or []
            if not question:
                raise DeepDocException("Question cannot be empty", sys)
            payload = {"input": question, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                self.log.error("No answer returned from the RAG chain")
                raise DeepDocException("No answer returned from the RAG chain", sys)
            self.log.info("Conversational RAG invoked successfully", question=question, session_id=self.session_id, answer=answer[:100])
            return answer
        except Exception as e:
            self.log.error(f"Error invoking Conversational RAG: {e}")
            raise DeepDocException("Error invoking Conversational RAG", sys)
        
    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if llm is None:
                raise DeepDocException("LLM model could not be loaded", sys)
            self.log.info("LLM model loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            self.log.error(f"Error loading LLM: {e}")
            raise DeepDocException("Error loading LLM", sys)

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    
    def _build_lcel_chain(self):
        try: 
            
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )
            retrieve_docs = question_rewriter | self.retriever | self._format_docs
            self.chain = (
                {
                    "context" : retrieve_docs, 
                    "input" : itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )
            self.log.info("LCEL chain built successfully", session_id=self.session_id)
        except Exception as e:
            self.log.error(f"Error building Conversational RAG chain: {e}")
            raise DeepDocException("Error building Conversational RAG chain", sys)



