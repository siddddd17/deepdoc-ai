import sys
from dotenv import load_dotenv
import pandas as pd
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from model.models import *
from prompt.prompt_library import PROPMT_REGISTRY
from utils.model_loader import ModelLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser

class DocumentComparatorLLM:
    def __init__(self):
        pass
    
    def compare_documents(self):
        """
        Compares two documents and returns a structured comparison.
        """
        pass
    
    def _format_response(self):
        """
        Formats the response from the LLM into a structured format.
        """
        pass