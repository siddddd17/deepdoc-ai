import sys
from dotenv import load_dotenv
import pandas as pd
from logger.custom_logger import CustomLogger
from exception.custom_exception_archive import DeepDocException
from model.models import *
from prompt.prompt_library import PROMPT_REGISTRY
from utils.model_loader import ModelLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from logger import GLOBAL_LOGGER as log

class DocumentComparatorLLM:
    def __init__(self):
        load_dotenv()
        self.loader = ModelLoader()
        self.llm = self.loader.load_llm()
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.prompt = PROMPT_REGISTRY["document_comparison"]
        self.chain = self.prompt | self.llm | self.parser  
        log.info("DocumentComparatorLLM initialized with model and parser.")

    def compare_documents(self,combined_docs: str) -> pd.DataFrame:
        """
        Compares two documents and returns a structured comparison.
        """
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instruction": self.parser.get_format_instructions()
            }
            log.info("Starting document comparison", inputs=inputs)
            response = self.chain.invoke(inputs)
            log.info("Chain invoked successfully", response_preview=str(response)[:200])
            return self._format_response(response)
        except Exception as e:
            log.error(f"Error in compare_documents: {e}")
            raise DeepDocException("An error occurred while comparing documents.",sys)
    
    def _format_response(self,response_parsed: list[dict]) -> pd.DataFrame: #type: ignore
        """
        Formats the response from the LLM into a structured format.
        """
        try:
            df = pd.DataFrame(response_parsed)
            log.info("Response formatted into DataFrame", dataframe=df)
            return df
        except Exception as e:
            log.error("Error formatting response into DataFrame", error=str(e))
            raise DeepDocException("Error formatting response",sys)