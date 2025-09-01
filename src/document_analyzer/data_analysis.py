import os
import sys
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union
from collections import Counter
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DeepDocException
from model.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Import your custom prompts
from prompt.prompt_library import METADATA_PROMPT_REGISTRY


class DocumentAnalyzer:
    
    def __init__(self, 
                 chunk_size: int = 1200,  # Larger chunks for better context
                 chunk_overlap: int = 300,  # More overlap for metadata continuity
                 max_workers: int = 3,
                 use_context_aware_prompts: bool = True):
        
        self.log = CustomLogger().get_logger(__name__)
        self.use_context_aware = use_context_aware_prompts
        
        try:
            self.loader = ModelLoader()
            self.llm = self.loader.load_llm()
            self.max_workers = max_workers
            
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            
            self.chunk_prompt = METADATA_PROMPT_REGISTRY["chunk_analysis"]
            self.full_doc_prompt = METADATA_PROMPT_REGISTRY["full_document"]
            self.context_prompt = METADATA_PROMPT_REGISTRY["context_aware_chunk"]
            self.consolidation_prompt = METADATA_PROMPT_REGISTRY["summary_consolidation"]

            # Tokenizer setup
            self.encoding = tiktoken.get_encoding("cl100k_base")

            def tiktoken_len(text):
                return len(self.encoding.encode(text))

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=tiktoken_len
            )

            self.log.info("DocumentAnalyzer initialized successfully")
            
        except Exception as e:
            self.log.error(f"Error initializing DocumentAnalyzer: {e}")
            raise DeepDocException("Error in DocumentAnalyzer initialization", sys)
    
    def _should_use_chunking(self, document_text: str, threshold: int = 6000) -> bool:
        """
        Decide whether to use chunking based on document size.
        """
        token_count = len(self.encoding.encode(document_text))
        return token_count > threshold

    def _get_document_preview(self, document_text: str, preview_length: int = 500) -> str:
        """
        Generate a preview of the document for context-aware analysis.
        """
        words = document_text.split()
        if len(words) <= preview_length:
            return document_text
        
        # Take first and last portions for context
        first_part = " ".join(words[:preview_length//2])
        last_part = " ".join(words[-preview_length//2:])
        return f"DOCUMENT START: {first_part}\n\n[...document continues...]\n\nDOCUMENT END: {last_part}"

    def _analyze_chunk(self, chunk_data: tuple) -> dict:
        """
        Analyze a single chunk with context-aware prompting.
        """
        chunk_text, chunk_index, total_chunks, doc_preview = chunk_data
        
        try:
            # Choose appropriate prompt based on settings and chunk position
            if self.use_context_aware and total_chunks > 2:
                prompt = self.context_prompt
                chain = prompt | self.llm | self.fixing_parser
                response = chain.invoke({
                    "format_instructions": self.parser.get_format_instructions(),
                    "document_text": chunk_text,
                    "chunk_position": chunk_index,
                    "total_chunks": total_chunks,
                    "document_preview": doc_preview
                })
            else:
                prompt = self.chunk_prompt
                chain = prompt | self.llm | self.fixing_parser
                response = chain.invoke({
                    "format_instructions": self.parser.get_format_instructions(),
                    "document_text": chunk_text,
                    "chunk_context": f"This is chunk {chunk_index} of {total_chunks}. Extract metadata available in this section."
                })
            
            self.log.info(f"Chunk {chunk_index}/{total_chunks} analysis successful")
            return response
            
        except Exception as e:
            self.log.error(f"Chunk {chunk_index} analysis failed", error=str(e))
            return self._get_empty_metadata()

    def _analyze_full_document(self, document_text: str) -> dict:
        """
        Analyze document without chunking using full document prompt.
        """
        try:
            chain = self.full_doc_prompt | self.llm | self.fixing_parser
            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            })
            self.log.info("Full document analysis successful")
            return response
        except Exception as e:
            self.log.error("Full document analysis failed", error=str(e))
            raise DeepDocException("Document analysis failed") from e

    def _get_empty_metadata(self) -> dict:
        """
        Return empty metadata structure matching schema.
        """
        return {
            "Summary": [""],
            "Title": "Not Available",
            "Author": "Not Available",
            "DateCreated": "Not Available",
            "LastModifiedDate": "Not Available",
            "Publisher": "Not Available",
            "Language": "Not Available",
            "PageCount": "Not Available",
            "SentimentTone": "Neutral"
        }

    def analyze_document(self, document_text: str) -> dict:
        """
        Main analysis method with intelligent routing and processing.
        """
        if not document_text or not document_text.strip():
            raise DeepDocException("Empty or invalid document text provided")

        try:
            # Smart routing based on document size
            if not self._should_use_chunking(document_text):
                self.log.info("Using full document analysis (small document)")
                return self._analyze_full_document(document_text)

            return self._analyze_with_chunking(document_text)

        except Exception as e:
            self.log.error("Document analysis failed", error=str(e))
            raise DeepDocException("Document analysis failed") from e

    def _analyze_with_chunking(self, document_text: str) -> dict:
        """
        Analyze document using intelligent chunking strategy.
        """
        chunks = self.text_splitter.split_text(document_text)
        total_chunks = len(chunks)
        doc_preview = self._get_document_preview(document_text)
        
        self.log.info(f"Document split into {total_chunks} chunks")

        # Prepare chunk data with context
        chunk_data = [
            (chunk, idx, total_chunks, doc_preview) 
            for idx, chunk in enumerate(chunks, 1)
        ]
        
        # Process chunks (with parallel execution for efficiency)
        results = []
        if total_chunks <= 3 or self.max_workers == 1:
            # Sequential processing for small number of chunks
            for data in chunk_data:
                result = self._analyze_chunk(data)
                results.append((data[1], result))  # (chunk_index, result)
        else:
            # Parallel processing for larger documents
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_chunk = {
                    executor.submit(self._analyze_chunk, data): data[1] 
                    for data in chunk_data
                }
                
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        result = future.result()
                        results.append((chunk_idx, result))
                    except Exception as e:
                        self.log.error(f"Chunk {chunk_idx} processing failed: {e}")
                        results.append((chunk_idx, self._get_empty_metadata()))

        # Sort results by chunk index
        results.sort(key=lambda x: x[0])
        chunk_results = [result[1] for result in results]

        # Intelligent merging with document context
        final_metadata = self._intelligent_merge(chunk_results, document_text, doc_preview)
        self.log.info("Chunked document analysis completed successfully")
        
        return final_metadata

    def _intelligent_merge(self, results: List[dict], original_text: str, doc_preview: str) -> dict:
        """
        Advanced merging with document context awareness.
        """
        try:
            merged = self._collect_field_data(results)
            
            return {
                "Summary": self._merge_summaries(merged["summaries"]),
                "Title": self._select_best_title(merged["titles"]),
                "Author": self._select_best_author(merged["authors"]),
                "DateCreated": self._select_best_date(merged["dates_created"]),
                "LastModifiedDate": self._select_best_date(merged["dates_modified"]),
                "Publisher": self._select_most_common(merged["publishers"]),
                "Language": self._detect_language(merged["languages"], original_text),
                "PageCount": self._merge_page_counts(merged["page_counts"]),
                "SentimentTone": self._merge_sentiment_tones(merged["sentiment_tones"])
            }
            
        except Exception as e:
            self.log.error(f"Intelligent merge failed: {e}")
            return self._fallback_merge(results)

    def _collect_field_data(self, results: List[dict]) -> dict:
        """
        Collect all field values from chunk results for intelligent processing.
        """
        collected = {
            "summaries": [],
            "titles": [],
            "authors": [],
            "dates_created": [],
            "dates_modified": [],
            "publishers": [],
            "languages": [],
            "page_counts": [],
            "sentiment_tones": []
        }

        for result in results:
            # Handle Summary (List[str])
            if result.get("Summary"):
                if isinstance(result["Summary"], list):
                    collected["summaries"].extend([s.strip() for s in result["Summary"] if s.strip()])
                else:
                    summary_str = str(result["Summary"]).strip()
                    if summary_str:
                        collected["summaries"].append(summary_str)
            
            # Collect other fields if they have meaningful values
            fields_map = {
                "titles": "Title",
                "authors": "Author", 
                "dates_created": "DateCreated",
                "dates_modified": "LastModifiedDate",
                "publishers": "Publisher",
                "languages": "Language",
                "page_counts": "PageCount",
                "sentiment_tones": "SentimentTone"
            }
            
            for collected_key, result_key in fields_map.items():
                value = result.get(result_key)
                if value and value != "Not Available" and value != "Neutral":
                    collected[collected_key].append(value)

        return collected

    def _merge_summaries(self, summaries: List[str]) -> List[str]:
        """
        Create consolidated summary using LLM-based consolidation.
        """
        if not summaries:
            return ["Document summary not available"]
        
        if len(summaries) == 1:
            return summaries
        
        try:
            # Use specialized consolidation prompt
            consolidated_summary = self.llm.invoke(
                self.consolidation_prompt.format(
                    summaries="\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))
                )
            )
            
            content = consolidated_summary.content if hasattr(consolidated_summary, 'content') else str(consolidated_summary)
            return [content.strip()] if content.strip() else summaries[:2]
            
        except Exception as e:
            self.log.error(f"Summary consolidation failed: {e}")
            return summaries[:3]  # Return top 3 as fallback

    def _select_best_title(self, titles: List[str]) -> str:
        """
        Select most appropriate title using heuristics.
        """
        if not titles:
            return "Not Available"
        
        if len(titles) == 1:
            return titles[0]
        
        # Prefer titles that are:
        # 1. Not too short (more than 2 words)
        # 2. Not too long (less than 15 words)
        # 3. Don't contain common non-title words
        
        scored_titles = []
        for title in titles:
            score = 0
            words = title.split()
            
            # Length scoring
            if 3 <= len(words) <= 12:
                score += 3
            elif len(words) > 12:
                score -= 1
            
            # Content scoring
            title_lower = title.lower()
            if any(word in title_lower for word in ['summary', 'conclusion', 'page', 'chapter']):
                score -= 2
            if any(word in title_lower for word in ['title', 'document', 'report', 'analysis']):
                score += 1
            
            # Prefer titles with capitalization
            if title != title.upper() and title != title.lower():
                score += 1
                
            scored_titles.append((title, score))
        
        # Return highest scoring title
        best_title = max(scored_titles, key=lambda x: x[1])
        return best_title[0]

    def _select_best_author(self, authors: List[str]) -> str:
        """
        Select best author with name validation.
        """
        if not authors:
            return "Not Available"
        
        # Remove duplicates while preserving case
        unique_authors = []
        seen_lower = set()
        for author in authors:
            author_lower = author.lower().strip()
            if author_lower not in seen_lower and author_lower:
                seen_lower.add(author_lower)
                unique_authors.append(author.strip())
        
        if len(unique_authors) == 1:
            return unique_authors[0]
        
        # Prefer authors that look like real names (2-4 words, proper capitalization)
        for author in unique_authors:
            words = author.split()
            if 2 <= len(words) <= 4 and all(word[0].isupper() for word in words if word):
                return author
        
        return unique_authors[0]  # Fallback to first

    def _select_best_date(self, dates: List[str]) -> str:
        """
        Select most recent or most complete date.
        """
        if not dates:
            return "Not Available"
        
        # Prefer dates that look more complete/formatted
        date_scores = []
        for date in dates:
            score = 0
            if len(date) > 8:  # Longer dates usually more complete
                score += 2
            if any(sep in date for sep in ['-', '/', ' ']):  # Has separators
                score += 1
            if any(month in date.lower() for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec']):
                score += 2
            date_scores.append((date, score))
        
        best_date = max(date_scores, key=lambda x: x[1])
        return best_date[0]

    def _select_most_common(self, values: List[str]) -> str:
        """
        Return most frequently occurring value.
        """
        if not values:
            return "Not Available"
        
        counter = Counter(values)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else values[0]

    def _detect_language(self, languages: List[str], original_text: str) -> str:
        """
        Detect language with fallback to text analysis.
        """
        if languages:
            return self._select_most_common(languages)
        
        # Simple heuristic language detection
        english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
        text_sample = original_text[:1000].lower()
        english_count = sum(1 for word in english_words if f' {word} ' in text_sample)
        
        return "English" if english_count >= 3 else "Not Available"

    def _merge_page_counts(self, page_counts: List[Union[int, str]]) -> Union[int, str]:
        """
        Intelligently merge page count information.
        """
        if not page_counts:
            return "Not Available"
        
        # Try to find numeric values
        numeric_counts = []
        for count in page_counts:
            if isinstance(count, int):
                numeric_counts.append(count)
            elif isinstance(count, str):
                try:
                    # Extract numbers from strings like "Page 5 of 10"
                    numbers = [int(s) for s in count.split() if s.isdigit()]
                    if numbers:
                        numeric_counts.append(max(numbers))  # Take largest number
                except:
                    continue
        
        if numeric_counts:
            return max(numeric_counts)  # Return highest page count
        
        # Return first non-"Not Available" string
        for count in page_counts:
            if str(count) != "Not Available":
                return count
        
        return "Not Available"

    def _merge_sentiment_tones(self, sentiment_tones: List[str]) -> str:
        """
        Merge sentiment tones using weighted approach.
        """
        if not sentiment_tones:
            return "Neutral"
        
        # Count sentiments and apply weighting
        sentiment_weights = {
            "Professional": 2,    # More weight to professional tone
            "Academic": 2,        # More weight to academic tone  
            "Positive": 1,
            "Negative": 1,
            "Mixed": 1,
            "Informal": 1,
            "Neutral": 0.5       # Less weight to neutral
        }
        
        weighted_counts = {}
        for tone in sentiment_tones:
            weight = sentiment_weights.get(tone, 1)
            weighted_counts[tone] = weighted_counts.get(tone, 0) + weight
        
        if weighted_counts:
            best_tone = max(weighted_counts.items(), key=lambda x: x[1])
            return best_tone[0]
        
        return "Neutral"

    def _fallback_merge(self, results: List[dict]) -> dict:
        """
        Simple fallback merge when intelligent merging fails.
        """
        merged = self._get_empty_metadata()
        
        for result in results:
            for field in ["Title", "Author", "DateCreated", "LastModifiedDate", 
                         "Publisher", "Language", "PageCount", "SentimentTone"]:
                if (merged[field] == "Not Available" and 
                    result.get(field) and 
                    result[field] != "Not Available"):
                    merged[field] = result[field]
            
            # Handle Summary list
            if result.get("Summary"):
                if isinstance(result["Summary"], list):
                    merged["Summary"].extend(result["Summary"])
                else:
                    merged["Summary"].append(str(result["Summary"]))
        
        if not any(s.strip() for s in merged["Summary"]):
            merged["Summary"] = ["Document summary not available"]
        
        return merged