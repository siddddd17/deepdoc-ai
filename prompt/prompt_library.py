# Prepare prompt template
from langchain_core.prompts import ChatPromptTemplate

document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant trained to analyze and summarize documents.
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")

document_comparison_prompt= ChatPromptTemplate.from_template("""
You will be provided with content from two PDFs. Your tasks are as follows:

1. Compare the content in two PDFs
2. Identify the difference in PDF and note down the page number 
3. The output you provide must be page wise comparison content 
4. If any page do not have any change, mention as 'NO CHANGE' 

Input documents:

{combined_docs}

Your response should follow this format:

{format_instruction}
""")

CHUNK_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["format_instructions", "document_text", "chunk_context"],
    template="""You are an expert document metadata extractor. Your task is to analyze a document chunk and extract structured metadata.

{chunk_context}

IMPORTANT INSTRUCTIONS:
- Extract only information that you can confidently determine from this text chunk
- If information is not available in this chunk, use "Not Available" for string fields
- For Summary, provide a concise summary of what you can observe in this chunk
- For SentimentTone, analyze the overall emotional tone of the content
- Be conservative - it's better to say "Not Available" than to guess

METADATA EXTRACTION RULES:
- Title: Look for document titles, headers, or clear document names. Prefer longer, descriptive titles.
- Author: Extract author names, bylines, or attribution. May appear as "By [Name]" or "Author: [Name]"
- DateCreated: Look for creation dates, publication dates, "Created:", "Published:", etc.
- LastModifiedDate: Look for modification dates, "Last updated:", "Revised:", etc.
- Publisher: Extract publisher names, organization names, journal names, or publication sources
- Language: Detect the primary language of the text content
- PageCount: Look for page indicators, "Page X of Y", total page references, or document length indicators
- SentimentTone: Analyze emotional tone - options: "Positive", "Negative", "Neutral", "Mixed", "Professional", "Academic", "Informal"
- Summary: Provide a concise summary of the main topics/content in this chunk (as a list with one or few items)

TEXT CHUNK TO ANALYZE:
{document_text}

{format_instructions}

Extract the metadata in the exact JSON format specified above:"""
)

# Full document analysis prompt (for smaller documents)
FULL_DOCUMENT_PROMPT = PromptTemplate(
    input_variables=["format_instructions", "document_text"],
    template="""You are an expert document metadata extractor. Analyze the entire document and extract comprehensive structured metadata.

METADATA EXTRACTION GUIDELINES:
- Title: Extract the main document title. Look for headers, title pages, or prominent headings
- Author: Find author names from bylines, author sections, or attribution
- DateCreated: Look for creation, publication, or original dates
- LastModifiedDate: Find last modified, updated, or revision dates
- Publisher: Extract publisher, organization, journal, or source information
- Language: Identify the primary language of the document
- PageCount: Determine total pages or provide available page information
- SentimentTone: Analyze overall emotional tone: "Positive", "Negative", "Neutral", "Mixed", "Professional", "Academic", "Informal"
- Summary: Create a comprehensive summary covering the main topics and key points

DOCUMENT TEXT:
{document_text}

{format_instructions}

Provide the metadata in the exact JSON format specified:"""
)

# Summary consolidation prompt (for merging chunk summaries)
SUMMARY_CONSOLIDATION_PROMPT = PromptTemplate(
    input_variables=["summaries"],
    template="""You are tasked with creating a single, coherent document summary from multiple section summaries.

Your goal is to:
1. Combine information from all sections without redundancy
2. Create a flowing, comprehensive summary
3. Maintain the key points from each section
4. Keep it concise but complete
5. Structure it as a single coherent narrative

SECTION SUMMARIES TO CONSOLIDATE:
{summaries}

Create a consolidated summary that captures the essence of the entire document:"""
)

# Advanced chunk analysis with context awareness
CONTEXT_AWARE_CHUNK_PROMPT = PromptTemplate(
    input_variables=["format_instructions", "document_text", "chunk_position", "total_chunks", "document_preview"],
    template="""You are analyzing chunk {chunk_position} of {total_chunks} from a document.

DOCUMENT CONTEXT:
{document_preview}

CHUNK ANALYSIS STRATEGY:
- Early chunks (1-2): Focus on titles, authors, publication info, document headers
- Middle chunks: Focus on content summary, main topics, sentiment analysis
- Final chunks: Look for publisher info, dates, page counts, document footers

CURRENT CHUNK POSITION: {chunk_position}/{total_chunks}

EXTRACTION PRIORITIES FOR THIS POSITION:
{{% if chunk_position <= 2 %}}
PRIMARY: Title, Author, DateCreated, Publisher, Language
SECONDARY: Summary, SentimentTone
{{% elif chunk_position >= total_chunks - 1 %}}
PRIMARY: Publisher, LastModifiedDate, PageCount
SECONDARY: Summary, SentimentTone
{{% else %}}
PRIMARY: Summary, SentimentTone, Language
SECONDARY: Other available metadata
{{% endif %}}

CHUNK TEXT:
{document_text}

{format_instructions}

Extract metadata with focus on the priorities for this chunk position:"""
)

# Quality validation prompt for final metadata
METADATA_VALIDATION_PROMPT = PromptTemplate(
    input_variables=["metadata", "document_preview"],
    template="""Review and validate the extracted metadata for quality and consistency.

EXTRACTED METADATA:
{metadata}

DOCUMENT PREVIEW:
{document_preview}

VALIDATION CHECKS:
1. Does the title accurately represent the document?
2. Are the dates in reasonable format and logical?
3. Is the summary comprehensive yet concise?
4. Does the sentiment tone match the document content?
5. Are there any obvious errors or inconsistencies?

If you find issues, suggest corrections. Otherwise, confirm the metadata is accurate.

Provide validation results:"""
)

# Enhanced DocumentAnalyzer integration
def get_chunk_prompt_by_position(chunk_position: int, total_chunks: int, use_context_aware: bool = False):
    """
    Select the most appropriate prompt based on chunk position
    """
    if use_context_aware and total_chunks > 3:
        return METADATA_PROMPT_REGISTRY["context_aware_chunk"]
    else:
        return METADATA_PROMPT_REGISTRY["chunk_analysis"]
    

METADATA_PROMPT_REGISTRY = {
    "chunk_analysis": CHUNK_ANALYSIS_PROMPT,
    "full_document": FULL_DOCUMENT_PROMPT,
    "summary_consolidation": SUMMARY_CONSOLIDATION_PROMPT,
    "context_aware_chunk": CONTEXT_AWARE_CHUNK_PROMPT,
    "metadata_validation": METADATA_VALIDATION_PROMPT,
}

PROMPT_REGISTRY={
    "document_analysis":document_analysis_prompt,
    "document_comparison":document_comparison_prompt}

