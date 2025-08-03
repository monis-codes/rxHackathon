import os
import logging
from io import BytesIO
import hashlib
from typing import List, Dict, Any

import google.generativeai as genai
# from pinecone.data.index import Index
from pinecone import Pinecone
from pinecone import PineconeException
from pypdf import PdfReader
from docx import Document  
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pinecone import Index

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def detect_file_type(content: bytes) -> str:
    """
    Detect file type based on content headers.
    """
    if content.startswith(b'%PDF'):
        return 'pdf'
    elif content.startswith(b'PK') and b'word/' in content[:1000]:
        return 'docx'
    else:
        raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")

def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF content."""
    try:
        reader = PdfReader(BytesIO(content))
        if len(reader.pages) == 0:
            raise ValueError("PDF contains no pages")
        
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                full_text += page_text
                logger.debug(f"Extracted text from PDF page {page_num + 1}")
            except Exception as e:
                logger.warning(f"Failed to extract text from PDF page {page_num + 1}: {e}")
                continue
        
        return full_text
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {e}")

def extract_text_from_docx(content: bytes) -> str:
    """Extract text from DOCX content."""
    try:
        doc = Document(BytesIO(content))
        full_text = ""
        
        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text += paragraph.text + "\n"
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        full_text += cell.text + " "
                full_text += "\n"
        
        logger.info(f"Extracted text from DOCX: {len(doc.paragraphs)} paragraphs, {len(doc.tables)} tables")
        return full_text
    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX: {e}")

def extract_text_from_document(content: bytes, file_type: str = None) -> str:
    """
    Extract text from document based on file type.
    """
    if not content:
        raise ValueError("Document content cannot be empty")
    
    # Auto-detect file type if not provided
    if not file_type:
        file_type = detect_file_type(content)
    
    logger.info(f"Processing {file_type.upper()} document")
    
    if file_type == 'pdf':
        text = extract_text_from_pdf(content)
    elif file_type == 'docx':
        text = extract_text_from_docx(content)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    if not text.strip():
        raise ValueError(f"No text could be extracted from the {file_type.upper()} document")
    
    return text

def smart_chunk_text(text: str, max_chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """
    Intelligently chunk text by sentences to avoid breaking mid-sentence.
    """
    if not text or not text.strip():
        return []
    
    # Split by sentences (basic approach)
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous chunk
                words = current_chunk.split()
                overlap_text = ' '.join(words[-overlap//10:]) if len(words) > overlap//10 else ""
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                # Single sentence is too long, force split
                chunks.append(sentence[:max_chunk_size])
                current_chunk = sentence[max_chunk_size:]
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter tiny chunks

def process_document(index: Any, document_content: bytes, namespace: str, file_type: str = None):
    """
    Processes a document (PDF or DOCX) by extracting text, chunking, embedding, and upserting into Pinecone.
    """
    # Input validation
    if not index:
        raise ValueError("Index cannot be None")
    if not document_content:
        raise ValueError("Document content cannot be empty")
    if not namespace or not namespace.strip():
        raise ValueError("Namespace cannot be empty")
    
    try:
        # 1. Extract Text from Document
        logger.info(f"Starting document processing for namespace: {namespace}")
        
        full_text = extract_text_from_document(document_content, file_type)
        
        # Clean the text
        full_text = full_text.replace('\n', ' ').replace('\r', ' ')
        full_text = ' '.join(full_text.split())  # Remove extra whitespace
        
        logger.info(f"Extracted {len(full_text)} characters from document")
        
        # 2. Chunk the Text
        text_chunks = smart_chunk_text(full_text, max_chunk_size=1500)
        
        if not text_chunks:
            raise ValueError("Failed to create any text chunks from the document")
        
        logger.info(f"Created {len(text_chunks)} text chunks")
        
        # 3. Embed Chunks
        embedding_model = "models/text-embedding-004"
        
        try:
            embedding_result = genai.embed_content(
                model=embedding_model,
                content=text_chunks,
                task_type="RETRIEVAL_DOCUMENT",
                title="Document Chunks"
            )
            
            if not embedding_result or 'embedding' not in embedding_result:
                raise ValueError("Failed to generate embeddings")
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise ValueError(f"Failed to generate embeddings: {e}")
        
        # 4. Prepare vectors for upsert
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(text_chunks, embedding_result['embedding'])):
            # Create unique ID using content hash + index
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
            vector_id = f"{namespace}_{chunk_hash}_{i}"
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": chunk,
                    "chunk_index": i,
                    "namespace": namespace,
                    "char_count": len(chunk),
                    "file_type": file_type or "unknown"
                }
            })
        
        # 5. Upsert to Pinecone in batches
        batch_size = 50
        successful_upserts = 0
        
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            try:
                index.upsert(vectors=batch, namespace=namespace)
                successful_upserts += len(batch)
                logger.info(f"Upserted batch {i//batch_size + 1}, vectors: {len(batch)}")
            except PineconeException as e:
                logger.error(f"Failed to upsert batch {i//batch_size + 1}: {e}")
                raise
        
        logger.info(f"Successfully processed and upserted {successful_upserts} chunks into namespace '{namespace}'")
        
    except ValueError:
        # Re-raise validation errors as-is
        raise
    except PineconeException as e:
        logger.error(f"Pinecone error during document processing: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during document processing: {e}")
        raise ValueError(f"Document processing failed: {e}")

def namespace_exists(index: Any, namespace: str) -> bool:
    """
    Checks if a namespace exists in the Pinecone index by querying its stats.
    """
    if not index or not namespace:
        raise ValueError("Index and namespace cannot be None or empty")
    
    try:
        stats = index.describe_index_stats()
        exists = namespace in stats.get('namespaces', {})
        logger.info(f"Namespace '{namespace}' exists: {exists}")
        return exists
    except PineconeException as e:
        logger.error(f"Pinecone API error checking namespace: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error checking namespace: {e}")
        raise

def get_answer_for_question(index: Any, question: str, namespace: str) -> str:
    """
    Main function - matches your existing API call with 3 parameters.
    This is what your main.py calls.
    """
    return get_answer_for_question_enhanced(
        index=index,
        question=question,
        namespace=namespace,
        top_k=5,
        similarity_threshold=0.5,  # More permissive for backward compatibility
        max_context_length=4000
    )

def get_answer_for_question_enhanced(
    index: Any, 
    question: str, 
    namespace: str,
    top_k: int = 5,
    similarity_threshold: float = 0.5,
    max_context_length: int = 4000
) -> str:
    """
    Embeds a question, queries Pinecone for relevant context, and generates an answer.
    
    Args:
        index: Pinecone index instance
        question: User's question
        namespace: Pinecone namespace to query
        top_k: Number of chunks to retrieve (default: 5)
        similarity_threshold: Minimum similarity score for relevance (default: 0.7)
        max_context_length: Maximum context length in characters (default: 4000)
    
    Returns:
        str: Generated answer based on retrieved context
        
    Raises:
        ValueError: If inputs are invalid
        PineconeException: If Pinecone query fails
        Exception: If embedding or generation fails
    """
    
    # Enhanced input validation
    if not index:
        raise ValueError("Index cannot be None")
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    if not namespace or not namespace.strip():
        raise ValueError("Namespace cannot be empty")
    if top_k <= 0 or top_k > 100:
        raise ValueError("top_k must be between 1 and 100")
    if similarity_threshold < 0 or similarity_threshold > 1:
        raise ValueError("similarity_threshold must be between 0 and 1")
    
    question = question.strip()
    logger.info(f"Processing question: '{question[:50]}...' in namespace: {namespace}")
    
    try:
        # 1. Embed the user's question
        embedding_model = "models/text-embedding-004"
        
        try:
            question_embedding_result = genai.embed_content(
                model=embedding_model,
                content=question,
                task_type="RETRIEVAL_QUERY"
            )
            
            if not question_embedding_result or 'embedding' not in question_embedding_result:
                raise ValueError("Failed to generate question embedding")
                
            question_embedding = question_embedding_result['embedding']
            logger.debug("Successfully generated question embedding")
            
        except Exception as e:
            logger.error(f"Failed to embed question: {e}")
            raise ValueError(f"Question embedding failed: {e}")
        
        # 2. Query Pinecone for relevant text chunks
        try:
            query_results = index.query(
                namespace=namespace,
                vector=question_embedding,
                top_k=top_k,
                include_metadata=True,
                include_values=False  # We don't need the vectors back
            )
            
            if not query_results or 'matches' not in query_results:
                logger.warning("No results returned from Pinecone query")
                return "I could not find any relevant information in the provided document."
                
            matches = query_results.get('matches', [])
            logger.info(f"Retrieved {len(matches)} chunks from Pinecone")
            
        except PineconeException as e:
            logger.error(f"Pinecone query failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Pinecone query: {e}")
            raise PineconeException(f"Query failed: {e}")
        
        # 3. Filter and construct context based on relevance
        relevant_chunks = []
        for match in matches:
            score = match.get('score', 0)
            metadata = match.get('metadata', {})
            text = metadata.get('text', '').strip()
            
            if score >= similarity_threshold and text:
                relevant_chunks.append({
                    'text': text,
                    'score': score,
                    'chunk_index': metadata.get('chunk_index', 'unknown')
                })
                logger.debug(f"Added relevant chunk (score: {score:.3f})")
        
        if not relevant_chunks:
            logger.warning(f"No chunks met similarity threshold of {similarity_threshold}")
            return "I could not find sufficiently relevant information in the provided document to answer your question."
        
        # Build context with length management
        context_parts = []
        current_length = 0
        
        # Sort by relevance score (highest first)
        relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        for chunk in relevant_chunks:
            chunk_text = chunk['text']
            # Add some separator and metadata
            formatted_chunk = f"[Relevance: {chunk['score']:.2f}] {chunk_text}"
            
            if current_length + len(formatted_chunk) <= max_context_length:
                context_parts.append(formatted_chunk)
                current_length += len(formatted_chunk)
            else:
                # Add partial chunk if there's room
                remaining_space = max_context_length - current_length - 50  # Leave buffer
                if remaining_space > 100:  # Only add if meaningful amount of text fits
                    context_parts.append(formatted_chunk[:remaining_space] + "...")
                break
        
        if not context_parts:
            return "The relevant information found was too lengthy to process. Please try a more specific question."
        
        context = "\n\n".join(context_parts)
        logger.info(f"Built context with {len(context_parts)} chunks, {len(context)} characters")
        
        # 4. Generate the final answer
        prompt_template = f"""You are a helpful AI assistant. Answer the following question based ONLY on the context provided below. 

IMPORTANT INSTRUCTIONS:
- Synthesize the information from the context into a concise and clear answer.
- Do not quote the context verbatim. Summarize the key points.
- If the answer is not found in the context, respond with "I could not find the answer in the provided document."
- If multiple relevant pieces of information exist, synthesize them coherently
- Do not make assumptions or add information not present in the context

Context:
{context}

Question: {question}

Answer:"""

        try:
            generative_model = genai.GenerativeModel('gemini-1.5-flash')
            response = generative_model.generate_content(
                prompt_template,
                generation_config={
                    'temperature': 0.1,  # Low temperature for more factual responses
                    'max_output_tokens': 2048,
                }
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from generative model")
            
            answer = response.text.strip()
            logger.info("Successfully generated answer")
            
            # Basic quality check
            if len(answer) < 10:
                logger.warning("Generated answer seems too short")
            
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise ValueError(f"Answer generation failed: {e}")
    
    except ValueError:
        # Re-raise validation errors as-is
        raise
    except PineconeException:
        # Re-raise Pinecone errors as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_answer_for_question: {e}")
        raise ValueError(f"Question processing failed: {e}")
