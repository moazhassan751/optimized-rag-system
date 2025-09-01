# rag_core.py - Production-Grade Universal RAG System
# Fixed Pinecone type issues and enhanced error handling

import asyncio
import time
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from chunking import ProductionChunker
from config import Config
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache
import re
import os
from collections import defaultdict
import hashlib
import logging
from typing import List, Dict, Optional, Any, Union
from dotenv import load_dotenv
import gc
import json

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_system.log')
    ]
)

def safe_convert_types(obj: Any) -> Any:
    """Convert NumPy and other non-serializable types to Python native types for Pinecone."""
    if obj is None:
        return ""  # Convert None to empty string for Pinecone
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        # Filter out None values and convert the rest
        return {k: safe_convert_types(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [safe_convert_types(item) for item in obj if item is not None]
    elif isinstance(obj, tuple):
        return tuple(safe_convert_types(item) for item in obj if item is not None)
    else:
        return obj

class ProductionRAGSystem:
    """Enterprise-grade RAG system with comprehensive error handling and monitoring."""
    
    def __init__(self):
        """Initialize with comprehensive error handling and monitoring."""
        self.start_time = time.time()
        load_dotenv()
        
        # Performance metrics
        self.metrics = {
            'initialization_time': 0,
            'documents_processed': 0,
            'queries_processed': 0,
            'avg_response_time': 0,
            'total_response_time': 0,
            'cache_hits': 0,
            'errors': 0,
            'pinecone_operations': 0,
            'successful_embeddings': 0
        }
        
        # Separate performance metrics for tests
        self.performance_metrics = {
            'total_queries': 0,
            'high_confidence_queries': 0,
            'multi_source_queries': 0,
            'avg_response_time': 0.0
        }
        
        try:
            self._initialize_components()
            self.metrics['initialization_time'] = time.time() - self.start_time
            logging.info(f"RAG system initialized in {self.metrics['initialization_time']:.2f}s")
        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            raise

    def _initialize_components(self):
        """Initialize all system components with proper error handling."""
        
        # 1. Initialize embeddings (most critical)
        logging.info("Initializing embeddings...")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={
                    'device': 'cpu',
                    'trust_remote_code': False
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 16  # Reduced for stability
                }
            )
            logging.info("Embeddings initialized successfully")
        except Exception as e:
            logging.error(f"Embeddings initialization failed: {e}")
            raise

        # 2. Initialize chunker
        logging.info("Initializing chunker...")
        self.chunker = ProductionChunker()
        logging.info("Chunker initialized")

        # 3. Initialize LLM with multiple fallbacks (NO MOCKS)
        logging.info("Initializing LLM...")
        self.llm = self._initialize_production_llm()
        if self.llm is None:
            raise RuntimeError("No valid LLM could be initialized. Please configure Google API key in .env file.")

        # 4. Initialize Pinecone
        logging.info("Initializing Pinecone...")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if self._is_valid_api_key(pinecone_api_key, "PINECONE"):
            try:
                self.pc = Pinecone(api_key=pinecone_api_key)
                # Test connection
                self.pc.list_indexes()
                self.use_real_pinecone = True
                logging.info("Pinecone initialized successfully")
            except Exception as e:
                logging.error(f"Pinecone initialization failed: {e}")
                self.pc = None
                self.use_real_pinecone = False
        else:
            logging.warning("No valid Pinecone API key - vector search disabled")
            self.pc = None
            self.use_real_pinecone = False

        # 5. Initialize reranker with timeout (cross-platform)
        logging.info("Initializing reranker...")
        try:
            def load_reranker():
                return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(load_reranker)
                self.reranker = future.result(timeout=30)
            logging.info("Reranker initialized successfully")
        except Exception as e:
            logging.error(f"Reranker initialization failed: {e}")
            self.reranker = None

        # Initialize caches and storage
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.document_chunks = []
        self._query_cache = {}
        self._embedding_cache = {}

    def _initialize_production_llm(self):
        """Initialize Google Gemini LLM only (no fallbacks)"""
        
        # Use only Google Gemini
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if self._is_valid_api_key(google_api_key, "GOOGLE"):
            try:
                logging.info("Initializing Google Gemini (only option)...")
                from langchain_google_genai import ChatGoogleGenerativeAI
                
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",  # Fast and efficient
                    google_api_key=google_api_key,
                    temperature=0.1,
                    max_tokens=512,
                    timeout=30
                )
                
                # Test the LLM
                test_response = llm.invoke("Hello")
                logging.info(f"Google Gemini initialized successfully: {test_response.content[:50]}...")
                return llm
                
            except Exception as e:
                logging.error(f"Google Gemini failed: {e}")
                # Provide helpful error message based on the specific error
                if "API_KEY_INVALID" in str(e):
                    raise RuntimeError("❌ Invalid Google API key. Please get a valid key from https://makersuite.google.com/app/apikey")
                elif "quota" in str(e).lower():
                    raise RuntimeError("❌ Google API quota exceeded. Please check your usage limits.")
                else:
                    raise RuntimeError(f"❌ Google Gemini initialization failed: {e}")
        else:
            logging.error("No valid Google API key found")
            raise RuntimeError("❌ Google API key is required. Please set GOOGLE_API_KEY in your .env file. Get it from: https://makersuite.google.com/app/apikey")

    def _is_valid_api_key(self, api_key: str, service: str) -> bool:
        """Enhanced API key validation"""
        if not api_key:
            return False
        if len(api_key) < 10:
            return False
        if api_key.startswith(("YOUR_", "mock_", "test_", "fake_", "demo_")):
            return False
        
        # Service-specific validation
        if service == "GOOGLE" and not api_key.startswith("AI"):
            return False
        if service == "OPENAI" and not api_key.startswith("sk-"):
            return False
            
        return True

    @lru_cache(maxsize=5000)
    def _preprocess_query(self, query: str) -> str:
        """Cached query preprocessing."""
        # Clean and normalize
        processed = re.sub(r'\s+', ' ', query.strip().lower())
        
        # Expand common abbreviations
        expansions = {
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'dl': 'deep learning'
        }
        
        for abbr, expansion in expansions.items():
            processed = processed.replace(f' {abbr} ', f' {expansion} ')
        
        return processed

    async def load_document_async(self, file_path: str) -> List[Document]:
        """Load document with comprehensive error handling and progress tracking."""
        logging.info(f"Loading document: {file_path}")
        load_start = time.time()
        
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size = os.path.getsize(file_path)
            if file_size > 100 * 1024 * 1024:  # 100MB limit
                raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB")
            
            # Choose appropriate loader
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.lower().endswith(('.txt', '.md')):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            # Load with timeout and progress tracking
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = loop.run_in_executor(executor, loader.load)
                pages = await asyncio.wait_for(future, timeout=120.0)  # 2 minute timeout
            
            load_time = time.time() - load_start
            logging.info(f"Document loaded: {len(pages)} pages in {load_time:.2f}s")
            self.metrics['documents_processed'] += 1
            
            return pages
            
        except asyncio.TimeoutError:
            error_msg = f"Document loading timed out after 120s: {file_path}"
            logging.error(error_msg)
            self.metrics['errors'] += 1
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Failed to load document {file_path}: {str(e)}"
            logging.error(error_msg)
            self.metrics['errors'] += 1
            raise

    def create_optimized_chunks(self, pages: List[Document], file_path: str) -> List[Document]:
        """Create chunks with performance monitoring and memory management."""
        logging.info("Creating optimized chunks...")
        chunk_start = time.time()
        
        try:
            # Combine pages into single text
            text = '\n\n'.join(page.page_content for page in pages)
            text_length = len(text)
            
            logging.info(f"Processing text of {text_length:,} characters")
            
            # Memory check
            if text_length > 5_000_000:  # 5MB text limit
                logging.warning(f"Large document ({text_length:,} chars), may be slow")
            
            # Create chunks using production chunker
            metadata = {
                'source': file_path,
                'document': os.path.basename(file_path),
                'pages': len(pages),
                'text_length': text_length
            }
            
            chunks = self.chunker.create_chunks(text, metadata)
            
            # CRITICAL FIX: Convert all metadata values to Python native types
            for chunk in chunks:
                chunk.metadata = safe_convert_types(chunk.metadata)
            
            chunk_time = time.time() - chunk_start
            logging.info(f"Created {len(chunks)} chunks in {chunk_time:.2f}s")
            
            # Force garbage collection for large documents
            if text_length > 1_000_000:
                gc.collect()
            
            return chunks
            
        except Exception as e:
            logging.error(f"Chunking failed: {e}")
            self.metrics['errors'] += 1
            raise

    def create_vectorstore(self, chunks: List[Document], index_name: str = None) -> PineconeVectorStore:
        """Create vectorstore with comprehensive monitoring and fallback handling."""
        if index_name is None:
            index_name = Config.DEFAULT_INDEX_NAME
            
        logging.info(f"Creating vectorstore with {len(chunks)} chunks...")
        vectorstore_start = time.time()
        
        try:
            # Store chunks for TF-IDF
            self.document_chunks = chunks
            
            # Build TF-IDF matrix for hybrid search
            logging.info("Building TF-IDF matrix...")
            tfidf_start = time.time()
            
            documents_text = [chunk.page_content for chunk in chunks]
            
            # Dynamic TF-IDF configuration based on document count
            if len(documents_text) < 5:
                # Small document set - use minimal filtering
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=min(5000, len(documents_text) * 1000),
                    stop_words=None,  # Don't filter stop words for small sets
                    ngram_range=(1, 1),  # Only unigrams for small sets
                    min_df=1,  # Keep all terms
                    max_df=1.0,  # Keep all terms
                    sublinear_tf=True,
                    norm='l2'
                )
            else:
                # Standard configuration for larger document sets
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=15000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                    norm='l2'
                )
            
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents_text)
                tfidf_time = time.time() - tfidf_start
                logging.info(f"TF-IDF matrix built: {self.tfidf_matrix.shape} in {tfidf_time:.2f}s")
            except ValueError as e:
                # Fallback for edge cases
                logging.warning(f"TF-IDF failed ({e}), using basic vectorizer")
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words=None,
                    ngram_range=(1, 1),
                    min_df=1,
                    max_df=1.0
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents_text)
                tfidf_time = time.time() - tfidf_start
                logging.info(f"Fallback TF-IDF matrix built: {self.tfidf_matrix.shape} in {tfidf_time:.2f}s")
            
            # Handle Pinecone operations
            pinecone_start = time.time()
            
            if not self.use_real_pinecone:
                logging.info("Using mock vectorstore")
                vectorstore = self._create_mock_vectorstore(chunks)
            else:
                # Real Pinecone operations with enhanced error handling
                vectorstore = self._create_real_vectorstore(chunks, index_name)
            
            pinecone_time = time.time() - pinecone_start
            total_time = time.time() - vectorstore_start
            
            logging.info(f"Vectorstore created in {total_time:.2f}s "
                        f"(TF-IDF: {tfidf_time:.1f}s, Vector: {pinecone_time:.1f}s)")
            
            self.metrics['pinecone_operations'] += 1
            return vectorstore
            
        except Exception as e:
            logging.error(f"Vectorstore creation failed: {e}")
            self.metrics['errors'] += 1
            raise

    def _create_real_vectorstore(self, chunks: List[Document], index_name: str) -> PineconeVectorStore:
        """Create real Pinecone vectorstore with enhanced error handling and type safety."""
        try:
            # Check existing indexes
            existing_indexes = [idx['name'] for idx in self.pc.list_indexes()]
            
            if index_name not in existing_indexes:
                logging.info(f"Creating new index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                
                # Wait for index with exponential backoff
                max_retries = 10
                for attempt in range(max_retries):
                    try:
                        index = self.pc.Index(index_name)
                        stats = index.describe_index_stats()
                        logging.info(f"Index ready after {attempt + 1} attempts")
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise Exception(f"Index creation failed after {max_retries} attempts: {e}")
                        wait_time = min(2 ** attempt, 16)  # Exponential backoff, max 16s
                        logging.info(f"Waiting {wait_time}s for index readiness...")
                        time.sleep(wait_time)
            
            # CRITICAL FIX: Ensure all metadata values are properly typed before sending to Pinecone
            safe_chunks = []
            for chunk in chunks:
                safe_chunk = Document(
                    page_content=chunk.page_content,
                    metadata=safe_convert_types(chunk.metadata)
                )
                safe_chunks.append(safe_chunk)
            
            # Create vectorstore with batch processing for large datasets
            if len(safe_chunks) > 1000:
                logging.info(f"Processing {len(safe_chunks)} chunks in batches...")
                vectorstore = PineconeVectorStore.from_documents(
                    documents=safe_chunks[:100],  # Initial batch
                    embedding=self.embeddings,
                    index_name=index_name
                )
                
                # Add remaining documents in batches
                for i in range(100, len(safe_chunks), 100):
                    batch = safe_chunks[i:i+100]
                    vectorstore.add_documents(batch)
                    logging.info(f"Processed batch {i//100 + 1}/{(len(safe_chunks)-1)//100 + 1}")
                    self.metrics['successful_embeddings'] += len(batch)
            else:
                vectorstore = PineconeVectorStore.from_documents(
                    documents=safe_chunks,
                    embedding=self.embeddings,
                    index_name=index_name
                )
                self.metrics['successful_embeddings'] += len(safe_chunks)
            
            return vectorstore
            
        except Exception as e:
            logging.warning(f"Real Pinecone failed: {e}, falling back to mock")
            return self._create_mock_vectorstore(chunks)

    def _create_mock_vectorstore(self, chunks: List[Document]):
        """Create production-quality mock vectorstore."""
        class MockVectorstore:
            def __init__(self, documents, embeddings):
                self.documents = documents
                self.embeddings = embeddings
                logging.info(f"Mock vectorstore created with {len(documents)} documents")
                
                # Pre-compute embeddings for better mock behavior
                try:
                    self.doc_embeddings = [
                        embeddings.embed_query(doc.page_content[:500])  # Limit for performance
                        for doc in documents[:50]  # Limit number for performance
                    ]
                except Exception:
                    self.doc_embeddings = []
            
            def similarity_search(self, query, k=10):
                """Mock similarity search with basic relevance."""
                if not self.documents:
                    return []
                
                # Simple keyword matching for better mock results
                query_words = set(query.lower().split())
                scored_docs = []
                
                for doc in self.documents:
                    doc_words = set(doc.page_content.lower().split())
                    overlap = len(query_words.intersection(doc_words))
                    scored_docs.append((doc, overlap))
                
                # Sort by overlap and return top k
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                result = [doc for doc, _ in scored_docs[:k]]
                
                logging.info(f"Mock search returned {len(result)} results for query: {query[:50]}...")
                return result
            
            def add_documents(self, documents):
                self.documents.extend(documents)
                logging.info(f"Added {len(documents)} documents to mock vectorstore")
        
        return MockVectorstore(chunks, self.embeddings)

    def load_existing_vectorstore(self, index_name: str = None) -> PineconeVectorStore:
        """Load existing vectorstore with comprehensive error handling."""
        if index_name is None:
            index_name = Config.DEFAULT_INDEX_NAME
            
        logging.info(f"Loading existing vectorstore: {index_name}")
        
        try:
            if not self.use_real_pinecone:
                logging.info("Mock Pinecone detected, creating empty mock vectorstore")
                self.document_chunks = []
                return self._create_mock_vectorstore([])
            
            # Load real vectorstore
            vectorstore = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=self.embeddings
            )
            
            # Try to reconstruct document chunks for TF-IDF
            try:
                index = self.pc.Index(index_name)
                stats = index.describe_index_stats()
                total_vectors = stats.get('total_vector_count', 0)
                
                logging.info(f"Found {total_vectors} vectors in existing index")
                
                if total_vectors > 0:
                    # Query to get sample documents for TF-IDF
                    query_response = index.query(
                        vector=[0] * 384,
                        top_k=min(5000, total_vectors),  # Reasonable limit
                        include_metadata=True
                    )
                    
                    self.document_chunks = []
                    for match in query_response.get('matches', []):
                        metadata = match.get('metadata', {})
                        if 'text' in metadata:
                            doc = Document(
                                page_content=metadata['text'],
                                metadata={k: v for k, v in metadata.items() if k != 'text'}
                            )
                            self.document_chunks.append(doc)
                    
                    # Rebuild TF-IDF if we have documents
                    if self.document_chunks:
                        logging.info(f"Rebuilding TF-IDF for {len(self.document_chunks)} documents")
                        documents_text = [chunk.page_content for chunk in self.document_chunks]
                        self.tfidf_vectorizer = TfidfVectorizer(
                            max_features=15000, stop_words='english', ngram_range=(1, 2),
                            min_df=1, max_df=0.95, sublinear_tf=True, norm='l2'
                        )
                        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents_text)
                        logging.info(f"TF-IDF matrix rebuilt: {self.tfidf_matrix.shape}")
                
            except Exception as e:
                logging.warning(f"Could not load existing documents for TF-IDF: {e}")
                self.document_chunks = []
            
            return vectorstore
            
        except Exception as e:
            logging.error(f"Failed to load existing vectorstore: {e}")
            self.document_chunks = []
            return self._create_mock_vectorstore([])

    def optimized_retrieve(self, query: str, vectorstore, k: int = 12) -> List[Document]:
        """Advanced retrieval with hybrid search and caching."""
        cache_key = hashlib.md5(f"{query}_{k}".encode()).hexdigest()
        
        # Check cache first
        if cache_key in self._query_cache:
            self.metrics['cache_hits'] += 1
            return self._query_cache[cache_key]
        
        retrieve_start = time.time()
        try:
            processed_query = self._preprocess_query(query)
            all_results = []
            
            # Semantic search
            try:
                if vectorstore is None:
                    logging.warning("No vectorstore available for semantic search")
                    semantic_results = []
                else:
                    semantic_results = vectorstore.similarity_search(processed_query, k=k)
                    logging.info(f"Semantic search returned {len(semantic_results)} results")
            except Exception as e:
                logging.warning(f"Semantic search failed: {e}")
                semantic_results = []
            
            # TF-IDF search (if available)
            tfidf_results = []
            if self.tfidf_vectorizer and self.tfidf_matrix is not None and self.document_chunks:
                try:
                    tfidf_query_vec = self.tfidf_vectorizer.transform([processed_query])
                    tfidf_scores = cosine_similarity(tfidf_query_vec, self.tfidf_matrix).flatten()
                    
                    # Get top k indices
                    top_indices = np.argsort(tfidf_scores)[-k:][::-1]  # Reverse for descending
                    tfidf_results = [
                        self.document_chunks[i] for i in top_indices 
                        if i < len(self.document_chunks) and tfidf_scores[i] > 0.01
                    ]
                    logging.info(f"TF-IDF search returned {len(tfidf_results)} results")
                except Exception as e:
                    logging.warning(f"TF-IDF search failed: {e}")
            
            # Combine and deduplicate results
            combined_results = semantic_results + tfidf_results
            seen_hashes = set()
            unique_results = []
            
            for doc in combined_results:
                doc_hash = hashlib.md5(doc.page_content[:500].encode()).hexdigest()
                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    unique_results.append(doc)
            
            # Rerank if we have results and reranker is available
            if unique_results and len(unique_results) > 1 and self.reranker is not None:
                try:
                    pairs = [(query, doc.page_content[:1000]) for doc in unique_results]
                    scores = self.reranker.predict(pairs)
                    
                    # Sort by reranker scores
                    scored_results = list(zip(unique_results, scores))
                    scored_results.sort(key=lambda x: x[1], reverse=True)
                    
                    final_results = [doc for doc, _ in scored_results[:k]]
                    logging.info(f"Reranked to {len(final_results)} final results")
                    
                except Exception as e:
                    logging.warning(f"Reranking failed: {e}")
                    final_results = unique_results[:k]
            else:
                final_results = unique_results[:k]
                if self.reranker is None:
                    logging.info("Reranker not available - using original order")
            
            # Cache the results
            if len(self._query_cache) < 1000:  # Prevent memory bloat
                self._query_cache[cache_key] = final_results
            
            retrieve_time = time.time() - retrieve_start
            logging.info(f"Retrieved {len(final_results)} documents in {retrieve_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            logging.error(f"Retrieval failed: {e}")
            self.metrics['errors'] += 1
            return []

    def ask_question(self, query: str, vectorstore) -> str:
        """Answer question with professional, high-quality responses."""
        start_time = time.time()
        self.metrics['queries_processed'] += 1
        
        try:
            logging.info(f"Processing query: {query[:100]}...")
            
            # Handle case when no vectorstore is available
            if vectorstore is None:
                logging.info("No vectorstore available, providing general response")
                response = self.llm.invoke(f"""
You are a helpful AI assistant. Please provide a comprehensive and informative response to this question:

Question: {query}

Please provide a detailed, accurate, and professional response based on your general knowledge.
""")
                response_content = response.content if hasattr(response, 'content') else str(response)
                
                # Calculate metrics
                end_time = time.time()
                response_time = end_time - start_time
                
                # Update metrics
                self.metrics['total_response_time'] += response_time
                self.metrics['avg_response_time'] = self.metrics['total_response_time'] / self.metrics['queries_processed']
                
                # Format final response
                metrics_info = (
                    f"\n\n--- Response Metrics ---\n"
                    f"Response Time: {response_time:.2f}s\n"
                    f"Sources Used: 0 documents (no vectorstore)\n"
                    f"Mode: General Knowledge"
                )
                
                return f"{response_content}{metrics_info}"
            
            # Retrieve relevant documents
            retrieved_docs = self.optimized_retrieve(query, vectorstore)
            
            if not retrieved_docs:
                return "I couldn't find any relevant information in the documents to answer your query. Please try rephrasing your question or check if the document contains the information you're looking for."
            
            # Build context with metadata
            context_parts = []
            sources = set()
            
            for i, doc in enumerate(retrieved_docs):
                chunk_id = doc.metadata.get('chunk_id', f'chunk_{i}')
                source = doc.metadata.get('source', 'document')
                sources.add(source)
                
                # Clean and format content
                content = doc.page_content.strip()
                if content:
                    context_parts.append(f"Section {i+1}: {content}")
            
            if not context_parts:
                return "The retrieved documents don't contain sufficient information to answer your question."
            
            context = '\n\n'.join(context_parts)
            
            # Professional prompt for high-quality responses
            prompt = f"""You are a professional document analyst. Provide a comprehensive, accurate response based on the document content provided.

DOCUMENT CONTENT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Give a direct, professional answer to the question
- Use specific information from the document sections
- Be clear, informative, and well-structured
- If information is limited, explain what is available
- Maintain a helpful, professional tone
- Structure your response with proper paragraphs when needed

PROFESSIONAL RESPONSE:"""

            # Generate response
            response = self.llm.invoke(prompt)
            
            # Process response
            if hasattr(response, 'content'):
                answer = response.content.strip()
            else:
                answer = str(response).strip()
            
            # Ensure quality response
            if not answer or len(answer) < 20:
                answer = "I apologize, but I couldn't generate a comprehensive response based on the available document content. Please try rephrasing your question."
            
            # Add response time to metrics
            response_time = time.time() - start_time
            self.metrics['total_response_time'] += response_time
            self.metrics['avg_response_time'] = self.metrics['total_response_time'] / self.metrics['queries_processed']
            
            logging.info(f"Query answered in {response_time:.2f}s with {len(retrieved_docs)} sources")
            
            return answer
            
        except Exception as e:
            logging.error(f"Question answering failed: {e}")
            self.metrics['errors'] += 1
            return f"I apologize, but I encountered an error while processing your question. Please try again with a different phrasing."

    def _calculate_confidence(self, retrieved_docs: List[Document], response: str) -> float:
        """Calculate response confidence score."""
        try:
            if not retrieved_docs or not response:
                return 0.0
            
            # Factor 1: Number of sources (normalized)
            source_score = min(1.0, len(retrieved_docs) / 10)
            
            # Factor 2: Average document length (indicates detail)
            avg_length = np.mean([len(doc.page_content) for doc in retrieved_docs])
            length_score = min(1.0, avg_length / 1000)
            
            # Factor 3: Response length (indicates thoroughness)
            response_score = min(1.0, len(response) / 500)
            
            # Weighted combination
            confidence = (source_score * 0.4 + length_score * 0.3 + response_score * 0.3)
            
            return round(confidence, 3)
            
        except Exception:
            return 0.5  # Default moderate confidence

    def preprocess_query_cached(self, query: str) -> str:
        """Preprocess query with caching for performance."""
        # Create cache key from query
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        # Check if we have cached result
        if not hasattr(self, '_preprocess_cache'):
            self._preprocess_cache = {}
        
        if cache_key in self._preprocess_cache:
            return self._preprocess_cache[cache_key]
        
        # Process the query
        processed = self._preprocess_query(query)
        
        # Cache the result (with size limit)
        if len(self._preprocess_cache) < 100:  # Limit cache size
            self._preprocess_cache[cache_key] = processed
        
        return processed

    def generate_query_variants(self, query: str) -> List[str]:
        """Generate variants of the query for improved retrieval."""
        variants = [query]  # Always include original
        
        try:
            # Create common variants
            query_lower = query.lower()
            
            # Add expanded forms of common abbreviations
            expansions = {
                'ai': 'artificial intelligence',
                'ml': 'machine learning',
                'dl': 'deep learning',
                'nlp': 'natural language processing',
                'cv': 'computer vision',
                'iot': 'internet of things',
                'api': 'application programming interface',
                'ui': 'user interface',
                'ux': 'user experience',
                'db': 'database'
            }
            
            for abbrev, expansion in expansions.items():
                if abbrev in query_lower:
                    variant = query.replace(abbrev, expansion)
                    if variant != query:
                        variants.append(variant)
            
            # Add question format variations
            if not query.endswith('?'):
                variants.append(f"{query}?")
            
            # Add "what is" format if not present
            if not query_lower.startswith(('what', 'how', 'why', 'when', 'where')):
                variants.append(f"What is {query}?")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_variants = []
            for variant in variants:
                if variant not in seen:
                    seen.add(variant)
                    unique_variants.append(variant)
            
            return unique_variants[:5]  # Limit to 5 variants
            
        except Exception as e:
            logging.warning(f"Query variant generation failed: {e}")
            return [query]  # Return original if generation fails

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        runtime = time.time() - self.start_time
        
        # Initialize missing performance metrics if not present
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = {
                'total_queries': 0,
                'high_confidence_queries': 0,
                'multi_source_queries': 0,
                'avg_response_time': 0.0,
                'cache_hit_rate': 0.0,
                'error_rate': 0.0
            }
        
        # Calculate derived metrics
        total_queries = self.metrics.get('queries_processed', 0)
        if total_queries > 0:
            cache_hit_rate = (self.metrics.get('cache_hits', 0) / total_queries) * 100
            error_rate = (self.metrics.get('errors', 0) / total_queries) * 100
        else:
            cache_hit_rate = 0.0
            error_rate = 0.0
        
        metrics = {
            'total_queries': total_queries,
            'high_confidence_queries': self.performance_metrics.get('high_confidence_queries', 0),
            'multi_source_queries': self.performance_metrics.get('multi_source_queries', 0),
            'avg_response_time': self.metrics.get('avg_response_time', 0.0),
            'cache_hit_rate': cache_hit_rate,
            'error_rate': error_rate,
            'system_runtime': runtime,
            'documents_processed': self.metrics.get('documents_processed', 0),
            'successful_embeddings': self.metrics.get('successful_embeddings', 0),
            'pinecone_operations': self.metrics.get('pinecone_operations', 0)
        }
        
        return metrics

    def calculate_confidence(self, docs: List[Document], response: str) -> float:
        """Calculate confidence score for a response (public method for tests)."""
        return self._calculate_confidence(docs, response)

    def optimized_ask_question(self, query: str, vectorstore) -> str:
        """Enhanced ask question with detailed metrics and validation."""
        start_time = time.time()
        
        try:
            # Initialize performance metrics if not present
            if not hasattr(self, 'performance_metrics'):
                self.performance_metrics = {
                    'total_queries': 0,
                    'high_confidence_queries': 0,
                    'multi_source_queries': 0,
                    'avg_response_time': 0.0
                }
            
            # Update query count
            self.performance_metrics['total_queries'] += 1
            
            # Retrieve relevant documents
            retrieved_docs = self.optimized_retrieve(query, vectorstore)
            
            if not retrieved_docs:
                return "I apologize, but I couldn't find relevant information to answer your question."
            
            # Count unique sources
            sources = set()
            for doc in retrieved_docs:
                source = doc.metadata.get('source', 'unknown')
                sources.add(source)
            
            # Update multi-source count
            if len(sources) > 1:
                self.performance_metrics['multi_source_queries'] += 1
            
            # Create context for LLM
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context provided."""
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Calculate confidence
            confidence = self.calculate_confidence(retrieved_docs, response_content)
            
            # Update high confidence count
            if confidence > 0.7:
                self.performance_metrics['high_confidence_queries'] += 1
            
            # Calculate timing
            end_time = time.time()
            response_time = end_time - start_time
            response_time_ms = int(response_time * 1000)
            
            # Update average response time
            total_queries = self.performance_metrics['total_queries']
            current_avg = self.performance_metrics['avg_response_time']
            self.performance_metrics['avg_response_time'] = (
                (current_avg * (total_queries - 1) + response_time) / total_queries
            )
            
            # Format response with metrics
            confidence_label = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
            
            formatted_response = (
                f"Response: {response_content}\n"
                f"Confidence: {confidence_label} ({confidence:.3f})\n"
                f"Multi-Source Validation: {len(sources)} sources\n"
                f"Time: {response_time_ms}ms\n"
            )
            
            return formatted_response
            
        except Exception as e:
            logging.error(f"Optimized question answering failed: {e}")
            self.metrics['errors'] += 1
            return f"Error processing question: {str(e)}"

    def quick_benchmark(self, query: str, vectorstore) -> Dict[str, Any]:
        """Quick benchmark test for performance validation."""
        try:
            start_time = time.perf_counter()
            response = self.optimized_ask_question(query, vectorstore)
            end_time = time.perf_counter()
            
            response_time = end_time - start_time
            
            # Parse response to extract metrics
            lines = response.split('\n')
            confidence_line = next((line for line in lines if line.startswith('Confidence:')), '')
            sources_line = next((line for line in lines if line.startswith('Multi-Source')), '')
            
            # Extract confidence score
            confidence = 0.0
            if '(' in confidence_line and ')' in confidence_line:
                try:
                    confidence_str = confidence_line.split('(')[1].split(')')[0]
                    confidence = float(confidence_str)
                except (IndexError, ValueError):
                    confidence = 0.0
            
            # Extract source count
            source_count = 1
            if 'sources' in sources_line:
                try:
                    source_count = int(sources_line.split()[2])
                except (IndexError, ValueError):
                    source_count = 1
            
            return {
                'response_time': response_time,
                'confidence': confidence,
                'source_count': source_count,
                'response_length': len(response),
                'success': True
            }
            
        except ZeroDivisionError:
            return {'error': 'division by zero'}
        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }

    def get_system_metrics(self) -> Dict:
        """Get comprehensive system metrics."""
        runtime = time.time() - self.start_time
        
        metrics = self.metrics.copy()
        metrics.update({
            'system_runtime': runtime,
            'cache_size': len(self._query_cache),
            'documents_in_memory': len(self.document_chunks),
            'tfidf_matrix_shape': self.tfidf_matrix.shape if self.tfidf_matrix is not None else None,
            'chunker_stats': self.chunker.get_chunk_statistics() if hasattr(self.chunker, 'get_chunk_statistics') else {},
            'using_real_pinecone': self.use_real_pinecone
        })
        
        return metrics

# Backward compatibility aliases
OptimizedHighPerformanceRAG = ProductionRAGSystem

# API functions
def initialize_rag() -> ProductionRAGSystem:
    """Initialize the production RAG system."""
    return ProductionRAGSystem()

async def process_document(rag_system: ProductionRAGSystem, file_path: str) -> PineconeVectorStore:
    """
    Process a document and create vectorstore with comprehensive error handling.
    
    Args:
        rag_system: The ProductionRAGSystem instance
        file_path: Path to the document to process
        
    Returns:
        PineconeVectorStore: The populated vector store
        
    Raises:
        FileNotFoundError: If the document file does not exist
        ValueError: If the document format is unsupported
        RuntimeError: If processing fails
    """
    try:
        logging.info(f"Processing document: {file_path}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        pages = await rag_system.load_document_async(file_path)
        if not pages:
            raise ValueError(f"No content extracted from: {file_path}")
            
        chunks = rag_system.create_optimized_chunks(pages, file_path)
        if not chunks:
            raise ValueError(f"No chunks created from: {file_path}")
            
        vectorstore = rag_system.create_vectorstore(chunks)
        logging.info("Document processing completed successfully")
        return vectorstore
    except Exception as e:
        logging.error(f"Document processing failed for {file_path}: {e}")
        raise RuntimeError(f"Failed to process document {file_path}: {e}")

# Alias for backward compatibility
process_document_async = process_document

def load_existing_database(rag_system: ProductionRAGSystem, index_name: Optional[str] = None):
    """Load existing vectorstore."""
    return rag_system.load_existing_vectorstore(index_name)

def ask_enhanced_question(rag_system: ProductionRAGSystem, query: str, vectorstore) -> str:
    """Ask question using the RAG system."""
    return rag_system.ask_question(query, vectorstore)