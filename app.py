"""
Scientific Literature RAG System - Complete Streamlit Application
A complete RAG system for scientific literature with document processing, 
vector search, and intelligent answer generation with OpenAI integration.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import time
import traceback
from typing import List, Dict, Any
import tempfile
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Scientific Literature RAG System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules (inline to avoid import issues)
import pypdf
import re
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
import uuid
import json
import platform
import psutil
import logging
from datetime import datetime

# Import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("OpenAI not available. Install with: pip install openai")

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    with st.spinner("Downloading NLTK data..."):
        nltk.download('punkt', quiet=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.processor = None
    st.session_state.embedder = None
    st.session_state.retriever = None
    st.session_state.db_client = None
    st.session_state.collection = None
    st.session_state.processing_status = ""
    st.session_state.query_count = 0
    st.session_state.generator = None

class DocumentProcessor:
    """Document processor for scientific literature"""
    
    def __init__(self):
        self.biology_terms = {
            'cellular', 'molecular', 'protein', 'dna', 'rna', 'gene', 'chromosome',
            'mitochondria', 'chloroplast', 'enzyme', 'metabolism', 'synthesis',
            'transcription', 'translation', 'replication', 'evolution', 'species',
            'organism', 'bacteria', 'virus', 'cell', 'tissue', 'organ', 'system',
            'photosynthesis', 'respiration', 'genetics', 'heredity', 'mutation',
            'adaptation', 'natural selection', 'biodiversity', 'ecosystem'
        }
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n[Page {page_num + 1}]\n{page_text}"
                    except Exception as e:
                        continue
                
                return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers patterns
        text = re.sub(r'Page\s+\d+', '', text)
        text = re.sub(r'\d+\s*/\s*\d+', '', text)
        
        # Fix common PDF extraction issues
        text = text.replace('�', ' ')
        text = text.replace('\x00', ' ')
        
        return text.strip()
    
    def detect_equations(self, text: str) -> List[str]:
        """Detect mathematical equations in text"""
        equation_patterns = [
            r'[A-Za-z]+\s*[=+\-*/]\s*[A-Za-z0-9\s+\-*/()]+',
            r'\b[A-Z][a-z]*\s*=\s*[0-9.]+',
            r'[∑∫∂∆αβγδεζηθικλμνξοπρστυφχψω]',
            r'\([^)]*[=+\-*/][^)]*\)',
            r'[0-9]+\s*[+\-*/]\s*[0-9]+\s*=\s*[0-9]+',
        ]
        
        equations = []
        for pattern in equation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            equations.extend(matches)
        
        return list(set([eq.strip() for eq in equations if len(eq.strip()) > 3]))
    
    def detect_citations(self, text: str) -> List[str]:
        """Detect citations in text"""
        citation_patterns = [
            r'\[[0-9,\s\-]+\]',
            r'\([^)]*[0-9]{4}[^)]*\)',
            r'[A-Z][a-z]+\s+et\s+al\.\s*\([0-9]{4}\)',
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return list(set([cit.strip() for cit in citations]))
    
    def calculate_biology_relevance(self, text: str) -> float:
        """Calculate relevance to biology domain"""
        text_lower = text.lower()
        biology_word_count = sum(1 for term in self.biology_terms if term in text_lower)
        
        total_words = len(text_lower.split())
        if total_words == 0:
            return 0.0
        
        relevance = min(1.0, biology_word_count / max(20, total_words * 0.1))
        return relevance
    
    def chunk_document(self, text: str, chunk_size: int = 400, overlap: int = 100) -> List[Dict[str, Any]]:
        """Split document into overlapping chunks with metadata"""
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_words = len(sentence.split())
            
            if word_count + sentence_words > chunk_size and current_chunk:
                chunk_data = self._create_chunk_data(current_chunk)
                chunks.append(chunk_data)
                
                overlap_text = ' '.join(current_chunk.split()[-overlap:])
                current_chunk = overlap_text + " " + sentence
                word_count = len(current_chunk.split())
            else:
                current_chunk += " " + sentence
                word_count += sentence_words
        
        if current_chunk.strip():
            chunk_data = self._create_chunk_data(current_chunk)
            chunks.append(chunk_data)
        
        return chunks
    
    def _create_chunk_data(self, text: str) -> Dict[str, Any]:
        """Create chunk data with metadata"""
        equations = self.detect_equations(text)
        citations = self.detect_citations(text)
        biology_relevance = self.calculate_biology_relevance(text)
        
        return {
            'content': text.strip(),
            'metadata': {
                'has_equations': len(equations) > 0,
                'has_citations': len(citations) > 0,
                'equation_count': len(equations),
                'citation_count': len(citations),
                'equations': equations[:5],
                'citations': citations[:5],
                'biology_relevance': biology_relevance,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
        }

class ScientificEmbedder:
    """Scientific text embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            with st.spinner("Loading embedding model..."):
                self.model = SentenceTransformer(self.model_name)
            st.success("✅ Embedding model loaded successfully")
        except Exception as e:
            st.error(f"❌ Error loading embedding model: {e}")
            try:
                self.model = SentenceTransformer("all-MiniLM-L12-v2")
                st.success("✅ Fallback embedding model loaded")
            except Exception as e2:
                st.error(f"❌ Fallback model also failed: {e2}")
                raise e2
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text(s)"""
        if self.model is None:
            raise ValueError("Model not loaded properly")
        
        try:
            valid_texts = [text for text in texts if text and text.strip()]
            
            if not valid_texts:
                return np.array([])
            
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
            return embeddings
            
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return np.zeros((len(texts), 384))

class EnhancedScientificAnswerGenerator:
    """Enhanced Scientific answer generator with OpenAI integration"""
    
    def __init__(self, api_key: str = None):
        """Initialize the answer generator with OpenAI"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key or not OPENAI_AVAILABLE:
            self.use_openai = False
            if not OPENAI_AVAILABLE:
                st.info("💡 OpenAI not available. Using template-based generation.")
            else:
                st.info("💡 No OpenAI API key found. Using template-based generation.")
        else:
            try:
                # Initialize OpenAI client with proper configuration
                self.client = OpenAI(
                    api_key=self.api_key,
                    # Remove any deprecated parameters
                )
                self.use_openai = True
                st.success("✅ OpenAI client initialized successfully")
            except Exception as e:
                st.warning(f"⚠️ OpenAI initialization failed: {e}. Using fallback mode.")
                self.use_openai = False
        
        # Biology domain keywords for context enhancement
        self.biology_keywords = {
            'cellular': ['cell membrane', 'cytoplasm', 'nucleus', 'organelles', 'mitochondria'],
            'molecular': ['DNA', 'RNA', 'protein', 'enzyme', 'molecule', 'amino acid'],
            'genetics': ['gene', 'chromosome', 'allele', 'mutation', 'inheritance'],
            'evolution': ['natural selection', 'adaptation', 'species', 'phylogeny'],
            'ecology': ['ecosystem', 'biodiversity', 'population', 'community'],
            'physiology': ['metabolism', 'homeostasis', 'respiration', 'circulation']
        }

    def generate_answer(self, query: str, context: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate enhanced scientific answer using OpenAI or fallback"""
        try:
            if not context or not context.strip():
                return {
                    'answer': f"I couldn't find relevant information about '{query}' in the uploaded documents. Please upload more relevant biology research papers or try rephrasing your question.",
                    'confidence': 0.0,
                    'sources_used': self._format_sources(sources or []),
                    'generation_method': 'no_context'
                }
            
            # Extract scientific elements from context
            scientific_elements = self._extract_scientific_elements(context)
            
            if self.use_openai:
                return self._generate_openai_answer(query, context, scientific_elements, sources)
            else:
                return self._generate_fallback_answer(query, context, scientific_elements, sources)
                
        except Exception as e:
            st.error(f"Error generating answer: {e}")
            return {
                'answer': f"I encountered an error while processing your question about '{query}'. Please try again or rephrase your question.",
                'confidence': 0.0,
                'sources_used': self._format_sources(sources or []),
                'generation_method': 'error'
            }

    def _generate_openai_answer(self, query: str, context: str, scientific_elements: Dict, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate answer using OpenAI GPT"""
        try:
            # Create enhanced prompt for scientific accuracy
            system_prompt = """You are an expert scientific assistant specializing in biology research. Your role is to:

1. Provide accurate, evidence-based answers from scientific literature
2. Use proper scientific terminology and maintain technical accuracy
3. Explain complex biological concepts clearly and systematically
4. Reference mathematical equations, chemical formulas, and measurements when relevant
5. Maintain scientific rigor while being accessible
6. Always base your answers on the provided research context

Guidelines:
- Be precise with scientific facts and terminology
- Explain biological processes step-by-step when asked
- Include relevant equations or formulas if present in the context
- Use proper scientific nomenclature (species names, chemical compounds)
- Acknowledge limitations if the context doesn't fully answer the question
- Structure answers logically: definition → mechanism → significance/applications"""

            user_prompt = f"""Question: {query}

Research Context:
{context[:3000]}

Scientific Elements Found:
"""
            
            if scientific_elements.get('equations'):
                user_prompt += f"Equations: {', '.join(scientific_elements['equations'][:3])}\n"
            
            if scientific_elements.get('citations'):
                user_prompt += f"Citations: {', '.join(scientific_elements['citations'][:3])}\n"

            user_prompt += """
Please provide a comprehensive, scientifically accurate answer based on this research context. 
Structure your response to directly address the question while incorporating relevant scientific details from the context.
If equations or specific measurements are mentioned, include them in your explanation.
"""
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=800,
                top_p=0.9
            )
            
            answer_text = response.choices[0].message.content.strip()
            
            # Calculate confidence based on response quality
            confidence = self._calculate_openai_confidence(answer_text, context, query)
            
            return {
                'answer': answer_text,
                'confidence': confidence,
                'sources_used': self._format_sources(sources or []),
                'generation_method': 'openai_gpt',
                'model_used': 'gpt-3.5-turbo'
            }
            
        except Exception as e:
            st.error(f"OpenAI generation failed: {e}")
            return self._generate_fallback_answer(query, context, scientific_elements, sources)

    def _generate_fallback_answer(self, query: str, context: str, scientific_elements: Dict, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate answer using template-based approach (fallback)"""
        # Extract relevant sentences from context
        sentences = re.split(r'[.!?]+', context)
        relevant_sentences = []
        
        query_words = set(query.lower().split())
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30:
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                if overlap > 0:
                    relevant_sentences.append({
                        'text': sentence,
                        'relevance': overlap / len(query_words)
                    })
        
        # Sort by relevance
        relevant_sentences.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Build answer
        answer_parts = ["Based on the scientific literature:"]
        
        for sent_info in relevant_sentences[:3]:
            sentence = sent_info['text']
            if len(sentence) > 20:
                answer_parts.append(sentence)
        
        # Add scientific elements if found
        if scientific_elements.get('equations'):
            answer_parts.append(f"Key equations include: {', '.join(scientific_elements['equations'][:2])}")
        
        answer = " ".join(answer_parts)
        if not answer.endswith('.'):
            answer += "."
        
        confidence = min(0.8, 0.3 + len(relevant_sentences) * 0.1)
        
        return {
            'answer': answer,
            'confidence': confidence,
            'sources_used': self._format_sources(sources or []),
            'generation_method': 'template_based'
        }

    def _extract_scientific_elements(self, context: str) -> Dict[str, List[str]]:
        """Extract scientific elements from context"""
        elements = {
            'equations': [],
            'citations': []
        }
        
        # Extract equations
        equation_patterns = [
            r'[A-Za-z]+\s*[=+\-*/]\s*[A-Za-z0-9\s+\-*/()]+',
            r'\b[A-Z][a-z]*\s*=\s*[0-9.]+',
            r'[∑∫∂∆αβγδεζηθικλμνξοπρστυφχψω]'
        ]
        
        equations = []
        for pattern in equation_patterns:
            matches = re.findall(pattern, context)
            equations.extend(matches)
        
        elements['equations'] = [eq.strip() for eq in equations if len(eq.strip()) > 3][:5]
        
        # Extract citations
        citation_patterns = [
            r'\[[0-9,\s\-]+\]',
            r'\([^)]*[0-9]{4}[^)]*\)',
            r'[A-Z][a-z]+\s+et\s+al\.\s*\([0-9]{4}\)'
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, context)
            citations.extend(matches)
        
        elements['citations'] = list(set([cit.strip() for cit in citations]))[:5]
        
        return elements

    def _calculate_openai_confidence(self, answer: str, context: str, query: str) -> float:
        """Calculate confidence for OpenAI-generated answer"""
        confidence = 0.4  # Base confidence for OpenAI responses
        
        # Check if answer contains specific scientific terms
        answer_lower = answer.lower()
        
        # Term overlap bonus
        query_words = set(query.lower().split())
        answer_words = set(answer_lower.split())
        overlap = len(query_words.intersection(answer_words))
        confidence += min(0.2, overlap / max(1, len(query_words)))
        
        # Scientific terminology bonus
        scientific_term_count = 0
        for terms in self.biology_keywords.values():
            for term in terms:
                if term.lower() in answer_lower:
                    scientific_term_count += 1
        
        confidence += min(0.2, scientific_term_count * 0.02)
        
        # Length bonus
        if len(answer) > 100:
            confidence += 0.1
        
        return min(0.95, confidence)

    def _format_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for display"""
        formatted_sources = []
        
        for i, source in enumerate(sources):
            metadata = source.get('metadata', {})
            
            formatted_source = {
                'source_number': i + 1,
                'filename': metadata.get('source_file', 'Unknown'),
                'similarity_score': source.get('similarity', 0.0),
                'has_equations': metadata.get('has_equations') == 'True',
                'has_citations': metadata.get('has_citations') == 'True',
                'word_count': metadata.get('word_count', 'Unknown')
            }
            formatted_sources.append(formatted_source)
        
        return formatted_sources

def initialize_system():
    """Initialize all system components"""
    try:
        with st.spinner("🚀 Initializing Scientific RAG System..."):
            # Initialize document processor
            st.session_state.processor = DocumentProcessor()
            
            # Initialize embedder
            st.session_state.embedder = ScientificEmbedder()
            
            # Initialize ChromaDB
            persist_directory = "chroma_db"
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            
            st.session_state.db_client = chromadb.PersistentClient(path=persist_directory)
            
            try:
                st.session_state.collection = st.session_state.db_client.get_collection("scientific_papers")
            except:
                st.session_state.collection = st.session_state.db_client.create_collection(
                    name="scientific_papers",
                    metadata={"description": "Scientific literature for biology RAG system"}
                )
            
            # Initialize enhanced answer generator
            api_key = os.getenv('OPENAI_API_KEY')
            st.session_state.generator = EnhancedScientificAnswerGenerator(api_key=api_key)
            
            st.session_state.initialized = True
            st.success("✅ System initialized successfully!")
            
    except Exception as e:
        st.error(f"❌ System initialization failed: {str(e)}")
        return False
    
    return True

def process_uploaded_file(uploaded_file, chunk_size=400, chunk_overlap=100):
    """Process uploaded PDF file"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name
        
        # Extract text
        text = st.session_state.processor.extract_text_from_pdf(temp_path)
        
        if not text.strip():
            st.error("No text could be extracted from the PDF")
            return []
        
        # Clean text
        clean_text = st.session_state.processor.clean_text(text)
        
        # Create chunks
        chunks = st.session_state.processor.chunk_document(clean_text, chunk_size, chunk_overlap)
        
        # Add source file metadata
        for chunk in chunks:
            chunk['metadata']['source_file'] = uploaded_file.name
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return chunks
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        return []

def add_documents_to_db(chunks):
    """Add document chunks to ChromaDB"""
    try:
        if not chunks:
            return False
        
        # Prepare data
        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [chunk['content'] for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = {}
            for key, value in chunk['metadata'].items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
                elif isinstance(value, list):
                    metadata[key] = ', '.join(str(v) for v in value[:3])
            metadatas.append(metadata)
        
        # Add to collection
        st.session_state.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return True
        
    except Exception as e:
        st.error(f"Error adding documents to database: {str(e)}")
        return False

def search_documents(query, n_results=5):
    """Search for relevant documents"""
    try:
        results = st.session_state.collection.query(
            query_texts=[query],
            n_results=min(n_results, 10),
            include=['documents', 'metadatas', 'distances']
        )
        
        formatted_results = []
        
        if results['documents'] and results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similarity = max(0, 1 - distance)
                
                formatted_result = {
                    'content': doc,
                    'metadata': metadata,
                    'similarity': similarity,
                    'rank': i + 1
                }
                formatted_results.append(formatted_result)
        
        return formatted_results
        
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return []

def search_and_answer(query, num_sources):
    """Search documents and generate answer using enhanced generator"""
    start_time = time.time()
    
    with st.spinner("🔍 Searching documents and generating answer..."):
        try:
            # Search documents
            search_results = search_documents(query, num_sources)
            
            # Generate answer using enhanced generator
            context_text = "\n\n".join([f"Source {i+1}: {result['content']}" for i, result in enumerate(search_results[:3])])
            answer_result = st.session_state.generator.generate_answer(query, context_text, search_results)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Store result
            st.session_state.last_result = {
                'query': query,
                'answer': answer_result,
                'search_results': search_results,
                'response_time': response_time
            }
            
            # Update query count
            st.session_state.query_count += 1
            
        except Exception as e:
            st.error(f"Error during search: {str(e)}")

def get_database_stats():
    """Get database statistics"""
    try:
        count = st.session_state.collection.count()
        
        if count == 0:
            return {'total_documents': 0, 'files': {}}
        
        all_docs = st.session_state.collection.get(include=['metadatas'])
        
        stats = {
            'total_documents': count,
            'files': {},
            'has_equations': 0,
            'has_citations': 0
        }
        
        for metadata in all_docs['metadatas']:
            source_file = metadata.get('source_file', 'Unknown')
            stats['files'][source_file] = stats['files'].get(source_file, 0) + 1
            
            if metadata.get('has_equations') == 'True':
                stats['has_equations'] += 1
            
            if metadata.get('has_citations') == 'True':
                stats['has_citations'] += 1
        
        return stats
        
    except Exception as e:
        st.error(f"Error getting database stats: {str(e)}")
        return {'total_documents': 0, 'files': {}}

def main():
    """Main application function"""
    
    # Header
    st.title("🔬 Scientific Literature RAG System")
    st.markdown("*Intelligent Question-Answering System for Biology Research Papers with AI Enhancement*")
    
    # Initialize system if not done
    if not st.session_state.initialized:
        if not initialize_system():
            st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 System Menu")
        
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "📄 Document Management", "🔍 Search & Query", "📊 System Status"]
        )
        
        st.divider()
        
        # API Key Configuration
        st.subheader("🔑 OpenAI Configuration")
        current_key = os.getenv('OPENAI_API_KEY', '')
        api_key_input = st.text_input(
            "OpenAI API Key:",
            value="***" if current_key else "",
            type="password",
            help="Enter your OpenAI API key for enhanced answers"
        )
        
        if api_key_input and api_key_input != "***":
            os.environ['OPENAI_API_KEY'] = api_key_input
            if hasattr(st.session_state, 'generator'):
                st.session_state.generator = EnhancedScientificAnswerGenerator(api_key=api_key_input)
        
        st.divider()
        
        # Quick stats
        stats = get_database_stats()
        st.metric("Documents Loaded", stats.get('total_documents', 0))
        
        if stats.get('total_documents', 0) > 0:
            st.metric("Files", len(stats.get('files', {})))
            st.metric("With Equations", stats.get('has_equations', 0))
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📄 Document Management":
        show_document_management()
    elif page == "🔍 Search & Query":
        show_search_page()
    elif page == "📊 System Status":
        show_system_status()

def show_home_page():
    """Display home page"""
    st.header("Welcome to the Enhanced Scientific Literature RAG System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What is this system?
        
        This is an **Enhanced Retrieval-Augmented Generation (RAG)** system for **Biology research papers** with **OpenAI integration**. It can:
        
        - 📚 **Process PDF research papers** and extract key information
        - 🔍 **Search through documents** using semantic similarity
        - 🤖 **Generate intelligent answers** using OpenAI GPT-3.5-turbo
        - 📊 **Handle mathematical equations** and scientific terminology
        - 📖 **Provide proper citations** for all answers
        - 🧠 **Advanced AI reasoning** for complex scientific queries
        
        ### 🚀 Quick Start:
        
        1. **Add API Key**: Enter your OpenAI API key in the sidebar (optional but recommended)
        2. **Upload Papers**: Go to "Document Management" and upload biology PDFs
        3. **Ask Questions**: Use "Search & Query" to ask about the papers
        4. **Get AI Answers**: Receive intelligent, citation-backed responses
        """)
        
        # Quick action buttons
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("📄 Upload Documents", use_container_width=True):
                st.session_state.current_page = "📄 Document Management"
                st.rerun()
        
        with col_b:
            if st.button("🔍 Ask Questions", use_container_width=True):
                st.session_state.current_page = "🔍 Search & Query"
                st.rerun()
        
        with col_c:
            if st.button("📊 View Status", use_container_width=True):
                st.session_state.current_page = "📊 System Status"
                st.rerun()
    
    with col2:
        st.markdown("### 📈 System Overview")
        
        stats = get_database_stats()
        
        st.metric("Total Documents", stats.get('total_documents', 0))
        st.metric("Queries Processed", st.session_state.query_count)
        
        # API Status
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            st.success("🤖 OpenAI: Enabled")
        else:
            st.warning("🤖 OpenAI: Disabled")
        
        if stats.get('files'):
            st.markdown("**Loaded Files:**")
            for filename, count in list(stats['files'].items())[:3]:
                st.text(f"• {filename} ({count} chunks)")

def show_document_management():
    """Display document management interface"""
    st.header("📄 Document Management")
    
    tabs = st.tabs(["📤 Upload Documents", "📋 Manage Existing"])
    
    with tabs[0]:
        st.subheader("Upload New Research Papers")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files to upload",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload biology research papers in PDF format"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size / 1024 / 1024:.1f} MB)")
            
            st.subheader("Processing Options")
            chunk_size = st.slider("Chunk Size (words):", 200, 800, 400)
            chunk_overlap = st.slider("Chunk Overlap:", 50, 200, 100)
            
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                process_documents(uploaded_files, chunk_size, chunk_overlap)
    
    with tabs[1]:
        st.subheader("Existing Documents")
        
        stats = get_database_stats()
        
        if stats.get('total_documents', 0) == 0:
            st.info("No documents uploaded yet.")
        else:
            file_data = [{'Filename': k, 'Chunks': v} for k, v in stats.get('files', {}).items()]
            st.dataframe(file_data, use_container_width=True)
            
            if st.button("🗑️ Clear All Documents", type="secondary"):
                clear_database()

def process_documents(uploaded_files, chunk_size, chunk_overlap):
    """Process uploaded documents"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        total_files = len(uploaded_files)
        processed_count = 0
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            chunks = process_uploaded_file(uploaded_file, chunk_size, chunk_overlap)
            
            if chunks:
                if add_documents_to_db(chunks):
                    processed_count += 1
                    st.success(f"✅ Processed {uploaded_file.name} - {len(chunks)} chunks created")
                else:
                    st.error(f"❌ Failed to add {uploaded_file.name} to database")
            else:
                st.error(f"❌ Failed to process {uploaded_file.name}")
            
            progress_bar.progress((i + 1) / total_files)
        
        if processed_count > 0:
            st.success(f"✅ Successfully processed {processed_count}/{total_files} documents!")
            st.balloons()
        else:
            st.warning("No documents were successfully processed.")
        
        status_text.empty()
        progress_bar.empty()
        
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        status_text.empty()
        progress_bar.empty()

def show_search_page():
    """Display search and query interface"""
    st.header("🔍 Search & Query Interface")
    
    stats = get_database_stats()
    
    if stats.get('total_documents', 0) == 0:
        st.warning("⚠️ No documents loaded yet. Please upload some research papers first.")
        return
    
    st.subheader("Ask Your Scientific Question")
    
    # API Status indicator
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        st.success("🤖 AI-Enhanced Mode: ON (OpenAI GPT-3.5-turbo)")
    else:
        st.info("🤖 AI-Enhanced Mode: OFF (Template-based responses)")
    
    with st.expander("💡 Sample Questions"):
        sample_questions = [
            "What is photosynthesis and how does it work?",
            "Explain the process of DNA replication",
            "What are the functions of mitochondria in cells?",
            "How do enzymes work in cellular metabolism?",
            "Describe the mechanism of protein synthesis",
            "What is the role of ribosomes in translation?",
            "How does natural selection drive evolution?",
            "What are the differences between prokaryotic and eukaryotic cells?"
        ]
        
        for question in sample_questions:
            if st.button(f"📝 {question}", key=f"sample_{hash(question)}"):
                st.session_state.current_query = question
    
    query = st.text_area(
        "Enter your question:",
        value=st.session_state.get('current_query', ''),
        height=100,
        placeholder="e.g., What is photosynthesis and how does it work in plant cells?",
        help="Ask any question about the biology research papers you've uploaded"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        num_sources = st.slider("Max Sources to Use:", 1, 10, 5)
    
    with col2:
        if st.button("🔍 Search & Answer", type="primary", use_container_width=True):
            if query.strip():
                search_and_answer(query, num_sources)
            else:
                st.warning("Please enter a question.")
    
    if 'last_result' in st.session_state and st.session_state.last_result:
        st.divider()
        display_results(st.session_state.last_result)

def display_results(result):
    """Display search results and answer"""
    st.subheader("🤖 Generated Answer")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(result['answer']['answer'])
        
        confidence = result['answer']['confidence']
        generation_method = result['answer'].get('generation_method', 'unknown')
        
        if confidence >= 0.7:
            st.success(f"🟢 High confidence ({confidence:.1%}) - Method: {generation_method}")
        elif confidence >= 0.4:
            st.warning(f"🟡 Medium confidence ({confidence:.1%}) - Method: {generation_method}")
        else:
            st.error(f"🔴 Low confidence ({confidence:.1%}) - Method: {generation_method}")
    
    with col2:
        st.metric("Response Time", f"{result['response_time']:.2f}s")
        st.metric("Sources Used", len(result['answer']['sources_used']))
        
        # Show AI model used if available
        if result['answer'].get('model_used'):
            st.info(f"AI Model: {result['answer']['model_used']}")
    
    # Sources section
    if result['answer']['sources_used']:
        st.subheader("📚 Sources")
        
        for source in result['answer']['sources_used']:
            with st.expander(f"Source {source['source_number']}: {source['filename']}"):
                st.write(f"**File:** {source['filename']}")
                st.write(f"**Similarity Score:** {source['similarity_score']:.1%}")
                
                if source.get('has_equations'):
                    st.write("📊 Contains equations")
                if source.get('has_citations'):
                    st.write("📚 Contains citations")
    
    # Context preview
    if result['search_results']:
        with st.expander("📄 View Retrieved Context"):
            context_text = ""
            for i, res in enumerate(result['search_results'][:3]):
                context_text += f"Source {i+1}: {res['content'][:500]}...\n\n"
            
            st.text_area(
                "Context used for answer generation:",
                value=context_text,
                height=300
            )

def show_system_status():
    """Display system status"""
    st.header("📊 System Status")
    
    tabs = st.tabs(["📈 Performance", "💾 Database", "⚙️ System Info", "🤖 AI Status"])
    
    with tabs[0]:
        st.subheader("Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Queries Processed", st.session_state.query_count)
        
        with col2:
            stats = get_database_stats()
            st.metric("Documents Loaded", stats.get('total_documents', 0))
        
        with col3:
            st.metric("Files Processed", len(stats.get('files', {})))
    
    with tabs[1]:
        st.subheader("Database Information")
        
        stats = get_database_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Chunks", stats.get('total_documents', 0))
        
        with col2:
            st.metric("With Equations", stats.get('has_equations', 0))
        
        with col3:
            st.metric("With Citations", stats.get('has_citations', 0))
        
        if stats.get('files'):
            st.subheader("Files Breakdown")
            file_data = [{'Filename': k, 'Chunks': v} for k, v in stats['files'].items()]
            st.dataframe(file_data, use_container_width=True)
        
        if st.button("🗑️ Clear Database", type="secondary"):
            clear_database()
    
    with tabs[2]:
        st.subheader("System Information")
        
        system_info = {
            'Platform': platform.platform(),
            'Python Version': platform.python_version(),
            'CPU Count': os.cpu_count(),
            'Available Memory (GB)': f"{psutil.virtual_memory().available / (1024**3):.2f}",
            'Total Memory (GB)': f"{psutil.virtual_memory().total / (1024**3):.2f}",
        }
        
        for key, value in system_info.items():
            st.write(f"**{key}:** {value}")
    
    with tabs[3]:
        st.subheader("AI Enhancement Status")
        
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            st.success("✅ OpenAI API Key: Configured")
            if OPENAI_AVAILABLE:
                st.success("✅ OpenAI Library: Installed")
                st.success("✅ AI Mode: Enhanced responses with GPT-3.5-turbo")
                
                # Test API connection
                if st.button("🧪 Test API Connection"):
                    try:
                        test_client = OpenAI(api_key=api_key)
                        response = test_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": "Hello"}],
                            max_tokens=5
                        )
                        st.success("✅ API Connection: Working")
                    except Exception as e:
                        st.error(f"❌ API Connection Failed: {e}")
            else:
                st.error("❌ OpenAI Library: Not installed")
        else:
            st.warning("⚠️ OpenAI API Key: Not configured")
            st.info("💡 Add your API key in the sidebar for enhanced AI responses")
        
        st.info("🔄 Fallback Mode: Template-based responses always available")

def clear_database():
    """Clear all documents from database"""
    try:
        st.session_state.db_client.delete_collection("scientific_papers")
        st.session_state.collection = st.session_state.db_client.create_collection(
            name="scientific_papers",
            metadata={"description": "Scientific literature for biology RAG system"}
        )
        st.success("✅ Database cleared successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing database: {str(e)}")

if __name__ == "__main__":
    main()