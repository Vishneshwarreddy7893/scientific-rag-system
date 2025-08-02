"""
Document processor for scientific literature with equation and citation detection
"""

import PyPDF2
import re
import os
from typing import List, Dict, Any
import nltk
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

class DocumentProcessor:
    def __init__(self):
        self.equation_patterns = [
            r'[A-Za-z]+\s*[=+\-*/]\s*[A-Za-z0-9\s+\-*/()]+',  # Basic equations
            r'\b[A-Z][a-z]*\s*=\s*[0-9.]+',  # Variable assignments
            r'[∑∫∂∆αβγδεζηθικλμνξοπρστυφχψω]',  # Greek letters and math symbols
            r'\([^)]*[=+\-*/][^)]*\)',  # Equations in parentheses
            r'[0-9]+\s*[+\-*/]\s*[0-9]+\s*=\s*[0-9]+',  # Numeric equations
        ]
        
        self.citation_patterns = [
            r'\[[0-9,\s\-]+\]',  # [1], [1,2], [1-3]
            r'\([^)]*[0-9]{4}[^)]*\)',  # (Author, 2023)
            r'[A-Z][a-z]+\s+et\s+al\.\s*\([0-9]{4}\)',  # Smith et al. (2023)
            r'[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s*\([0-9]{4}\)',  # Smith and Jones (2023)
        ]
        
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
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n[Page {page_num + 1}]\n{page_text}"
                    except Exception as e:
                        print(f"Error extracting page {page_num}: {e}")
                        continue
                
                return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
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
        equations = []
        
        for pattern in self.equation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            equations.extend(matches)
        
        # Remove duplicates and filter short matches
        equations = list(set([eq.strip() for eq in equations if len(eq.strip()) > 3]))
        return equations
    
    def detect_citations(self, text: str) -> List[str]:
        """Detect citations in text"""
        citations = []
        
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        # Remove duplicates
        citations = list(set([cit.strip() for cit in citations]))
        return citations
    
    def detect_section_type(self, text: str) -> str:
        """Detect the type of section based on content"""
        text_lower = text.lower()
        
        # Abstract detection
        if any(word in text_lower for word in ['abstract', 'summary']):
            return 'abstract'
        
        # Methods detection
        elif any(word in text_lower for word in ['method', 'methodology', 'procedure', 'protocol', 'experiment']):
            return 'methods'
        
        # Results detection
        elif any(word in text_lower for word in ['result', 'finding', 'data', 'analysis', 'observation']):
            return 'results'
        
        # Discussion detection
        elif any(word in text_lower for word in ['discussion', 'conclusion', 'interpretation', 'implication']):
            return 'discussion'
        
        # Introduction detection
        elif any(word in text_lower for word in ['introduction', 'background', 'literature review']):
            return 'introduction'
        
        # References detection
        elif any(word in text_lower for word in ['reference', 'bibliography', 'citation']):
            return 'references'
        
        else:
            return 'content'
    
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
            # Split into sentences
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback if NLTK fails
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
            
            # If adding this sentence would exceed chunk size and we have content
            if word_count + sentence_words > chunk_size and current_chunk:
                # Create chunk with metadata
                chunk_data = self._create_chunk_data(current_chunk)
                chunks.append(chunk_data)
                
                # Start new chunk with overlap
                overlap_text = ' '.join(current_chunk.split()[-overlap:])
                current_chunk = overlap_text + " " + sentence
                word_count = len(current_chunk.split())
            else:
                # Add sentence to current chunk
                current_chunk += " " + sentence
                word_count += sentence_words
        
        # Add final chunk
        if current_chunk.strip():
            chunk_data = self._create_chunk_data(current_chunk)
            chunks.append(chunk_data)
        
        return chunks
    
    def _create_chunk_data(self, text: str) -> Dict[str, Any]:
        """Create chunk data with metadata"""
        equations = self.detect_equations(text)
        citations = self.detect_citations(text)
        section_type = self.detect_section_type(text)
        biology_relevance = self.calculate_biology_relevance(text)
        
        return {
            'content': text.strip(),
            'metadata': {
                'chunk_type': section_type,
                'has_equations': len(equations) > 0,
                'has_citations': len(citations) > 0,
                'equation_count': len(equations),
                'citation_count': len(citations),
                'equations': equations[:5],  # Store up to 5 equations
                'citations': citations[:5],  # Store up to 5 citations
                'biology_relevance': biology_relevance,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
        }
    
    def process_document(self, file_path: str, chunk_size: int = 400, overlap: int = 100) -> List[Dict[str, Any]]:
        """Process a document from start to finish"""
        try:
            # Extract text
            raw_text = self.extract_text_from_pdf(file_path)
            if not raw_text:
                return []
            
            # Clean text
            clean_text = self.clean_text(raw_text)
            if not clean_text:
                return []
            
            # Create chunks
            chunks = self.chunk_document(clean_text, chunk_size, overlap)
            
            # Add source file metadata to all chunks
            filename = os.path.basename(file_path)
            for chunk in chunks:
                chunk['metadata']['source_file'] = filename
                chunk['metadata']['file_path'] = file_path
            
            return chunks
            
        except Exception as e:
            print(f"Error processing document {file_path}: {e}")
            return []