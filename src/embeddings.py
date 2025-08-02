"""
Scientific text embeddings using sentence transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import os

class ScientificEmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding generator with a scientific-friendly model"""
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            # Fallback to a smaller model
            try:
                print("Trying fallback model: all-MiniLM-L12-v2")
                self.model = SentenceTransformer("all-MiniLM-L12-v2")
                print("✅ Fallback embedding model loaded successfully")
            except Exception as e2:
                print(f"❌ Fallback model also failed: {e2}")
                raise e2
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)"""
        if self.model is None:
            raise ValueError("Model not loaded properly")
        
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            
            if not valid_texts:
                return np.array([])
            
            # Generate embeddings
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), 384))  # 384 is the dimension for MiniLM
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.model is None:
            return 384  # Default for MiniLM models
        
        # Generate a dummy embedding to get dimension
        try:
            dummy_embedding = self.model.encode(["test"], convert_to_numpy=True)
            return dummy_embedding.shape[1]
        except:
            return 384
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query for retrieval"""
        return self.generate_embeddings([query])[0]
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode multiple documents for indexing"""
        return self.generate_embeddings(documents)
    
    def compute_similarity_batch(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and documents"""
        try:
            # Normalize embeddings
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
            
            # Compute cosine similarity
            similarities = np.dot(doc_norms, query_norm)
            return similarities
            
        except Exception as e:
            print(f"Error computing similarities: {e}")
            return np.zeros(len(document_embeddings))
    
    def enhance_scientific_text(self, text: str) -> str:
        """Enhance text for better scientific embedding"""
        # Add context markers for better embedding
        enhanced_text = text
        
        # Mark equations
        if any(char in text for char in ['=', '+', '-', '*', '/', '∑', '∫']):
            enhanced_text = f"[CONTAINS_EQUATION] {enhanced_text}"
        
        # Mark citations
        if any(pattern in text for pattern in ['[', ']', '(', ')']):
            if any(char.isdigit() for char in text):
                enhanced_text = f"[CONTAINS_CITATION] {enhanced_text}"
        
        # Mark scientific terms
        scientific_indicators = ['protein', 'dna', 'rna', 'cell', 'molecule', 'gene', 'enzyme', 'bacteria']
        if any(term in text.lower() for term in scientific_indicators):
            enhanced_text = f"[BIOLOGY_CONTENT] {enhanced_text}"
        
        return enhanced_text