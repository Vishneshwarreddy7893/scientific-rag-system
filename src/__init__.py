"""
Scientific Literature RAG System - Source Package
"""

# Version info
__version__ = "1.0.0"
__author__ = "Biology RAG System"
__description__ = "A RAG system for scientific literature in biology domain"

# Import main components
try:
    from .document_processor import DocumentProcessor
    from .embeddings import ScientificEmbeddingGenerator
    from .retriever import RAGRetriever, DocumentRetriever
    from .generator import ScientificAnswerGenerator
    from .utils import FileManager, PerformanceTracker, SystemManager, ValidationUtils, LoggingUtils
    
    __all__ = [
        'DocumentProcessor',
        'ScientificEmbeddingGenerator',
        'RAGRetriever',
        'DocumentRetriever',
        'ScientificAnswerGenerator',
        'FileManager',
        'PerformanceTracker',
        'SystemManager',
        'ValidationUtils',
        'LoggingUtils'
    ]
    
except ImportError as e:
    print(f"Warning: Some components could not be imported: {e}")
    __all__ = []