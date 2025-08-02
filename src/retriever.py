"""
RAG retriever using ChromaDB for scientific literature
"""

import chromadb
from chromadb.config import Settings
import os
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

class DocumentRetriever:
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize ChromaDB client"""
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persist directory if it doesn't exist
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection("scientific_papers")
                print("✅ Connected to existing ChromaDB collection")
            except:
                self.collection = self.client.create_collection(
                    name="scientific_papers",
                    metadata={"description": "Scientific literature for biology RAG system"}
                )
                print("✅ Created new ChromaDB collection")
                
        except Exception as e:
            print(f"❌ Error initializing ChromaDB: {e}")
            raise e
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector database"""
        if not documents:
            return
        
        try:
            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []
            
            for doc in documents:
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                
                # Get text content
                content = doc.get('content', '')
                texts.append(content)
                
                # Prepare metadata (ChromaDB requires string values)
                metadata = doc.get('metadata', {})
                clean_metadata = {}
                
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        clean_metadata[key] = str(value)
                    elif isinstance(value, list):
                        # Convert lists to comma-separated strings
                        clean_metadata[key] = ', '.join(str(v) for v in value[:3])  # Limit to first 3 items
                
                metadatas.append(clean_metadata)
            
            # Add to collection
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Added {len(documents)} documents to ChromaDB")
            
        except Exception as e:
            print(f"❌ Error adding documents to ChromaDB: {e}")
            raise e
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, 10),  # Limit to avoid errors
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score
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
            print(f"❌ Error searching documents: {e}")
            return []
    
    def search_by_section_type(self, query: str, section_type: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for documents of a specific section type"""
        try:
            # Query with metadata filter
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, 10),
                where={"chunk_type": section_type},
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
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
            print(f"❌ Error searching by section type: {e}")
            return self.search_documents(query, n_results)  # Fallback to general search
    
    def search_with_equations(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for documents containing equations"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, 10),
                where={"has_equations": "True"},
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
            print(f"❌ Error searching with equations: {e}")
            return self.search_documents(query, n_results)
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        try:
            # Get collection info
            count = self.collection.count()
            
            if count == 0:
                return {
                    'total_documents': 0,
                    'files': {},
                    'section_types': {},
                    'has_equations': 0,
                    'has_citations': 0
                }
            
            # Get all documents to analyze
            all_docs = self.collection.get(include=['metadatas'])
            
            stats = {
                'total_documents': count,
                'files': {},
                'section_types': {},
                'has_equations': 0,
                'has_citations': 0
            }
            
            # Analyze metadata
            for metadata in all_docs['metadatas']:
                # Count by source file
                source_file = metadata.get('source_file', 'Unknown')
                stats['files'][source_file] = stats['files'].get(source_file, 0) + 1
                
                # Count by section type
                section_type = metadata.get('chunk_type', 'content')
                stats['section_types'][section_type] = stats['section_types'].get(section_type, 0) + 1
                
                # Count equations and citations
                if metadata.get('has_equations') == 'True':
                    stats['has_equations'] += 1
                
                if metadata.get('has_citations') == 'True':
                    stats['has_citations'] += 1
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting document stats: {e}")
            return {'total_documents': 0, 'files': {}, 'section_types': {}}
    
    def delete_by_source_file(self, filename: str):
        """Delete all documents from a specific source file"""
        try:
            # Get all documents from the file
            results = self.collection.get(
                where={"source_file": filename},
                include=['ids']
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"✅ Deleted {len(results['ids'])} documents from {filename}")
            
        except Exception as e:
            print(f"❌ Error deleting documents: {e}")
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection("scientific_papers")
            self.collection = self.client.create_collection(
                name="scientific_papers",
                metadata={"description": "Scientific literature for biology RAG system"}
            )
            print("✅ Collection cleared successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")
            return False


class RAGRetriever:
    def __init__(self):
        """Initialize the RAG retriever"""
        self.document_retriever = DocumentRetriever()
    
    def retrieve_context(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Retrieve context for a query"""
        try:
            # Search for relevant documents
            search_results = self.document_retriever.search_documents(query, n_results)
            
            if not search_results:
                return {
                    'context': '',
                    'sources': [],
                    'total_results': 0
                }
            
            # Format context
            context_parts = []
            sources = []
            
            for i, result in enumerate(search_results):
                content = result['content']
                if content and len(content.strip()) > 10:
                    context_parts.append(f"[Source {i+1}] {content}")
                    
                    # Add source information
                    metadata = result['metadata']
                    source_info = {
                        'source_number': i + 1,
                        'filename': metadata.get('source_file', 'Unknown'),
                        'section_type': metadata.get('chunk_type', 'content'),
                        'similarity_score': result['similarity'],
                        'has_equations': metadata.get('has_equations') == 'True',
                        'has_citations': metadata.get('has_citations') == 'True'
                    }
                    sources.append(source_info)
            
            context = '\n\n'.join(context_parts)
            
            return {
                'context': context,
                'sources': search_results,  # Return original results for compatibility
                'formatted_sources': sources,
                'total_results': len(search_results)
            }
            
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            return {
                'context': '',
                'sources': [],
                'total_results': 0
            }