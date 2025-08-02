"""
Scientific answer generator with citation support
"""

import re
from typing import List, Dict, Any
import random

class ScientificAnswerGenerator:
    def __init__(self):
        """Initialize the answer generator"""
        self.biology_keywords = {
            'cell': ['cellular', 'cells', 'cytoplasm', 'membrane'],
            'dna': ['genetic', 'genome', 'nucleotide', 'chromosome'],
            'protein': ['amino acid', 'enzyme', 'peptide', 'polypeptide'],
            'evolution': ['natural selection', 'adaptation', 'species', 'mutation'],
            'metabolism': ['energy', 'atp', 'respiration', 'photosynthesis'],
            'microbiology': ['bacteria', 'microorganism', 'microbe', 'culture']
        }
        
        self.answer_templates = {
            'definition': [
                "Based on the scientific literature, {term} refers to {definition}.",
                "According to the research papers, {term} can be defined as {definition}.",
                "The scientific evidence indicates that {term} is {definition}."
            ],
            'process': [
                "The biological process involves {steps}.",
                "Research shows that this process occurs through {steps}.",
                "Scientific studies demonstrate that {steps}."
            ],
            'general': [
                "Based on the available research, {content}.",
                "According to the scientific literature, {content}.",
                "The research papers indicate that {content}."
            ],
            'comparison': [
                "The studies show that {comparison}.",
                "Research demonstrates the following differences: {comparison}.",
                "Scientific evidence reveals that {comparison}."
            ]
        }
    
    def generate_answer(self, query: str, context: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a scientific answer with citations"""
        try:
            if not context or not context.strip():
                return {
                    'answer': f"I couldn't find relevant information about '{query}' in the uploaded documents. Please upload more relevant biology research papers or try rephrasing your question.",
                    'confidence': 0.0,
                    'sources_used': [],
                    'citations': [],
                    'equations_found': [],
                    'generation_method': 'no_context'
                }
            
            # Analyze query type
            query_type = self._analyze_query_type(query)
            
            # Extract key information from context
            key_info = self._extract_key_information(context, query)
            
            # Generate answer based on query type and context
            answer = self._generate_contextual_answer(query, context, query_type, key_info)
            
            # Extract citations and equations
            citations = self._extract_citations(context)
            equations = self._extract_equations(context)
            
            # Calculate confidence based on context quality
            confidence = self._calculate_confidence(query, context, sources or [])
            
            # Format sources
            sources_used = self._format_sources(sources or [])
            
            return {
                'answer': answer,
                'confidence': confidence,
                'sources_used': sources_used,
                'citations': citations,
                'equations_found': equations,
                'generation_method': 'template_based'
            }
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                'answer': f"I encountered an error while processing your question about '{query}'. Please try again or rephrase your question.",
                'confidence': 0.0,
                'sources_used': [],
                'citations': [],
                'equations_found': [],
                'generation_method': 'error'
            }
    
    def _analyze_query_type(self, query: str) -> str:
        """Analyze the type of query"""
        query_lower = query.lower()
        
        # Definition questions
        if any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning of']):
            return 'definition'
        
        # Process questions
        elif any(word in query_lower for word in ['how does', 'how do', 'process', 'mechanism', 'procedure']):
            return 'process'
        
        # Comparison questions
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'between']):
            return 'comparison'
        
        # Quantitative questions
        elif any(word in query_lower for word in ['how many', 'how much', 'number of', 'amount']):
            return 'quantitative'
        
        else:
            return 'general'
    
    def _extract_key_information(self, context: str, query: str) -> Dict[str, Any]:
        """Extract key information from context relevant to the query"""
        query_words = set(query.lower().split())
        
        # Find sentences that contain query keywords
        sentences = re.split(r'[.!?]+', context)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Skip very short sentences
                sentence_words = set(sentence.lower().split())
                
                # Check overlap with query
                overlap = len(query_words.intersection(sentence_words))
                if overlap > 0:
                    relevant_sentences.append({
                        'text': sentence,
                        'overlap_score': overlap / len(query_words),
                        'has_numbers': bool(re.search(r'\d+', sentence)),
                        'has_technical_terms': self._has_technical_terms(sentence)
                    })
        
        # Sort by relevance
        relevant_sentences.sort(key=lambda x: x['overlap_score'], reverse=True)
        
        return {
            'relevant_sentences': relevant_sentences[:5],  # Top 5 most relevant
            'has_quantitative_info': any(s['has_numbers'] for s in relevant_sentences),
            'technical_density': sum(s['has_technical_terms'] for s in relevant_sentences) / max(1, len(relevant_sentences))
        }
    
    def _has_technical_terms(self, text: str) -> bool:
        """Check if text contains technical biology terms"""
        text_lower = text.lower()
        for category, terms in self.biology_keywords.items():
            if any(term in text_lower for term in terms):
                return True
        return False
    
    def _generate_contextual_answer(self, query: str, context: str, query_type: str, key_info: Dict[str, Any]) -> str:
        """Generate answer based on context and query type"""
        relevant_sentences = key_info.get('relevant_sentences', [])
        
        if not relevant_sentences:
            # Fallback: use first part of context
            first_part = context[:800] + "..." if len(context) > 800 else context
            return f"Based on the available documents: {first_part}"
        
        # Build answer from most relevant sentences
        answer_parts = []
        
        if query_type == 'definition':
            answer_parts.append("Based on the scientific literature:")
            
            for sentence_info in relevant_sentences[:3]:
                sentence = sentence_info['text']
                if len(sentence) > 30:  # Ensure substantial content
                    answer_parts.append(sentence.strip())
        
        elif query_type == 'process':
            answer_parts.append("According to the research papers, the process involves:")
            
            for i, sentence_info in enumerate(relevant_sentences[:4]):
                sentence = sentence_info['text']
                if len(sentence) > 30:
                    answer_parts.append(f"{i+1}. {sentence.strip()}")
        
        elif query_type == 'quantitative':
            answer_parts.append("The research provides the following quantitative information:")
            
            # Prioritize sentences with numbers
            quantitative_sentences = [s for s in relevant_sentences if s['has_numbers']]
            
            for sentence_info in (quantitative_sentences or relevant_sentences)[:3]:
                sentence = sentence_info['text']
                if len(sentence) > 30:
                    answer_parts.append(sentence.strip())
        
        elif query_type == 'comparison':
            answer_parts.append("The scientific literature shows the following comparisons:")
            
            for sentence_info in relevant_sentences[:3]:
                sentence = sentence_info['text']
                if len(sentence) > 30:
                    answer_parts.append(sentence.strip())
        
        else:  # general
            answer_parts.append("Based on the available research:")
            
            for sentence_info in relevant_sentences[:3]:
                sentence = sentence_info['text']
                if len(sentence) > 30:
                    answer_parts.append(sentence.strip())
        
        # Join answer parts
        if len(answer_parts) > 1:
            answer = answer_parts[0] + " " + " ".join(answer_parts[1:])
        else:
            answer = " ".join(answer_parts) if answer_parts else context[:800]
        
        # Ensure proper ending
        if not answer.endswith('.'):
            answer += "."
        
        return answer
    
    def _extract_citations(self, context: str) -> List[str]:
        """Extract citations from context"""
        citation_patterns = [
            r'\[[0-9,\s\-]+\]',  # [1], [1,2], [1-3]
            r'\([^)]*[0-9]{4}[^)]*\)',  # (Author, 2023)
            r'[A-Z][a-z]+\s+et\s+al\.\s*\([0-9]{4}\)',  # Smith et al. (2023)
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, context)
            citations.extend(matches)
        
        # Remove duplicates and return first 10
        return list(set(citations))[:10]
    
    def _extract_equations(self, context: str) -> List[str]:
        """Extract mathematical equations from context"""
        equation_patterns = [
            r'[A-Za-z]+\s*[=+\-*/]\s*[A-Za-z0-9\s+\-*/()]+',
            r'\b[A-Z][a-z]*\s*=\s*[0-9.]+',
            r'[∑∫∂∆αβγδεζηθικλμνξοπρστυφχψω]',
        ]
        
        equations = []
        for pattern in equation_patterns:
            matches = re.findall(pattern, context)
            equations.extend(matches)
        
        # Filter and clean equations
        clean_equations = []
        for eq in equations:
            eq = eq.strip()
            if len(eq) > 3 and len(eq) < 100:  # Reasonable length
                clean_equations.append(eq)
        
        return list(set(clean_equations))[:5]  # Return first 5 unique equations
    
    def _calculate_confidence(self, query: str, context: str, sources: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the answer"""
        confidence = 0.3  # Base confidence
        
        # Query-context relevance
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        
        # Word overlap
        overlap = len(query_words.intersection(context_words))
        overlap_score = min(0.3, overlap / max(1, len(query_words)))
        confidence += overlap_score
        
        # Source quality
        if sources:
            avg_similarity = sum(s.get('similarity', 0) for s in sources) / len(sources)
            confidence += min(0.3, avg_similarity)
            
            # Bonus for multiple sources
            if len(sources) >= 3:
                confidence += 0.1
        
        # Context length (more context usually means better answer)
        if len(context) > 500:
            confidence += 0.1
        
        # Technical content bonus
        if any(term in context.lower() for terms in self.biology_keywords.values() for term in terms):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format sources for display"""
        formatted_sources = []
        
        for i, source in enumerate(sources):
            metadata = source.get('metadata', {})
            
            formatted_source = {
                'source_number': i + 1,
                'filename': metadata.get('source_file', 'Unknown'),
                'section_type': metadata.get('chunk_type', 'content'),
                'similarity_score': source.get('similarity', 0.0),
                'has_equations': metadata.get('has_equations') == 'True',
                'has_citations': metadata.get('has_citations') == 'True',
                'word_count': metadata.get('word_count', 'Unknown')
            }
            formatted_sources.append(formatted_source)
        
        return formatted_sources