"""
Enhanced Scientific answer generator with OpenAI integration for advanced responses
"""

import re
import os
from typing import List, Dict, Any
import openai
from openai import OpenAI
import json

class ScientificAnswerGenerator:
    def __init__(self, api_key: str = None):
        """Initialize the answer generator with OpenAI"""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            print("⚠️ No OpenAI API key found. Using fallback template-based generation.")
            self.use_openai = False
        else:
            try:
                self.client = OpenAI(api_key=self.api_key)
                self.use_openai = True
                print("✅ OpenAI client initialized successfully")
            except Exception as e:
                print(f"⚠️ OpenAI initialization failed: {e}. Using fallback mode.")
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
        
        # Scientific terminology patterns
        self.scientific_patterns = {
            'equations': r'[A-Za-z]+\s*[=+\-*/]\s*[A-Za-z0-9\s+\-*/()]+',
            'citations': r'\[[0-9,\s\-]+\]|\([^)]*[0-9]{4}[^)]*\)',
            'measurements': r'\d+\.?\d*\s*(mg|kg|ml|l|cm|mm|μm|nm|°C|pH)',
            'chemical_formulas': r'[A-Z][a-z]?[0-9]*(?:[A-Z][a-z]?[0-9]*)*',
            'species_names': r'[A-Z][a-z]+\s+[a-z]+'
        }

    def generate_answer(self, query: str, context: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate enhanced scientific answer using OpenAI or fallback"""
        try:
            if not context or not context.strip():
                return self._generate_no_context_response(query)
            
            # Extract scientific elements from context
            scientific_elements = self._extract_scientific_elements(context)
            
            if self.use_openai:
                return self._generate_openai_answer(query, context, scientific_elements, sources)
            else:
                return self._generate_fallback_answer(query, context, scientific_elements, sources)
                
        except Exception as e:
            print(f"Error generating answer: {e}")
            return self._generate_error_response(query, str(e))

    def _generate_openai_answer(self, query: str, context: str, scientific_elements: Dict, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate answer using OpenAI GPT"""
        try:
            # Create enhanced prompt for scientific accuracy
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(query, context, scientific_elements)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # or "gpt-4" for better results
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for scientific accuracy
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
                'citations': scientific_elements.get('citations', []),
                'equations_found': scientific_elements.get('equations', []),
                'scientific_terms': scientific_elements.get('scientific_terms', []),
                'generation_method': 'openai_gpt',
                'model_used': 'gpt-3.5-turbo'
            }
            
        except Exception as e:
            print(f"OpenAI generation failed: {e}")
            return self._generate_fallback_answer(query, context, scientific_elements, sources)

    def _create_system_prompt(self) -> str:
        """Create system prompt for OpenAI"""
        return """You are an expert scientific assistant specializing in biology research. Your role is to:

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

    def _create_user_prompt(self, query: str, context: str, scientific_elements: Dict) -> str:
        """Create user prompt with context and scientific elements"""
        prompt = f"""Question: {query}

Research Context:
{context[:3000]}  # Limit context to avoid token limits

Scientific Elements Found:
"""
        
        if scientific_elements.get('equations'):
            prompt += f"Equations: {', '.join(scientific_elements['equations'][:3])}\n"
        
        if scientific_elements.get('citations'):
            prompt += f"Citations: {', '.join(scientific_elements['citations'][:3])}\n"
        
        if scientific_elements.get('measurements'):
            prompt += f"Measurements: {', '.join(scientific_elements['measurements'][:5])}\n"
        
        if scientific_elements.get('species_names'):
            prompt += f"Species: {', '.join(scientific_elements['species_names'][:3])}\n"

        prompt += """
Please provide a comprehensive, scientifically accurate answer based on this research context. 
Structure your response to directly address the question while incorporating relevant scientific details from the context.
If equations or specific measurements are mentioned, include them in your explanation.
"""
        
        return prompt

    def _extract_scientific_elements(self, context: str) -> Dict[str, List[str]]:
        """Extract scientific elements from context"""
        elements = {
            'equations': [],
            'citations': [],
            'measurements': [],
            'chemical_formulas': [],
            'species_names': [],
            'scientific_terms': []
        }
        
        # Extract equations
        equations = re.findall(self.scientific_patterns['equations'], context)
        elements['equations'] = [eq.strip() for eq in equations if len(eq.strip()) > 3][:5]
        
        # Extract citations
        citations = re.findall(self.scientific_patterns['citations'], context)
        elements['citations'] = list(set([cit.strip() for cit in citations]))[:5]
        
        # Extract measurements
        measurements = re.findall(self.scientific_patterns['measurements'], context)
        elements['measurements'] = list(set(measurements))[:10]
        
        # Extract chemical formulas
        chemical_formulas = re.findall(self.scientific_patterns['chemical_formulas'], context)
        # Filter to likely chemical formulas (contain numbers)
        elements['chemical_formulas'] = [f for f in chemical_formulas if re.search(r'\d', f)][:5]
        
        # Extract species names (capitalized genus + lowercase species)
        species_names = re.findall(self.scientific_patterns['species_names'], context)
        elements['species_names'] = list(set(species_names))[:5]
        
        # Extract scientific terms
        context_lower = context.lower()
        scientific_terms = []
        for category, terms in self.biology_keywords.items():
            for term in terms:
                if term.lower() in context_lower:
                    scientific_terms.append(term)
        elements['scientific_terms'] = list(set(scientific_terms))[:10]
        
        return elements

    def _calculate_openai_confidence(self, answer: str, context: str, query: str) -> float:
        """Calculate confidence for OpenAI-generated answer"""
        confidence = 0.4  # Base confidence for OpenAI responses
        
        # Check if answer contains specific scientific terms
        answer_lower = answer.lower()
        context_lower = context.lower()
        
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
        
        # Length and structure bonus
        if len(answer) > 100:
            confidence += 0.1
        
        # Context utilization bonus
        context_words = set(context_lower.split())
        answer_context_overlap = len(answer_words.intersection(context_words))
        if answer_context_overlap > 10:
            confidence += 0.1
        
        return min(0.95, confidence)  # Cap at 95% for AI-generated content

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
        
        if scientific_elements.get('measurements'):
            answer_parts.append(f"Relevant measurements: {', '.join(scientific_elements['measurements'][:3])}")
        
        answer = " ".join(answer_parts)
        if not answer.endswith('.'):
            answer += "."
        
        confidence = self._calculate_template_confidence(query, context, relevant_sentences)
        
        return {
            'answer': answer,
            'confidence': confidence,
            'sources_used': self._format_sources(sources or []),
            'citations': scientific_elements.get('citations', []),
            'equations_found': scientific_elements.get('equations', []),
            'scientific_terms': scientific_elements.get('scientific_terms', []),
            'generation_method': 'template_based'
        }

    def _calculate_template_confidence(self, query: str, context: str, relevant_sentences: List) -> float:
        """Calculate confidence for template-based answers"""
        if not relevant_sentences:
            return 0.1
        
        # Base confidence from relevance scores
        avg_relevance = sum(s['relevance'] for s in relevant_sentences) / len(relevant_sentences)
        confidence = min(0.7, avg_relevance)
        
        # Context quality bonus
        if len(context) > 500:
            confidence += 0.1
        
        # Multiple sources bonus
        if len(relevant_sentences) >= 3:
            confidence += 0.1
        
        return min(0.8, confidence)  # Cap template-based confidence at 80%

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

    def _generate_no_context_response(self, query: str) -> Dict[str, Any]:
        """Generate response when no context is available"""
        return {
            'answer': f"I couldn't find relevant information about '{query}' in the uploaded documents. Please upload biology research papers that contain information about this topic, or try rephrasing your question with more specific scientific terms.",
            'confidence': 0.0,
            'sources_used': [],
            'citations': [],
            'equations_found': [],
            'scientific_terms': [],
            'generation_method': 'no_context'
        }

    def _generate_error_response(self, query: str, error_msg: str) -> Dict[str, Any]:
        """Generate response when an error occurs"""
        return {
            'answer': f"I encountered an error while processing your question about '{query}'. Please try rephrasing your question or upload more relevant documents. Error details: {error_msg}",
            'confidence': 0.0,
            'sources_used': [],
            'citations': [],
            'equations_found': [],
            'scientific_terms': [],
            'generation_method': 'error'
        }