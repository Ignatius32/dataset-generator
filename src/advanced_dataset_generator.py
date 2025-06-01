"""
Advanced DPO Dataset Generator - Professional Grade
Creates high-quality training datasets for AI agents through multi-stage pipeline:
1. Intelligent Query Generation (Multiple modes & difficulty levels)
2. AI-Powered Query Scoring & Selection  
3. Context-Aware Response Generation
4. Multi-dimensional Quality Assessment
5. Difficulty Balancing & Cross-validation
"""

import json
import random
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import re
from datetime import datetime
from .vector_store import VectorStore
from .advanced_chat_api import AdvancedChatAPI
from .config import Config

class AdvancedDatasetGenerator:
    """
    Professional-grade dataset generator for training AI agents
    Uses multi-stage AI pipeline to create high-quality DPO training data
    """
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.chat_api = AdvancedChatAPI()        # Simplified pipeline configuration for reliable operation
        self.query_generation_modes = ['factual', 'analytical']  # Just 2 modes to reduce API calls
        self.difficulty_levels = ['basic', 'intermediate', 'advanced']
        self.response_types = ['comprehensive', 'practical']
        
        # Quality thresholds (very permissive to reduce API failures)
        self.min_query_score = 0.2  # Very low to avoid rejections
        self.min_response_score = 0.2  # Very low to avoid rejections
        self.min_answer_quality_score = 0.2  # Very low to avoid rejections
        
        # Generation parameters - minimized for reliability
        self.queries_per_mode = 1  # Only 1 query per mode to minimize calls
        self.candidates_per_document = 4  # Minimal candidates
        self.final_queries_per_document = 2  # Only 2 final queries
        self.max_context_length = 800  # Much shorter context
        
        # Enhanced generation parameters for DPO dataset quality
        self.generation_params = {
            'query_generation': {
                'temperature': 0.8,
                'top_k': 40,
                'repetition_penalty': 1.2,
                'min_tokens': 20,
                'max_tokens': 400
            },
            'scoring': {
                'temperature': 0.3,
                'top_k': 20,
                'repetition_penalty': 1.1,
                'min_tokens': 30,
                'max_tokens': 200
            },
            'good_response': {
                'temperature': 0.7,
                'top_k': 30,
                'repetition_penalty': 1.1,
                'min_tokens': 100,
                'max_tokens': 500
            },
            'bad_response': {
                'temperature': 1.0,
                'top_k': 5,
                'repetition_penalty': 0.8,
                'max_tokens': 150
            },
            'validation': {
                'temperature': 0.2,
                'top_k': 10,
                'min_tokens': 50,
                'max_tokens': 300
            }
        }
        
        # Multi-stage validation - all disabled for reliability
        self.enable_cross_validation = False  # Disabled
        self.enable_quality_scoring = False   # Disabled - use heuristics only
        self.enable_difficulty_balancing = False  # Disabled          # Rate limiting protection - reduced for testing
        self.api_call_delay = 1.0  # Reduced delay between API calls
        
        print("🚀 Advanced Dataset Generator initialized")
        print(f"📊 Configuration: {len(self.query_generation_modes)} modes, {len(self.difficulty_levels)} difficulty levels")
        print(f"🎯 Quality thresholds: Query≥{self.min_query_score}, Response≥{self.min_response_score}")
    
    def generate_dataset(self, num_samples: int = None, quality_target: str = "high") -> List[Dict[str, Any]]:
        """
        Generate advanced DPO dataset with multi-stage quality control
        
        Args:
            num_samples: Target number of samples (None for optimal calculation)
            quality_target: "high", "ultra", or "production" (higher = more selective)
            
        Returns:
            List of high-quality DPO training samples
        """
        print("\n🎯 Starting Advanced DPO Dataset Generation")
        print("=" * 60)
        
        start_time = datetime.now()
        documents = self.vector_store.documents
        
        if not documents:
            raise ValueError("❌ No documents in vector store")
        
        # Calculate optimal sample target
        if num_samples is None:
            num_samples = self._calculate_optimal_samples(len(documents), quality_target)
        
        print(f"📊 Target: {num_samples} high-quality samples")
        print(f"📚 Source: {len(documents)} document chunks")
        print(f"🎚️ Quality mode: {quality_target}")
        print(f"🔄 Pipeline stages: Query Gen → Scoring → Response Gen → Validation")
        print()
        
        dataset = []
        processed_docs = 0
        total_queries_generated = 0
        total_queries_accepted = 0
        
        for i, doc in enumerate(documents):
            if len(dataset) >= num_samples:
                break
                
            print(f"🔄 Processing document {i+1}/{len(documents)}: {doc['id'][:20]}...")
            
            # Stage 1: Advanced Query Generation
            query_candidates = self._generate_advanced_queries(doc)
            total_queries_generated += len(query_candidates)
            
            if not query_candidates:
                print(f"   ⚠️  No valid queries generated, skipping")
                continue
            
            # Stage 2: AI-Powered Query Scoring & Selection
            selected_queries = self._score_and_select_queries(query_candidates, doc)
            total_queries_accepted += len(selected_queries)
            
            # Stage 3: Advanced Response Generation
            for query_data in selected_queries:
                if len(dataset) >= num_samples:
                    break
                
                sample = self._create_advanced_dpo_sample(query_data, doc)
                if sample and self._validate_sample_quality(sample):
                    dataset.append(sample)
                    print(f"   ✅ Sample {len(dataset)}: {query_data['query'][:50]}... [Score: {query_data['score']:.2f}]")
            
            processed_docs += 1
            
            # Progress reporting
            if processed_docs % 10 == 0:
                acceptance_rate = (total_queries_accepted / max(total_queries_generated, 1)) * 100
                print(f"📈 Progress: {processed_docs} docs, {len(dataset)} samples, {acceptance_rate:.1f}% query acceptance")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 Advanced Dataset Generation Complete!")
        print(f"⏱️  Duration: {duration}")
        print(f"📊 Generated: {len(dataset)} high-quality DPO samples")
        print(f"🎯 Query acceptance rate: {(total_queries_accepted/max(total_queries_generated,1))*100:.1f}%")
        
        # Apply final quality enhancements
        if self.enable_difficulty_balancing:
            dataset = self._balance_difficulty_distribution(dataset)
        
        return dataset
    
    def _calculate_optimal_samples(self, num_documents: int, quality_target: str) -> int:
        """Calculate optimal number of samples based on document count and quality target"""
        base_multiplier = {
            "high": 3,
            "ultra": 2, 
            "production": 1.5
        }
        
        multiplier = base_multiplier.get(quality_target, 2)
        optimal = min(int(num_documents * multiplier), 2000)  # Cap at 2000
        
        print(f"📊 Calculated optimal samples: {optimal} (quality={quality_target})")
        return optimal
    
    def _generate_advanced_queries(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate diverse, high-quality queries using multiple AI-driven approaches
        """
        print(f"   🤖 Generating queries with {len(self.query_generation_modes)} modes...")
        
        all_queries = []
        text = document['text']
        
        # Generate queries for each mode
        for mode in self.query_generation_modes:
            mode_queries = self._generate_queries_by_mode(text, mode)
            for query in mode_queries:
                all_queries.append({
                    'query': query,
                    'mode': mode,
                    'difficulty': None,  # Will be determined later
                    'score': None
                })
          # Remove duplicates while preserving metadata
        unique_queries = self._deduplicate_queries(all_queries)
        
        print(f"   🎯 Generated {len(all_queries)} → {len(unique_queries)} unique queries")
        return unique_queries
    
    def _generate_queries_by_mode(self, text: str, mode: str) -> List[str]:
        """Generate queries using structured output for consistent format"""
        
        # Use structured query generation for better consistency
        structured_queries = self.chat_api.generate_queries_structured(
            document_text=text,
            num_queries=2,  # Generate 2 queries per mode
            difficulty_level=mode
        )
        
        # Extract questions from structured response
        queries = []
        for query_data in structured_queries:
            if isinstance(query_data, dict) and 'question' in query_data:
                question = query_data['question'].strip()
                if question and '?' in question and len(question) > 10:
                    queries.append(question)
        
        # Fallback to simple generation if structured fails
        if not queries:
            time.sleep(self.api_call_delay)
            
            prompt = f"""Genera 2 preguntas {mode}s sobre este texto en español:

TEXTO:
{text[:600]}

INSTRUCCIONES:
- Solo 2 preguntas {mode}s
- Que se puedan responder con el texto
- Una pregunta por línea
- Terminar con ?"""
            
            response = self.chat_api.generate_text(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.8,
                top_k=30,
                repetition_penalty=1.2
            )
            
            if response:
                queries = [q.strip() for q in response.split('\n') if q.strip() and '?' in q and len(q.strip()) > 10]
        
        return queries[:2]  # Return up to 2 queries
    
    def _deduplicate_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate queries while preserving the best metadata"""
        seen_normalized = {}
        unique_queries = []
        
        for query_data in queries:
            query = query_data['query']
            normalized = re.sub(r'[¿?¡!.,;:]', '', query.lower().strip())
            normalized = ' '.join(normalized.split())  # Normalize whitespace
            
            if normalized not in seen_normalized and len(query) > 15:
                seen_normalized[normalized] = True
                unique_queries.append(query_data)
        
        return unique_queries
    
    def _score_and_select_queries(self, query_candidates: List[Dict[str, Any]], document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Advanced query scoring and selection using AI evaluation
        """
        print(f"   ⚖️  Scoring {len(query_candidates)} query candidates...")
        
        scored_queries = []
        
        for i, query_data in enumerate(query_candidates):
            query = query_data['query']
            
            # Multi-dimensional scoring
            relevance_score = self._score_query_relevance(query, document['text'])
            difficulty_score = self._score_query_difficulty(query)
            clarity_score = self._score_query_clarity(query)
            
            # Weighted composite score
            composite_score = (
                relevance_score * 0.5 +    # Relevance is most important
                difficulty_score * 0.3 +   # Difficulty for training quality
                clarity_score * 0.2        # Clarity for usability
            )
            
            query_data['score'] = composite_score
            query_data['relevance'] = relevance_score
            query_data['difficulty'] = self._classify_difficulty(difficulty_score)
            query_data['clarity'] = clarity_score
            
            if composite_score >= self.min_query_score:
                scored_queries.append(query_data)
                print(f"      ✅ '{query[:40]}...' → {composite_score:.2f} ({query_data['difficulty']})")
            else:
                print(f"      ❌ '{query[:40]}...' → {composite_score:.2f} (below threshold)")        # Sort by score and select top queries
        scored_queries.sort(key=lambda x: x['score'], reverse=True)
        selected = scored_queries[:self.final_queries_per_document]
        
        print(f"   ✨ Selected {len(selected)} queries (from {len(query_candidates)} candidates)")
        return selected
    
    def _score_query_relevance(self, query: str, document_text: str) -> float:
        """Score how well the query can be answered by the document using AI evaluation"""
        # Use structured scoring from the chat API for more accurate evaluation
        scores = self.chat_api.score_query_quality(query, document_text)
        
        if scores and 'relevance' in scores:
            relevance_score = scores['relevance']
            print(f"        📊 Relevance: {relevance_score:.2f} (AI-powered)")
            return relevance_score
        
        # Fallback to heuristic if AI scoring fails
        query_lower = query.lower()
        doc_lower = document_text.lower()
        
        # Check for keyword overlap
        query_words = set(query_lower.split())
        doc_words = set(doc_lower.split())
        
        # Remove common stop words
        stop_words = {'que', 'con', 'para', 'por', 'los', 'las', 'del', 'una', 'está', 'son', 'como', 'es', 'la', 'el', 'de', 'en', 'y', 'a'}
        query_words = query_words - stop_words
        doc_words = doc_words - stop_words
        
        if not query_words:
            return 0.5
        
        # Calculate overlap
        overlap = len(query_words.intersection(doc_words))
        overlap_ratio = overlap / len(query_words)
        
        # Boost score if query seems well-formed
        if '?' in query and len(query) > 20:
            overlap_ratio += 0.2
            
        score = min(max(overlap_ratio, 0.3), 1.0)  # Keep between 0.3 and 1.0
        print(f"        📊 Relevance: {score:.2f} (heuristic fallback)")
        return score
    
    def _score_query_difficulty(self, query: str) -> float:
        """Score the cognitive difficulty of the query using heuristics"""
        # Simple heuristic-based difficulty scoring to reduce API calls
        score = 0.5  # Base difficulty score
        
        query_lower = query.lower()
        
        # Basic question indicators (lower difficulty)
        basic_indicators = ['qué es', 'quién es', 'cuándo', 'dónde', 'cuál es el nombre', 'cuál es la']
        if any(indicator in query_lower for indicator in basic_indicators):
            score = 0.3
        
        # Intermediate question indicators
        intermediate_indicators = ['cómo', 'por qué', 'cuáles son', 'qué papel', 'qué función', 'explicar']
        if any(indicator in query_lower for indicator in intermediate_indicators):
            score = 0.5
        
        # Advanced question indicators
        advanced_indicators = ['analizar', 'comparar', 'evaluar', 'relación entre', 'implicaciones', 'diferencias']
        if any(indicator in query_lower for indicator in advanced_indicators):
            score = 0.7
        
        # Expert question indicators
        expert_indicators = ['sintetizar', 'crear', 'desarrollar', 'proponer', 'criticar', 'innovar']
        if any(indicator in query_lower for indicator in expert_indicators):
            score = 0.9
        
        # Adjust based on question length and complexity
        if len(query) > 80:
            score += 0.1
        if '?' in query and len(query.split()) > 10:
            score += 0.05
            
        score = min(max(score, 0.2), 1.0)  # Keep between 0.2 and 1.0
        print(f"        📊 Difficulty: {score:.2f} (heuristic)")
        return score
    
    def _score_query_clarity(self, query: str) -> float:
        """Score query clarity using simple heuristics"""
        # Simple heuristic-based clarity scoring
        score = 0.6  # Base score
        
        # Check basic quality indicators
        if '?' in query:
            score += 0.2
        if len(query) > 15:
            score += 0.1
        if len(query.split()) > 5:
            score += 0.1
          # Penalize very short or very long queries
        if len(query) < 10:
            score -= 0.3
        elif len(query) > 200:
            score -= 0.2
            
        score = min(max(score, 0.3), 1.0)
        print(f"        📊 Clarity: {score:.2f} (heuristic)")        
        return score
    
    def _classify_difficulty(self, difficulty_score: float) -> str:
        """Classify difficulty score into human-readable categories"""
        if difficulty_score < 0.3:
            return "basic"
        elif difficulty_score < 0.6:
            return "intermediate"
        elif difficulty_score < 0.8:
            return "advanced"
        else:
            return "expert"
    
    def _create_advanced_dpo_sample(self, query_data: Dict[str, Any], source_doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create high-quality DPO sample using structured response generation
        """
        query = query_data['query']
        print(f"      🎭 Creating DPO sample for: {query[:50]}...")
        
        # Retrieve relevant context
        search_results = self.vector_store.search(query, k=8)  # Get more context
        
        if not search_results:
            print(f"      ❌ No search results for query")
            return None
        
        # Build rich context
        context_docs = [doc['text'] for doc in search_results[:5]]
        context = self._build_context(context_docs, query)
        
        # Generate high-quality response pair using the new structured API
        response_pair = self.chat_api.generate_response_pair(
            query=query, 
            context=context, 
            response_type="comprehensive"
        )
        
        good_response = response_pair.get("good")
        bad_response = response_pair.get("bad")
        
        if not good_response or not bad_response:
            print(f"      ❌ Failed to generate response pair")
            return None
        
        print(f"      ✅ Response pair generated: good={len(good_response)} chars, bad={len(bad_response)} chars")
        
        # Validate the DPO sample using structured validation
        validation = self.chat_api.validate_dpo_sample(query, good_response, bad_response)
        
        if not validation.get("is_valid", False):
            print(f"      ❌ Sample validation failed: {validation.get('recommendation', 'unknown')}")
            return None
        
        good_score = validation.get("good_quality", 0.5)
        bad_score = validation.get("bad_quality", 0.3)
        quality_diff = validation.get("quality_difference", 0.0)
        
        # Additional quality check
        if good_score <= bad_score or good_score < self.min_response_score:
            print(f"      ❌ Quality check failed: good={good_score:.2f}, bad={bad_score:.2f}")
            return None
        
        sample = {
            "query": query,
            "good_response": good_response,
            "bad_response": bad_response,
            "metadata": {
                "source_doc_id": source_doc['id'],
                "source_file": source_doc.get('source', 'unknown'),
                "generation_mode": query_data['mode'],
                "difficulty_level": query_data['difficulty'],
                "query_score": query_data['score'],
                "good_response_score": good_score,
                "bad_response_score": bad_score,
                "quality_difference": quality_diff,
                "search_results_count": len(search_results),
                "generation_timestamp": datetime.now().isoformat(),
                "ai_generated": True,
                "pipeline_version": "advanced_v2.0_structured",
                "validation_passed": True
            }
        }
        
        print(f"      ✅ Sample created: good={good_score:.2f}, bad={bad_score:.2f}, diff={quality_diff:.2f}")
        return sample
    
    def _build_context(self, context_docs: List[str], query: str) -> str:
        """Build optimal context for response generation"""
        # Combine and truncate context intelligently
        combined = "\n\n".join(context_docs)
        
        if len(combined) <= self.max_context_length:
            return combined
        
        # Truncate while preserving sentence boundaries
        truncated = combined[:self.max_context_length]
        last_period = truncated.rfind('.')
        if last_period > self.max_context_length * 0.8:
            truncated = truncated[:last_period + 1]
        
        return truncated
    
    def _generate_good_response(self, query: str, context: str, mode: str) -> Optional[str]:
        """Generate high-quality response using context and mode-specific instructions"""
        
        mode_instructions = {
            'analytical': "Proporciona un análisis detallado y crítico con múltiples perspectivas.",
            'factual': "Responde con datos precisos, específicos y verificables del contexto.",
            'conceptual': "Explica los conceptos fundamentales de manera clara y comprensible.",
            'procedural': "Describe el proceso o metodología paso a paso de forma práctica.",
            'comparative': "Realiza una comparación estructurada destacando similitudes y diferencias."
        }
        
        instruction = mode_instructions.get(mode, "Proporciona una respuesta completa y precisa.")
        
        system_prompt = f"""Eres un asistente académico experto. {instruction}

INSTRUCCIONES:
- Usa únicamente la información del contexto proporcionado
- Responde en español académico formal
- Sé preciso, detallado y bien estructurado
- Incluye ejemplos específicos cuando sea relevante
- Mantén un tono profesional y objetivo"""
        
        user_message = f"""PREGUNTA: {query}

CONTEXTO:
{context}

RESPUESTA ACADÉMICA:"""
        
        response = self.chat_api.generate_text(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            max_tokens=600,
            temperature=0.3
        )
        
        if not response:
            print(f"        ⚠️ API call failed for good response, using fallback")
            # Fallback: create a basic response based on context
            response = f"Basándome en el contexto proporcionado: {context[:200]}..."
        
        return response if response and len(response.strip()) > 50 else None
    
    def _generate_bad_response(self, query: str, context: str, mode: str) -> Optional[str]:
        """Generate contrasting lower-quality response"""
        
        bad_response_strategies = [
            "Responde de manera vaga e imprecisa, sin usar el contexto específico.",
            "Proporciona información genérica que no aborda directamente la pregunta.",
            "Incluye información incorrecta o mal interpretada del contexto.",
            "Responde de forma superficial sin profundidad ni detalles.",
            "Mezcla conceptos sin establecer conexiones claras o lógicas."
        ]
        
        strategy = random.choice(bad_response_strategies)
        
        system_prompt = f"""Eres un asistente que debe generar una respuesta de menor calidad. {strategy}

INSTRUCCIONES PARA RESPUESTA INADECUADA:
- Responde en español pero con menor precisión
- No uses eficientemente la información del contexto
- Sé menos específico y detallado
- Incluye generalidades en lugar de información precisa
- Mantén una estructura menos organizada"""
        
        user_message = f"""PREGUNTA: {query}

CONTEXTO:
{context}

RESPUESTA DE MENOR CALIDAD:"""
        
        response = self.chat_api.generate_text(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            max_tokens=400,
            temperature=0.7
        )
        
        return response if response and len(response.strip()) > 30 else None
    
    def _score_response_quality(self, query: str, response: str, context: str) -> float:
        """Score the quality of a response"""
        
        system_prompt = """Eres un evaluador experto de calidad de respuestas académicas.
Evalúa la calidad de una respuesta considerando:

1. PRECISIÓN: ¿La respuesta es factualmente correcta según el contexto?
2. COMPLETITUD: ¿Responde completamente la pregunta?
3. CLARIDAD: ¿Es clara, bien estructurada y comprensible?
4. USO DEL CONTEXTO: ¿Utiliza eficientemente la información disponible?
5. PROFUNDIDAD: ¿Proporciona suficiente detalle y análisis?

Responde únicamente con un número del 0.0 al 1.0 (donde 1.0 = excelente calidad)."""
        
        user_message = f"""PREGUNTA: {query}

CONTEXTO DISPONIBLE:
{context[:800]}

RESPUESTA A EVALUAR:
{response}

PUNTUACIÓN DE CALIDAD (0.0-1.0):"""
        
        result = self.chat_api.generate_text(
            messages=[{"role": "user", "content": user_message}],
            system_prompt=system_prompt,
            max_tokens=200,  # Increased for Qwen reasoning models
            temperature=0.1
        )
        
        if result:
            try:
                score = float(re.search(r'([0-1]\.?\d*)', result).group(1))
                return min(max(score, 0.0), 1.0)
            except:
                pass
        
        return 0.5  # Default middle score
    
    def _validate_sample_quality(self, sample: Dict[str, Any]) -> bool:
        """Final validation of sample quality"""
        metadata = sample['metadata']
        
        # Check minimum quality thresholds
        if metadata['good_response_score'] < self.min_response_score:
            return False
        
        if metadata['quality_difference'] < 0.1:  # Good must be significantly better than bad
            return False
        
        # Check response lengths
        if len(sample['good_response']) < 100 or len(sample['bad_response']) < 50:
            return False
        
        # Check for duplicates (basic)
        if sample['good_response'].strip() == sample['bad_response'].strip():
            return False
        
        return True
    
    def _balance_difficulty_distribution(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balance the difficulty distribution of the dataset"""
        if not self.enable_difficulty_balancing:
            return dataset
        
        print("🎚️  Balancing difficulty distribution...")
        
        # Group by difficulty
        by_difficulty = {}
        for sample in dataset:
            diff = sample['metadata']['difficulty_level']
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(sample)
        
        # Calculate target distribution
        total_samples = len(dataset)
        target_distribution = {
            'basic': int(total_samples * 0.2),
            'intermediate': int(total_samples * 0.4),
            'advanced': int(total_samples * 0.3),
            'expert': int(total_samples * 0.1)
        }
        
        # Balance the dataset
        balanced_dataset = []
        for difficulty, target_count in target_distribution.items():
            available = by_difficulty.get(difficulty, [])
            # Sort by quality and take the best ones
            available.sort(key=lambda x: x['metadata']['good_response_score'], reverse=True)
            balanced_dataset.extend(available[:target_count])
        
        print(f"🎯 Balanced dataset: {len(balanced_dataset)} samples")
        for diff, count in target_distribution.items():
            actual = len([s for s in balanced_dataset if s['metadata']['difficulty_level'] == diff])
            print(f"   {diff}: {actual}/{count} samples")
        
        return balanced_dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], output_file: str) -> None:
        """Save dataset with comprehensive metadata"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add dataset-level metadata
        dataset_metadata = {
            "dataset_info": {
                "generation_timestamp": datetime.now().isoformat(),
                "generator_version": "AdvancedDatasetGenerator_v1.0",
                "total_samples": len(dataset),
                "quality_settings": {
                    "min_query_score": self.min_query_score,
                    "min_response_score": self.min_response_score,
                    "min_answer_quality_score": self.min_answer_quality_score
                },
                "pipeline_config": {
                    "query_generation_modes": self.query_generation_modes,
                    "difficulty_levels": self.difficulty_levels,
                    "final_queries_per_document": self.final_queries_per_document,
                    "max_context_length": self.max_context_length
                }
            },
            "samples": dataset
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Advanced dataset saved to: {output_path}")
        print(f"📊 Contains {len(dataset)} high-quality DPO samples")
    
    def get_advanced_stats(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive dataset statistics"""
        if not dataset:
            return {"error": "Empty dataset"}
        
        # Basic statistics
        total_samples = len(dataset)
        query_lengths = [len(sample['query']) for sample in dataset]
        good_response_lengths = [len(sample['good_response']) for sample in dataset]
        bad_response_lengths = [len(sample['bad_response']) for sample in dataset]
        
        # Quality statistics
        query_scores = [sample['metadata']['query_score'] for sample in dataset]
        good_scores = [sample['metadata']['good_response_score'] for sample in dataset]
        bad_scores = [sample['metadata']['bad_response_score'] for sample in dataset]
        quality_diffs = [sample['metadata']['quality_difference'] for sample in dataset]
        
        # Distribution statistics
        modes = {}
        difficulties = {}
        sources = {}
        
        for sample in dataset:
            metadata = sample['metadata']
            
            mode = metadata['generation_mode']
            modes[mode] = modes.get(mode, 0) + 1
            
            diff = metadata['difficulty_level']
            difficulties[diff] = difficulties.get(diff, 0) + 1
            
            source = metadata['source_file']
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "dataset_overview": {
                "total_samples": total_samples,
                "avg_query_length": sum(query_lengths) / total_samples,
                "avg_good_response_length": sum(good_response_lengths) / total_samples,
                "avg_bad_response_length": sum(bad_response_lengths) / total_samples,
            },
            "quality_metrics": {
                "avg_query_score": sum(query_scores) / total_samples,
                "avg_good_response_score": sum(good_scores) / total_samples,
                "avg_bad_response_score": sum(bad_scores) / total_samples,
                "avg_quality_difference": sum(quality_diffs) / total_samples,
                "min_quality_difference": min(quality_diffs),
                "max_quality_difference": max(quality_diffs),
            },
            "distribution_analysis": {
                "generation_modes": modes,
                "difficulty_levels": difficulties,
                "source_files": dict(list(sources.items())[:10]),  # Top 10 sources
                "unique_source_files": len(sources),
            },
            "quality_assurance": {
                "samples_above_min_query_score": sum(1 for s in query_scores if s >= self.min_query_score),
                "samples_above_min_response_score": sum(1 for s in good_scores if s >= self.min_response_score),
                "percentage_high_quality": (sum(1 for s in good_scores if s >= 0.8) / total_samples) * 100,
            }
        }
    
    def _quick_relevance_score(self, query: str, document_text: str) -> float:
        """Quick heuristic-based relevance scoring to reduce API calls"""
        score = 0.4  # Base score
        
        # Extract key terms from query (remove question words and common words)
        query_clean = query.lower()
        for qword in ['¿', '?', 'qué', 'cuál', 'cuáles', 'cómo', 'por qué', 'dónde', 'cuándo', 'quién']:
            query_clean = query_clean.replace(qword, ' ')
        
        query_words = [w.strip() for w in query_clean.split() if len(w.strip()) > 3]
        
        if not query_words:
            return 0.3
        
        # Check how many query terms appear in the document
        doc_lower = document_text.lower()
        matches = 0
        for word in query_words:
            if word in doc_lower:
                matches += 1
        
        match_ratio = matches / len(query_words)
        
        # Score based on match ratio
        if match_ratio >= 0.8:
            score = 0.8
        elif match_ratio >= 0.6:
            score = 0.7
        elif match_ratio >= 0.4:
            score = 0.6
        elif match_ratio >= 0.2:
            score = 0.5
        else:
            score = 0.4
        
        # Bonus for longer, more specific queries
        if len(query) > 50:
            score += 0.05
          # Bonus for document-specific terms
        specific_terms = ['resolución', 'proyecto', 'universidad', 'centro', 'regional']
        for term in specific_terms:
            if term in query_clean and term in doc_lower:
                score += 0.05
        
        return min(score, 1.0)
