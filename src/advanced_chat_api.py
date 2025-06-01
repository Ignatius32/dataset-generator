"""
Simple Chat API - Functional replacement for AdvancedChatAPI
"""

import requests
import time
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from .config import Config

class AdvancedChatAPI:
    """
    Simple but functional chat API that works with the gradient API
    """
    
    def __init__(self, max_retries: int = None, base_delay: float = None):
        Config.validate()
        self.api_key = Config.GS_API_KEY
        self.api_url = Config.CHAT_API_URL
        self.model = Config.CHAT_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"        }
        self.max_retries = max_retries or Config.MAX_RETRIES
        self.base_delay = base_delay or Config.BASE_RETRY_DELAY
        self.timeout = Config.REQUEST_TIMEOUT
        
        print("🚀 Chat API initialized")
        
    def generate_text_robust(self, messages: List[Dict[str, str]], 
                           max_tokens: int = 800,  # Increased default for Qwen reasoning models
                           temperature: float = 0.7,
                           system_prompt: str = None,
                           max_attempts: int = 3) -> Optional[str]:
        """
        Robust text generation with multiple attempts
        """
        best_response = None
        best_score = 0.0
        
        for attempt in range(max_attempts):
            response = self._make_api_call(messages, max_tokens, temperature, system_prompt)
            
            if response:
                # Basic quality check
                quality_score = self._assess_response_quality(response, messages[-1]['content'])
                
                if quality_score > best_score:
                    best_response = response
                    best_score = quality_score
                
                # If we get a high-quality response, use it
                if quality_score > 0.5:  # Lowered threshold
                    break
                      
            # Slight temperature adjustment for next attempt
            temperature = min(temperature + 0.1, 1.0)
        
        return best_response
    
    def generate_text(self, messages: List[Dict[str, str]], 
                     max_tokens: int = 800,  # Increased default for Qwen reasoning models
                     temperature: float = 0.7,
                     system_prompt: str = None,
                     **api_params) -> Optional[str]:
        """
        Generate text using robust generation (compatibility method)
        """
        return self.generate_text_robust(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            max_attempts=3,
            **api_params
        )
    
    def generate_structured_text(self, messages: List[Dict[str, str]], 
                               response_format: Dict[str, Any] = None,
                               guided_json: Any = None,  # Can be string or dict
                               guided_regex: str = None,
                               guided_choice: List[str] = None,
                               guided_grammar: str = None,
                               guided_decoding_backend: str = None,
                               guided_whitespace_pattern: str = None,
                               max_tokens: int = 800,
                               temperature: float = 0.7,
                               top_k: int = None,
                               min_p: float = None,
                               repetition_penalty: float = None,
                               length_penalty: float = None,
                               use_beam_search: bool = None,
                               min_tokens: int = None,
                               stop_token_ids: List[int] = None,
                               include_stop_str_in_output: bool = None,
                               ignore_eos: bool = None,
                               skip_special_tokens: bool = None,
                               spaces_between_special_tokens: bool = None,
                               add_special_tokens: bool = None,
                               truncate_prompt_tokens: int = None,
                               allowed_token_ids: List[int] = None,
                               prompt_logprobs: int = None,
                               priority: int = None,
                               logits_processors: List = None,
                               return_tokens_as_token_ids: bool = None,
                               system_prompt: str = None) -> Optional[str]:
        """
        Generate structured text with guided output formats
        Perfect for dataset generation where consistency is crucial
        
        Now supports all documented Gradient Sur API parameters
        """
        # Build API parameters for structured generation
        api_params = {}
        
        # Guided generation parameters
        if response_format:
            api_params['response_format'] = response_format
        if guided_json:
            api_params['guided_json'] = guided_json
        if guided_regex:
            api_params['guided_regex'] = guided_regex
        if guided_choice:
            api_params['guided_choice'] = guided_choice
        if guided_grammar:
            api_params['guided_grammar'] = guided_grammar
        if guided_decoding_backend:
            api_params['guided_decoding_backend'] = guided_decoding_backend
        if guided_whitespace_pattern:
            api_params['guided_whitespace_pattern'] = guided_whitespace_pattern
            
        # Sampling parameters
        if top_k is not None:
            api_params['top_k'] = top_k
        if min_p is not None:
            api_params['min_p'] = min_p
        if repetition_penalty is not None:
            api_params['repetition_penalty'] = repetition_penalty
        if length_penalty is not None:
            api_params['length_penalty'] = length_penalty
        if use_beam_search is not None:
            api_params['use_beam_search'] = use_beam_search
        if min_tokens is not None:
            api_params['min_tokens'] = min_tokens
        if stop_token_ids is not None:
            api_params['stop_token_ids'] = stop_token_ids
        if include_stop_str_in_output is not None:
            api_params['include_stop_str_in_output'] = include_stop_str_in_output
        if ignore_eos is not None:
            api_params['ignore_eos'] = ignore_eos
        if skip_special_tokens is not None:
            api_params['skip_special_tokens'] = skip_special_tokens
        if spaces_between_special_tokens is not None:
            api_params['spaces_between_special_tokens'] = spaces_between_special_tokens
            
        # Advanced parameters
        if add_special_tokens is not None:
            api_params['add_special_tokens'] = add_special_tokens
        if truncate_prompt_tokens is not None:
            api_params['truncate_prompt_tokens'] = truncate_prompt_tokens
        if allowed_token_ids is not None:
            api_params['allowed_token_ids'] = allowed_token_ids
        if prompt_logprobs is not None:
            api_params['prompt_logprobs'] = prompt_logprobs
        if priority is not None:
            api_params['priority'] = priority
        if logits_processors is not None:
            api_params['logits_processors'] = logits_processors
        if return_tokens_as_token_ids is not None:
            api_params['return_tokens_as_token_ids'] = return_tokens_as_token_ids
            
        return self._make_api_call(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            **api_params
        )
    
    def _make_api_call(self, messages: List[Dict[str, str]], 
                      max_tokens: int, temperature: float, 
                      system_prompt: str = None, **api_params) -> Optional[str]:
        """Core API call with robust error handling and enhanced parameter support"""
        
        # Add system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add any additional API parameters
        data.update(api_params)
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(                    self.api_url,
                    headers=self.headers,
                    json=data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        choice = result["choices"][0]
                        message = choice["message"]
                          # Get primary content
                        content = message.get("content")
                        reasoning_content = message.get("reasoning_content")
                        
                        # Handle Qwen model responses - they often have both content and reasoning
                        if content:
                            # Primary response is in content field - this is the expected case
                            pass
                        elif reasoning_content:
                            # Fallback: extract useful content from reasoning when content is empty
                            print(f"🧠 Using reasoning_content as fallback")
                            reasoning = reasoning_content.strip()
                            
                            # For Qwen models, try to extract the final response from reasoning
                            if reasoning:
                                lines = reasoning.split('\n')
                                # Look for the actual response content, usually at the end
                                useful_lines = []
                                for line in reversed(lines):
                                    line = line.strip()
                                    # Skip meta-commentary and reasoning steps
                                    if (line and 
                                        not line.startswith('I need to') and 
                                        not line.startswith('Let me') and 
                                        not line.startswith('The user') and
                                        not line.startswith('First,') and
                                        not line.startswith('Okay,') and
                                        len(line) > 10):
                                        useful_lines.insert(0, line)
                                        if len(useful_lines) >= 2:
                                            break
                                
                                if useful_lines:
                                    content = ' '.join(useful_lines)
                                else:
                                    # Last resort: use a portion of reasoning content
                                    content = reasoning[-200:] if len(reasoning) > 200 else reasoning
                        
                        # Strip whitespace and ensure we have valid content
                        if content:
                            content = content.strip()
                            if content:  # Only return if we have actual content after stripping
                                return content
                                
                    # If we get here, we got a 200 but with empty content
                    print(f"⚠️  Got 200 response but with empty content: {result}")
                    return None
                        
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.base_delay * (2 ** attempt) + (attempt * 2)  # More aggressive backoff
                    print(f"⏳ Rate limited, waiting {wait_time:.1f}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    print(f"⚠️  API error {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                print(f"⏰ Request timeout (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                print(f"🌐 Request error: {e} (attempt {attempt + 1})")
            except Exception as e:
                print(f"❌ Unexpected error: {e} (attempt {attempt + 1})")
            
            # Wait before retry with exponential backoff
            if attempt < self.max_retries:
                wait_time = min(self.base_delay * (2 ** attempt) + attempt, Config.MAX_RETRY_DELAY)
                print(f"⏳ Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
        
        print(f"💀 All retries failed for chat API request")
        return None
    
    def _assess_response_quality(self, response: str, prompt: str) -> float:
        """Quick quality assessment of generated response"""
        if not response or len(response.strip()) < 5:  # Very lenient
            return 0.0
        
        score = 0.5  # Base score
        
        # Length appropriateness - very lenient
        if 10 <= len(response) <= 5000:
            score += 0.3
        
        # Language detection (prefer Spanish responses for our use case)
        spanish_indicators = ['que', 'con', 'para', 'por', 'los', 'las', 'del', 'una', 'está', 'son', 'hola', 'como']
        if any(word in response.lower() for word in spanish_indicators):
            score += 0.2
        
        return min(score, 1.0)

    # Placeholder methods for compatibility with existing code
    def generate_specialized_queries(self, document_text: str, mode: str, num_queries: int = 4) -> List[str]:
        """Simple query generation for compatibility"""
        prompt = f"""Genera exactamente {num_queries} preguntas claras y específicas sobre el siguiente texto.
        
Texto: {document_text[:500]}

Por favor, responde ÚNICAMENTE con las preguntas, una por línea, numeradas del 1 al {num_queries}. 
No incluyas explicaciones adicionales.

Ejemplo de formato:
1. ¿Pregunta uno?
2. ¿Pregunta dos?
3. ¿Pregunta tres?"""

        messages = [{"role": "user", "content": prompt}]
        response = self.generate_text(messages, max_tokens=500)
        
        if response:
            # Extract questions more reliably
            lines = response.split('\n')
            queries = []
            for line in lines:
                line = line.strip()
                # Look for numbered questions or questions with question marks
                if line and ('?' in line):
                    # Clean up the line
                    line = re.sub(r'^\d+\.?\s*', '', line)  # Remove numbering
                    line = re.sub(r'^-\s*', '', line)      # Remove dashes
                    line = re.sub(r'^¿\s*', '¿', line)     # Fix question mark spacing
                    if len(line) > 10 and line.count('?') > 0:
                        queries.append(line)
            
            # If we didn't get enough questions, generate some default ones
            if len(queries) < num_queries:
                default_queries = [
                    "¿Cuál es el tema principal del documento?",
                    "¿Qué información importante se presenta en el texto?",
                    "¿Cuáles son los puntos clave mencionados?",
                    "¿Qué aspectos relevantes se destacan en el contenido?"
                ]
                for i, default_q in enumerate(default_queries):
                    if len(queries) < num_queries:
                        queries.append(default_q)
            
            return queries[:num_queries]
        
        # Fallback if API fails
        return [
            "¿Cuál es el contenido principal del documento?",
            "¿Qué información relevante se presenta?",
            "¿Cuáles son los aspectos más importantes mencionados?"
        ][:num_queries]
        
        return []
    
    def generate_text_with_reasoning(self, messages: List[Dict[str, str]], 
                                   max_tokens: int = 800,
                                   temperature: float = 0.7,
                                   system_prompt: str = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate text and return both the content and reasoning (if available)
        
        Returns:
            Tuple of (content, reasoning_content) - reasoning_content may be None
        """
        # Add system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    message = choice["message"]
                    
                    content = message.get("content", "").strip()
                    reasoning = message.get("reasoning_content", "").strip()
                    
                    return (content or None, reasoning or None)
                    
        except Exception as e:
            print(f"❌ Error in generate_text_with_reasoning: {e}")
        
        return (None, None)

    # =====================================================================
    # SPECIALIZED METHODS FOR DPO DATASET GENERATION
    # =====================================================================
    
    def generate_queries_structured(self, document_text: str, num_queries: int = 5, 
                                   difficulty_level: str = "mixed") -> List[Dict[str, Any]]:
        """
        Generate structured queries for DPO dataset with consistent format
        Uses guided JSON to ensure proper structure
        """
        # Define the JSON schema for structured query output
        query_schema = {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {
                        "type": "object", 
                        "properties": {
                            "question": {"type": "string"},
                            "difficulty": {"type": "string", "enum": ["basic", "intermediate", "advanced"]},
                            "type": {"type": "string", "enum": ["factual", "analytical", "conceptual", "procedural"]},
                            "keywords": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["question", "difficulty", "type"]
                    }
                }
            },
            "required": ["queries"]
        }
        
        prompt = f"""Analiza el siguiente texto y genera exactamente {num_queries} preguntas de alta calidad para entrenamiento de modelos de IA.

TEXTO A ANALIZAR:
{document_text[:1000]}

INSTRUCCIONES:
- Crea preguntas variadas que cubran diferentes aspectos del texto
- Incluye diferentes niveles de dificultad: basic, intermediate, advanced
- Tipos de preguntas: factual, analytical, conceptual, procedural
- Cada pregunta debe ser clara, específica y responder directamente al contenido

Responde ÚNICAMENTE en el formato JSON especificado, sin texto adicional."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.generate_structured_text(
            messages=messages,
            guided_json=query_schema,
            temperature=0.8,  # Higher creativity for diverse queries
            top_k=40,
            repetition_penalty=1.2,
            min_tokens=50,
            max_tokens=800
        )
        
        if response:
            try:
                data = json.loads(response)
                return data.get("queries", [])
            except json.JSONDecodeError:
                print("⚠️ Failed to parse JSON response for query generation")
                
        return self._generate_fallback_queries(document_text, num_queries)
    
    def score_query_quality(self, query: str, document_text: str) -> Dict[str, float]:
        """
        Score query quality using structured output for consistent evaluation
        """
        scoring_schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "object",
                    "properties": {
                        "relevance": {"type": "number", "minimum": 0, "maximum": 1},
                        "clarity": {"type": "number", "minimum": 0, "maximum": 1},
                        "difficulty": {"type": "number", "minimum": 0, "maximum": 1},
                        "answerability": {"type": "number", "minimum": 0, "maximum": 1},
                        "overall": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["relevance", "clarity", "difficulty", "answerability", "overall"]
                },
                "reasoning": {"type": "string"}
            },
            "required": ["scores"]
        }
        
        prompt = f"""Evalúa la calidad de esta pregunta en relación al texto proporcionado.

PREGUNTA: {query}

TEXTO DE REFERENCIA:
{document_text[:800]}

CRITERIOS DE EVALUACIÓN (escala 0.0 a 1.0):
- relevance: ¿Qué tan relevante es la pregunta al contenido del texto?
- clarity: ¿Qué tan clara y bien formulada está la pregunta?
- difficulty: ¿Qué nivel de dificultad cognitiva requiere? (0.0=básico, 1.0=experto)
- answerability: ¿Se puede responder bien usando el texto proporcionado?
- overall: Puntuación general de calidad

Proporciona puntuaciones objetivas y una breve justificación."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.generate_structured_text(
            messages=messages,
            guided_json=scoring_schema,
            temperature=0.3,  # Lower temperature for consistent scoring
            top_k=20,
            repetition_penalty=1.1,
            min_tokens=30
        )
        
        if response:
            try:
                data = json.loads(response)
                return data.get("scores", {})
            except json.JSONDecodeError:
                pass
                
        # Fallback to heuristic scoring
        return self._heuristic_quality_score(query, document_text)
    
    def generate_response_pair(self, query: str, context: str, 
                              response_type: str = "comprehensive") -> Dict[str, str]:
        """
        Generate a good/bad response pair for DPO training
        Uses different parameters to create contrasting quality
        """
        # Generate good response with optimal parameters
        good_prompt = f"""Responde a la siguiente pregunta de manera completa, precisa y útil usando el contexto proporcionado.

PREGUNTA: {query}

CONTEXTO:
{context}

INSTRUCCIONES:
- Proporciona una respuesta completa y bien estructurada
- Usa información específica del contexto
- Mantén un tono profesional y claro
- Incluye detalles relevantes y ejemplos cuando sea apropiado"""

        good_response = self.generate_structured_text(
            messages=[{"role": "user", "content": good_prompt}],
            temperature=0.7,
            top_k=30,
            repetition_penalty=1.1,
            min_tokens=100,
            max_tokens=500
        )
        
        # Generate bad response with parameters that encourage poor quality
        bad_prompt = f"""Responde brevemente a esta pregunta:

PREGUNTA: {query}

CONTEXTO:
{context[:200]}

Responde de manera muy breve."""

        bad_response = self.generate_structured_text(
            messages=[{"role": "user", "content": bad_prompt}],
            temperature=1.0,  # High temperature for inconsistency
            top_k=5,  # Limited vocabulary
            repetition_penalty=0.8,  # Allow repetition
            max_tokens=150  # Limited length
        )
        
        return {
            "good": good_response or "Respuesta no disponible.",
            "bad": bad_response or "Sin respuesta."
        }
    
    def validate_dpo_sample(self, query: str, good_response: str, 
                           bad_response: str) -> Dict[str, Any]:
        """
        Validate a complete DPO sample using structured evaluation
        """
        validation_schema = {
            "type": "object",
            "properties": {
                "validation": {
                    "type": "object",
                    "properties": {
                        "is_valid": {"type": "boolean"},
                        "good_quality": {"type": "number", "minimum": 0, "maximum": 1},
                        "bad_quality": {"type": "number", "minimum": 0, "maximum": 1},
                        "quality_difference": {"type": "number", "minimum": -1, "maximum": 1},
                        "issues": {"type": "array", "items": {"type": "string"}},
                        "recommendation": {"type": "string", "enum": ["accept", "reject", "revise"]}
                    },
                    "required": ["is_valid", "good_quality", "bad_quality", "recommendation"]
                }
            },
            "required": ["validation"]
        }
        
        prompt = f"""Evalúa este sample de entrenamiento DPO y determina si es válido para entrenar un modelo.

PREGUNTA: {query}

RESPUESTA BUENA:
{good_response}

RESPUESTA MALA:
{bad_response}

CRITERIOS:
- La respuesta "buena" debe ser notablemente mejor que la "mala"
- Ambas respuestas deben estar relacionadas con la pregunta
- Debe haber una diferencia clara en calidad
- La respuesta buena debe ser informativa y precisa

Evalúa objetivamente y proporciona puntuaciones de calidad (0.0-1.0)."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.generate_structured_text(
            messages=messages,
            guided_json=validation_schema,
            temperature=0.2,  # Very low for consistent validation
            top_k=10,
            min_tokens=50
        )
        
        if response:
            try:
                data = json.loads(response)
                return data.get("validation", {})
            except json.JSONDecodeError:
                pass
                
        # Fallback validation
        return self._heuristic_validation(query, good_response, bad_response)
    
    def _generate_fallback_queries(self, document_text: str, num_queries: int) -> List[Dict[str, Any]]:
        """Generate fallback queries when structured generation fails"""
        base_queries = [
            {"question": "¿Cuál es el tema principal del documento?", "difficulty": "basic", "type": "factual"},
            {"question": "¿Qué información importante se presenta en el texto?", "difficulty": "intermediate", "type": "analytical"},
            {"question": "¿Cuáles son los aspectos más relevantes mencionados?", "difficulty": "basic", "type": "conceptual"},
            {"question": "¿Cómo se relacionan los conceptos presentados?", "difficulty": "advanced", "type": "analytical"},
            {"question": "¿Qué conclusiones se pueden extraer del contenido?", "difficulty": "advanced", "type": "conceptual"}
        ]
        return base_queries[:num_queries]
    
    def _heuristic_quality_score(self, query: str, document_text: str) -> Dict[str, float]:
        """Simple heuristic scoring when AI scoring fails"""
        # Basic heuristic implementation
        relevance = 0.6 if len(set(query.lower().split()) & set(document_text.lower().split())) > 2 else 0.3
        clarity = 0.8 if '?' in query and len(query) > 20 else 0.4
        difficulty = 0.5  # Default medium difficulty
        answerability = 0.7 if len(document_text) > 100 else 0.4
        overall = (relevance + clarity + answerability) / 3
        
        return {
            "relevance": relevance,
            "clarity": clarity, 
            "difficulty": difficulty,
            "answerability": answerability,
            "overall": overall
        }
    
    def _heuristic_validation(self, query: str, good: str, bad: str) -> Dict[str, Any]:
        """Simple heuristic validation when AI validation fails"""
        good_len = len(good) if good else 0
        bad_len = len(bad) if bad else 0
        
        is_valid = good_len > bad_len and good_len > 50
        good_quality = min(good_len / 200, 1.0) if good else 0.0
        bad_quality = min(bad_len / 200, 0.6) if bad else 0.0
        
        return {
            "is_valid": is_valid,
            "good_quality": good_quality,
            "bad_quality": bad_quality,
            "quality_difference": good_quality - bad_quality,
            "recommendation": "accept" if is_valid else "reject"
        }
