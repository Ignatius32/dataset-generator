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
            "Content-Type": "application/json"
        }
        self.max_retries = max_retries or Config.MAX_RETRIES
        self.base_delay = base_delay or Config.BASE_RETRY_DELAY
        self.timeout = Config.REQUEST_TIMEOUT
        
        print("🚀 Chat API initialized")
      def generate_text_robust(self, messages: List[Dict[str, str]], 
                           max_tokens: int = 500, 
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
                     max_tokens: int = 500, 
                     temperature: float = 0.7,
                     system_prompt: str = None) -> Optional[str]:
        """
        Generate text using robust generation (compatibility method)
        """
        return self.generate_text_robust(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            max_attempts=3
        )

    def _make_api_call(self, messages: List[Dict[str, str]], 
                      max_tokens: int, temperature: float, 
                      system_prompt: str = None) -> Optional[str]:
        """Core API call with robust error handling"""
        
        # Add system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        for attempt in range(self.max_retries + 1):
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
                        content = result["choices"][0]["message"]["content"]
                        # Strip whitespace and ensure we have valid content
                        if content:
                            content = content.strip()
                            if content:  # Only return if we have actual content after stripping
                                return content
                        
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
        messages = [{"role": "user", "content": f"Genera {num_queries} preguntas sobre este texto: {document_text[:500]}..."}]
        response = self.generate_text(messages, max_tokens=300)
        
        if response:
            # Simple extraction of questions
            lines = response.split('\n')
            queries = []
            for line in lines:
                line = line.strip()
                if line and ('?' in line or line.endswith('?')):
                    # Clean up the line
                    line = re.sub(r'^\d+\.?\s*', '', line)  # Remove numbering
                    line = re.sub(r'^-\s*', '', line)      # Remove dashes
                    if len(line) > 10:
                        queries.append(line)
            
            return queries[:num_queries]
        
        return []
