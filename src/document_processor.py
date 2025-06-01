import os
import re
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import tiktoken
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from .config import Config
from .advanced_pdf_processor import AdvancedPDFProcessor

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
        self.max_tokens = Config.MAX_TOKENS_PER_CHUNK
        self.min_chunk_size = Config.MIN_CHUNK_SIZE
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = None
        
        # Initialize advanced PDF processor
        self.advanced_pdf_processor = AdvancedPDFProcessor()
    
    def process_directory(self, input_dir: str) -> List[Dict[str, Any]]:
        """
        Process all documents in a directory with enhanced error handling
        
        Args:
            input_dir: Path to directory containing documents
            
        Returns:
            List of document chunks with metadata
        """
        documents = []
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        print(f"🔍 Scanning directory: {input_path}")
        
        # Find all files to process
        all_files = list(input_path.rglob("*"))
        processable_files = [f for f in all_files if f.is_file() and f.suffix.lower() in ['.pdf', '.txt', '.md']]
        
        print(f"📁 Found {len(processable_files)} processable files out of {len(all_files)} total files")
        
        successful_files = 0
        failed_files = 0
        
        # Process all supported file types
        for file_num, file_path in enumerate(processable_files, 1):
            try:
                print(f"🔄 Processing ({file_num}/{len(processable_files)}): {file_path.name}")
                
                if file_path.suffix.lower() == '.pdf':
                    doc_chunks = self.process_pdf(file_path)
                elif file_path.suffix.lower() in ['.txt', '.md']:
                    doc_chunks = self.process_text_file(file_path)
                else:
                    print(f"   ⚠️  Unsupported file type: {file_path.suffix}")
                    continue
                
                if doc_chunks:
                    documents.extend(doc_chunks)
                    successful_files += 1
                    print(f"✅ Successfully processed {file_path.name}: {len(doc_chunks)} chunks")
                else:
                    failed_files += 1
                    print(f"❌ Failed to process {file_path.name}: No content extracted")
                    
            except Exception as e:
                failed_files += 1
                print(f"❌ Error processing {file_path.name}: {e}")
                continue
        
        print(f"\n📊 Processing Summary:")
        print(f"   ✅ Successful: {successful_files} files")
        print(f"   ❌ Failed: {failed_files} files")
        print(f"   📄 Total chunks: {len(documents)}")
        
        return documents
    
    def process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF using advanced rotation handling"""
        try:
            print(f"   📖 Extracting text from PDF with advanced rotation handling...")
            
            # Use the advanced PDF processor
            text_pages = self.advanced_pdf_processor.extract_text_with_rotation_handling(file_path)
            
            if not text_pages:
                print(f"   ❌ No text content found in PDF")
                return []
            
            # Filter out empty pages and combine
            valid_pages = [page for page in text_pages if page.strip()]
            
            if not valid_pages:
                print(f"   ❌ No valid pages with text content")
                return []
            
            total_chars = sum(len(page) for page in valid_pages)
            print(f"   ✅ Successfully extracted {total_chars:,} characters from {len(valid_pages)}/{len(text_pages)} pages")
            
            # Combine all page content
            full_text = "\n\n".join(valid_pages)
            return self.create_chunks(full_text, str(file_path), "pdf")
                
        except Exception as e:
            print(f"   ❌ Critical error processing PDF {file_path}: {e}")
            return []
    
    def process_text_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process text file with multiple encoding attempts"""
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                print(f"   📝 Trying encoding: {encoding}")
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read()
                
                if text.strip():
                    print(f"   ✅ Successfully read with {encoding} encoding")
                    return self.create_chunks(text, str(file_path), "text")
                else:
                    print(f"   ⚠️  File appears to be empty")
                    return []
                    
            except UnicodeDecodeError:
                print(f"   ❌ Failed with {encoding}, trying next...")
                continue
            except Exception as e:
                print(f"   ❌ Error with {encoding}: {e}")
                continue
        
        print(f"   ❌ Could not read file with any encoding")
        return []
    
    def create_chunks(self, text: str, source: str, doc_type: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with improved boundary detection
        
        Args:
            text: Text to chunk
            source: Source file path
            doc_type: Type of document (pdf, text, etc.)
            
        Returns:
            List of document chunks with metadata
        """
        # Clean text
        text = self.clean_text(text)
        
        if not text.strip():
            print(f"   ⚠️  No content after cleaning")
            return []
        
        print(f"   ✂️  Creating chunks from {len(text):,} characters...")
        
        chunks = []
        chunk_id = 0
        
        # Improved chunking with better boundary detection
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to end at good boundaries
            if end < len(text):
                # Look for the best boundary within the last 20% of the chunk
                search_start = start + int(self.chunk_size * 0.8)
                
                # Priority order: paragraph break, sentence end, clause break
                boundaries = [
                    text.rfind('\n\n', search_start, end),  # Paragraph break
                    text.rfind('. ', search_start, end),     # Sentence end
                    text.rfind('; ', search_start, end),     # Clause break
                    text.rfind(', ', search_start, end),     # Comma break
                    text.rfind(' ', search_start, end)       # Word break
                ]
                
                # Use the best available boundary
                for boundary in boundaries:
                    if boundary > search_start:
                        end = boundary + 1
                        break
            
            chunk_text = text[start:end].strip()
            
            # Only keep chunks that meet minimum size requirements
            if len(chunk_text) >= self.min_chunk_size:
                # Count tokens if tokenizer is available
                token_count = self.count_tokens(chunk_text)
                
                # Skip chunks that exceed token limit
                if token_count <= self.max_tokens:
                    chunks.append({
                        'id': f"{Path(source).stem}_{chunk_id:04d}",
                        'text': chunk_text,
                        'source': source,
                        'doc_type': doc_type,
                        'chunk_index': chunk_id,
                        'char_count': len(chunk_text),
                        'token_count': token_count,
                        'metadata': {
                            'start_char': start,
                            'end_char': end,
                            'file_name': Path(source).name,
                            'total_file_chars': len(text)
                        }
                    })
                    chunk_id += 1
                else:
                    print(f"   ⚠️  Skipping chunk {chunk_id} - too many tokens: {token_count}")
            
            # Move start position with overlap, ensuring progress
            next_start = end - self.chunk_overlap
            start = max(next_start, start + self.min_chunk_size)
            
            if start >= len(text):
                break
        
        print(f"   ✅ Created {len(chunks)} valid chunks")
        return chunks
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text - Spanish language optimized"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Spanish characters and basic punctuation
        # Preserve: ñ, á, é, í, ó, ú, ü, ¿, ¡, and other Spanish characters
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)¿¡ñáéíóúüÑÁÉÍÓÚÜ]', ' ', text)
        
        # Remove repeated periods and other punctuation
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'-{3,}', '---', text)
        
        return text.strip()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        
        # Fallback: approximate token count
        return int(len(text.split()) * 1.3)  # Rough approximation
