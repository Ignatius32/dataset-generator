import os
import re
import io
from typing import List, Dict, Any, Tuple
from pathlib import Path
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

class AdvancedPDFProcessor:
    """Advanced PDF processor with comprehensive rotation handling"""
    
    def __init__(self):
        self.ocr_enabled = True
        try:
            # Set Tesseract path for Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            # Test if Tesseract is available
            pytesseract.get_tesseract_version()
            print("   ✅ Tesseract OCR is available")
        except Exception as e:
            print(f"   ⚠️  Tesseract OCR not available: {e}, disabling OCR fallback")
            self.ocr_enabled = False
    
    def extract_text_with_rotation_handling(self, file_path: Path) -> List[str]:
        """
        Extract text from PDF with comprehensive rotation handling.
        Each page is processed individually with multiple strategies.
        """
        print(f"   📖 Processing PDF with advanced rotation handling...")
        
        # Try multiple extraction methods and pick the best overall result
        methods = [
            ("PyMuPDF", self._extract_with_pymupdf_advanced),
            ("OCR", self._extract_with_ocr_all_pages),
            ("pdfplumber", self._extract_with_pdfplumber_advanced),
        ]
        
        best_extraction = []
        best_total_chars = 0
        
        for method_name, method in methods:
            try:
                print(f"   🔄 Trying {method_name}...")
                pages_text = method(file_path)
                total_chars = sum(len(page) for page in pages_text)
                
                print(f"   📊 {method_name}: {total_chars:,} characters from {len(pages_text)} pages")
                
                if total_chars > best_total_chars:
                    best_extraction = pages_text
                    best_total_chars = total_chars
                    print(f"   ✅ {method_name} is now the best extraction")
                    
                # If we get really good results (>1000 chars), use it
                if total_chars > 1000:
                    print(f"   🎉 Excellent extraction with {method_name}, stopping here")
                    break
                    
            except Exception as e:
                print(f"   ❌ {method_name} failed: {e}")
                continue
        
        return best_extraction
    
    def _extract_with_pymupdf_advanced(self, file_path: Path) -> List[str]:
        """Extract using PyMuPDF with individual page rotation testing"""
        text_pages = []
        
        doc = fitz.open(str(file_path))
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Test all 4 orientations for this page
            best_text = ""
            best_char_count = 0
            best_rotation = 0
            
            for test_rotation in [0, 90, 180, 270]:
                try:
                    # Set rotation and extract
                    page.set_rotation(test_rotation)
                    
                    # Try multiple extraction methods
                    extraction_methods = [
                        lambda: page.get_text(),
                        lambda: page.get_text("text"),
                        lambda: self._extract_text_blocks(page),
                    ]
                    
                    for extract_method in extraction_methods:
                        try:
                            text = extract_method()
                            char_count = len(text.strip())
                            
                            if char_count > best_char_count:
                                best_text = text
                                best_char_count = char_count
                                best_rotation = test_rotation
                        except:
                            continue
                            
                except Exception as e:
                    continue
            
            if best_char_count > 10:  # Minimum threshold
                if best_rotation != 0:
                    print(f"   🔄 Page {page_num + 1}: Best at {best_rotation}° ({best_char_count} chars)")
                text_pages.append(self._clean_text(best_text))
            else:
                print(f"   ⚠️  Page {page_num + 1}: Very poor extraction ({best_char_count} chars)")
                text_pages.append("")
        
        doc.close()
        return text_pages
    
    def _extract_text_blocks(self, page) -> str:
        """Extract text using text blocks method"""
        blocks = page.get_text("blocks")
        text_parts = []
        
        for block in blocks:
            if len(block) >= 5 and isinstance(block[4], str):
                text_parts.append(block[4])        
        return "\n".join(text_parts)
    
    def _extract_with_ocr_all_pages(self, file_path: Path) -> List[str]:
        """Extract all pages using OCR with rotation detection (using PyMuPDF instead of poppler)"""
        if not self.ocr_enabled:
            return []
        
        text_pages = []
        
        try:
            # Use PyMuPDF to convert pages to images (no poppler dependency)
            print(f"   🖼️  Converting PDF pages to images for OCR...")
            doc = fitz.open(str(file_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                best_text = ""
                best_char_count = 0
                best_rotation = 0
                
                # Try OCR at different rotations
                for rotation in [0, 90, 180, 270]:
                    try:
                        # Set page rotation
                        page.set_rotation(rotation)
                        
                        # Convert page to high-resolution image
                        mat = fitz.Matrix(2.0, 2.0)  # 2x scaling for better OCR
                        pix = page.get_pixmap(matrix=mat)
                          # Convert to PIL Image
                        img_data = pix.tobytes("ppm")
                        image = Image.open(io.BytesIO(img_data))
                        
                        # Try multiple OCR configurations
                        configs = [
                            '--psm 1 --oem 3',  # Automatic page segmentation with OSD
                            '--psm 3 --oem 3',  # Fully automatic page segmentation, no OSD
                            '--psm 6 --oem 3',  # Assume single uniform block of text
                            '--psm 4 --oem 3',  # Assume single column of text
                        ]
                        
                        for config in configs:
                            try:
                                ocr_text = pytesseract.image_to_string(
                                    image, 
                                    lang='spa+eng',  # Spanish + English
                                    config=config
                                )
                                
                                char_count = len(ocr_text.strip())
                                
                                # Check for meaningful text (not just noise)
                                if char_count > best_char_count and self._is_meaningful_text(ocr_text):
                                    best_text = ocr_text
                                    best_char_count = char_count
                                    best_rotation = rotation
                                    
                                # If we get good results, stop trying configs
                                if char_count > 500:
                                    break
                                    
                            except Exception as e:
                                continue
                        
                        # If we found good text, stop rotating
                        if best_char_count > 500:
                            break
                    
                    except Exception as e:
                        continue
                
                if best_char_count > 50:  # OCR threshold
                    if best_rotation != 0:
                        print(f"   ✅ Page {page_num + 1}: OCR success at {best_rotation}° ({best_char_count} chars)")
                    else:
                        print(f"   ✅ Page {page_num + 1}: OCR success ({best_char_count} chars)")
                    text_pages.append(self._clean_text(best_text))
                else:
                    print(f"   ❌ Page {page_num + 1}: OCR failed to find meaningful text")
                    text_pages.append("")
            
            doc.close()
        
        except Exception as e:
            print(f"   ❌ OCR processing failed: {e}")
            return []
        
        return text_pages
    
    def _extract_with_pdfplumber_advanced(self, file_path: Path) -> List[str]:
        """Extract using pdfplumber with advanced strategies"""
        text_pages = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                best_text = ""
                best_char_count = 0
                
                # Multiple extraction strategies
                strategies = [
                    lambda p: p.extract_text(),
                    lambda p: p.extract_text(layout=True),
                    lambda p: p.extract_text(x_tolerance=1, y_tolerance=1),
                    lambda p: p.extract_text(x_tolerance=3, y_tolerance=3),
                    lambda p: self._extract_with_bbox_strategy(p),
                ]
                
                for i, strategy in enumerate(strategies):
                    try:
                        text = strategy(page)
                        if text:
                            char_count = len(text.strip())
                            if char_count > best_char_count:
                                best_text = text
                                best_char_count = char_count
                                
                            # If we get good text, stop trying more strategies
                            if char_count > 200:
                                break
                    except:
                        continue
                
                if best_char_count > 10:
                    text_pages.append(self._clean_text(best_text))
                else:
                    text_pages.append("")
        
        return text_pages
    
    def _extract_with_bbox_strategy(self, page) -> str:
        """Extract text using bounding box strategy"""
        try:
            # Get all text objects
            chars = page.chars
            if not chars:
                return ""
            
            # Sort by position (top to bottom, left to right)
            chars = sorted(chars, key=lambda x: (x['top'], x['x0']))
            
            # Group into lines and words
            lines = []
            current_line = []
            current_top = None
            
            for char in chars:
                if current_top is None or abs(char['top'] - current_top) < 5:
                    current_line.append(char['text'])
                    current_top = char['top']
                else:
                    if current_line:
                        lines.append(''.join(current_line))
                    current_line = [char['text']]
                    current_top = char['top']
            
            if current_line:
                lines.append(''.join(current_line))
            
            return '\n'.join(lines)
        except:
            return ""
    
    def _is_meaningful_text(self, text: str) -> bool:
        """Check if OCR text contains meaningful content"""
        if not text or len(text.strip()) < 20:
            return False
        
        # Check for reasonable word/character ratio
        words = text.split()
        if len(words) < 3:
            return False
        
        # Check if it's mostly noise characters
        noise_chars = sum(1 for c in text if c in '!@#$%^&*()_+=[]{}|;":,.<>?~`')
        if noise_chars > len(text) * 0.3:  # More than 30% noise
            return False
        
        # Check for some alphabetic content
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars < len(text) * 0.5:  # Less than 50% alphabetic
            return False
        
        return True
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        # Remove form feed and other control characters
        text = re.sub(r'[\x0c\x00]', '', text)
        
        # Fix common OCR/extraction errors
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
        text = re.sub(r'(\w)([.!?])([A-Z])', r'\1\2 \3', text)  # Add space after punctuation
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        return text.strip()
