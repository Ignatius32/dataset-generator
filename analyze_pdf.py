import fitz
from pathlib import Path
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

def analyze_pdf():
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    pdf_path = Path('data/documents/02034002 TOULMIN, GOODFIED- La trama de los cielos cap 2.pdf')
    doc = fitz.open(str(pdf_path))

    print(f'PDF has {len(doc)} pages')

    # Check first page in detail
    page = doc[0]
    print(f'\nPage 0 details:')
    print(f'  Page rotation: {page.rotation}')
    print(f'  Page rect: {page.rect}')
    print(f'  Page cropbox: {page.cropbox}')
    print(f'  Page mediabox: {page.mediabox}')

    # Try different extraction methods
    methods = [
        ('get_text()', lambda: page.get_text()),
        ('get_text("text")', lambda: page.get_text('text')),
        ('get_text("dict")', lambda: len(str(page.get_text('dict')))),
        ('get_text("html")', lambda: len(page.get_text('html'))),
        ('get_text("xhtml")', lambda: len(page.get_text('xhtml'))),
    ]

    for name, method in methods:
        try:
            result = method()
            if isinstance(result, str):
                print(f'  {name}: {len(result)} chars - "{result[:100]}..."')
            else:
                print(f'  {name}: {result} length')
        except Exception as e:
            print(f'  {name}: Error - {e}')

    # Check if it's an image-based PDF
    print(f'\nPage images:')
    images = page.get_images()
    print(f'  Found {len(images)} images')
    for i, img in enumerate(images[:3]):  # Show first 3
        print(f'    Image {i}: {img}')

    # Check text blocks
    print(f'\nText blocks:')
    blocks = page.get_text('dict')['blocks']
    print(f'  Found {len(blocks)} blocks')
    for i, block in enumerate(blocks[:3]):  # Show first 3
        if 'lines' in block:
            print(f'    Block {i}: text block with {len(block["lines"])} lines')
            if block['lines']:
                for j, line in enumerate(block['lines'][:2]):
                    if 'spans' in line:
                        text = ''.join(span.get('text', '') for span in line['spans'])
                        print(f'      Line {j}: "{text}"')
        else:
            print(f'    Block {i}: image block')

    doc.close()

    # Try OCR on first page
    print(f'\nTrying OCR on first page...')
    try:
        # Convert first page to image
        images = convert_from_path(str(pdf_path), first_page=1, last_page=1, dpi=300)
        if images:
            img = images[0]
            
            # Try OCR with different rotations
            for rotation in [0, 90, 180, 270]:
                rotated_img = img.rotate(rotation, expand=True)
                ocr_text = pytesseract.image_to_string(rotated_img, lang='spa+eng')
                print(f'  OCR at {rotation}°: {len(ocr_text)} chars - "{ocr_text[:100]}..."')
                if len(ocr_text) > 100:
                    break
    except Exception as e:
        print(f'  OCR failed: {e}')

if __name__ == "__main__":
    analyze_pdf()
