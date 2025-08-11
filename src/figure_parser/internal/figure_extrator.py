import fitz  # PyMuPDF
import os
import re
import json
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
import traceback

@dataclass
class ExtractedElement:
    """Stores information about an extracted element."""
    element_type: str
    number: str
    page_num: int
    bbox: fitz.Rect
    file_path: str
    caption: str = ""
    confidence: float = 1.0
    extraction_method: str = ""
    multi_page: bool = False
    page_range: Optional[Tuple[int, int]] = None

class ScientificPDFExtractor:
    """
    A robust extractor focused on tables and figures, with comprehensive detection
    and extraction capabilities for scientific papers.
    """
    
    def __init__(self, pdf_path: str, output_dir: str = "output", dpi: int = 200, 
                 padding: int = 20, min_element_size: int = 30, 
                 enable_logging: bool = True, strict_mode: bool = False):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.dpi = dpi
        self.padding = padding
        self.min_element_size = min_element_size
        self.strict_mode = strict_mode
        self.doc = None
        self.elements = []
        self.failed_extractions = []
        self.stats = {
            'total_captions_found': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'multi_page_elements': 0
        }
        
        # Comprehensive patterns for different caption formats
        self.patterns = {
            'figure': [
                re.compile(r"(?:figure|fig\.?)\s*([\d\.]+[a-zA-Z]?)", re.IGNORECASE),
                re.compile(r"([\d\.]+[a-zA-Z]?)\s*(?:figure|fig\.?)", re.IGNORECASE),
                re.compile(r"^([\d\.]+[a-zA-Z]?)\s*[:\-]", re.IGNORECASE),  # Just numbers at start
            ],
            'table': [
                re.compile(r"(?:table|tab\.?)\s*([\d\.]+[a-zA-Z]?)", re.IGNORECASE),
                re.compile(r"([\d\.]+[a-zA-Z]?)\s*(?:table|tab\.?)", re.IGNORECASE),
                re.compile(r"^([\d\.]+[a-zA-Z]?)\s*[:\-]", re.IGNORECASE),  # Just numbers at start
            ],
            'graph': [
                re.compile(r"(?:graph|chart|plot|diagram)\s*([\d\.]+[a-zA-Z]?)", re.IGNORECASE),
                re.compile(r"([\d\.]+[a-zA-Z]?)\s*(?:graph|chart|plot|diagram)", re.IGNORECASE),
            ]
        }
        
        # Setup logging
        if enable_logging:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
            
        self._setup_directories()

    def _setup_directories(self):
        """Creates the necessary output directories."""
        for subdir in ['figures', 'tables', 'graphs', 'metadata', 'failed']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

    def _log(self, message: str, level: str = "info"):
        """Centralized logging with fallback to print."""
        if self.logger:
            if level == "info":
                self.logger.info(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)
            elif level == "debug":
                self.logger.debug(message)
        else:
            print(f"[{level.upper()}] {message}")

    def _find_captions(self) -> List[Dict[str, Any]]:
        """
        Comprehensive caption detection using multiple strategies.
        Looks for captions above, below, and around elements.
        """
        captions = []
        
        for page_num, page in enumerate(self.doc):
            try:
                # Get all text blocks on the page
                blocks = page.get_text("blocks", sort=True)
                
                # Strategy 1: Look for captions at the start of text blocks
                for block in blocks:
                    block_text = block[4].strip().replace('\n', ' ')
                    if not block_text:
                        continue
                        
                    for el_type, patterns in self.patterns.items():
                        for pattern in patterns:
                            match = pattern.search(block_text)
                            if match and self._is_valid_caption(block_text, match.group(0)):
                                captions.append({
                                    'type': el_type, 'number': match.group(1), 'text': block_text,
                                    'page_num': page_num, 'page': page, 'bbox': fitz.Rect(block[:4]),
                                    'detection_method': 'start_of_block'
                                })
                                break
                
                # Strategy 2: Look for captions in different positions
                self._find_captions_strategy_2(page, page_num, captions)
                
                # Strategy 3: Look for isolated caption-like text
                self._find_captions_strategy_3(page, page_num, captions)
                
            except Exception as e:
                self._log(f"Error processing page {page_num + 1}: {e}", "warning")
                continue
        
        # Remove duplicates and sort by page number
        unique_captions = self._deduplicate_captions(captions)
        self.stats['total_captions_found'] = len(unique_captions)
        
        return unique_captions

    def _find_captions_strategy_2(self, page: fitz.Page, page_num: int, captions: List[Dict]):
        """Look for captions in different positions relative to visual elements."""
        try:
            # Get visual elements
            drawings = page.get_drawings()
            images = page.get_image_info()
            
            for visual in drawings + images:
                visual_bbox = fitz.Rect(visual['rect'] if 'rect' in visual else visual['bbox'])
                
                # Look for captions above the visual element
                search_area_above = fitz.Rect(
                    page.rect.x0, max(page.rect.y0, visual_bbox.y0 - 200),
                    page.rect.x1, visual_bbox.y0
                )
                
                # Look for captions below the visual element
                search_area_below = fitz.Rect(
                    page.rect.x0, visual_bbox.y1,
                    page.rect.x1, min(page.rect.y1, visual_bbox.y1 + 200)
                )
                
                for search_area, position in [(search_area_above, 'above'), (search_area_below, 'below')]:
                    blocks = page.get_text("blocks", clip=search_area, sort=True)
                    for block in blocks:
                        block_text = block[4].strip().replace('\n', ' ')
                        if not block_text:
                            continue
                            
                        for el_type, patterns in self.patterns.items():
                            for pattern in patterns:
                                match = pattern.search(block_text)
                                if match and self._is_valid_caption(block_text, match.group(0)):
                                    captions.append({
                                        'type': el_type, 'number': match.group(1), 'text': block_text,
                                        'page_num': page_num, 'page': page, 'bbox': fitz.Rect(block[:4]),
                                        'detection_method': f'{position}_visual',
                                        'associated_visual': visual_bbox
                                    })
                                    break
        except Exception as e:
            self._log(f"Error in strategy 2 for page {page_num + 1}: {e}", "debug")

    def _find_captions_strategy_3(self, page: fitz.Page, page_num: int, captions: List[Dict]):
        """Look for isolated caption-like text that might have been missed."""
        try:
            # Get all text blocks and look for isolated caption patterns
            blocks = page.get_text("blocks", sort=True)
            
            for block in blocks:
                block_text = block[4].strip().replace('\n', ' ')
                if not block_text or len(block_text) < 10:  # Skip very short blocks
                    continue
                    
                # Look for patterns that might be captions but weren't caught
                for el_type, patterns in self.patterns.items():
                    for pattern in patterns:
                        match = pattern.search(block_text)
                        if match and self._is_likely_caption(block_text):
                            # Check if this caption is already captured
                            if not any(c['text'] == block_text for c in captions):
                                captions.append({
                                    'type': el_type, 'number': match.group(1), 'text': block_text,
                                    'page_num': page_num, 'page': page, 'bbox': fitz.Rect(block[:4]),
                                    'detection_method': 'isolated_caption'
                                })
        except Exception as e:
            self._log(f"Error in strategy 3 for page {page_num + 1}: {e}", "debug")

    def _is_valid_caption(self, text: str, pattern_match: str) -> bool:
        """Validate if a text block is a legitimate caption."""
        # Skip very long text blocks (likely paragraphs)
        if len(text) > 500:
            return False
            
        # Skip text that's mostly numbers or symbols
        if len(re.findall(r'[a-zA-Z]', text)) < 3:
            return False
            
        # Skip text that looks like references or citations
        if re.search(r'\[\d+\]|\(\d+\)|et al\.|doi:|http', text):
            return False
            
        return True

    def _is_likely_caption(self, text: str) -> bool:
        """Additional validation for isolated captions."""
        # Look for typical caption characteristics
        caption_indicators = [
            r'^[A-Z]',  # Starts with capital letter
            r'[a-z].*[a-z]',  # Contains lowercase letters
            r'[.!?]$',  # Ends with punctuation
            r'\b(?:shows?|displays?|illustrates?|presents?|depicts?)\b'  # Caption verbs
        ]
        
        return any(re.search(indicator, text, re.IGNORECASE) for indicator in caption_indicators)

    def _deduplicate_captions(self, captions: List[Dict]) -> List[Dict]:
        """Remove duplicate captions and sort by page number."""
        seen = set()
        unique_captions = []
        
        for caption in captions:
            # Create a unique key based on type, number, and page
            key = (caption['type'], caption['number'], caption['page_num'])
            if key not in seen:
                seen.add(key)
                unique_captions.append(caption)
        
        # Sort by page number and then by type
        unique_captions.sort(key=lambda x: (x['page_num'], x['type']))
        return unique_captions

    def _get_element_bbox(self, caption_info: Dict[str, Any]) -> Optional[fitz.Rect]:
        """
        Enhanced bounding box calculation using multiple strategies.
        """
        page = caption_info['page']
        caption_bbox = caption_info['bbox']
        element_type = caption_info['type']
        element_bbox = None
        
        try:
            if element_type == 'table':
                element_bbox = self._detect_table_bbox(page, caption_bbox)
            elif element_type in ['figure', 'graph']:
                element_bbox = self._detect_figure_bbox(page, caption_bbox)
            
            # Validate the detected bounding box
            if element_bbox and self._is_valid_bbox(element_bbox, page):
                return element_bbox
            else:
                # Try fallback strategies
                return self._fallback_bbox_detection(page, caption_bbox, element_type)
                
        except Exception as e:
            self._log(f"Error in bounding box detection: {e}", "warning")
            return self._fallback_bbox_detection(page, caption_bbox, element_type)

    def _detect_table_bbox(self, page: fitz.Page, caption_bbox: fitz.Rect) -> Optional[fitz.Rect]:
        """Enhanced table detection using text block analysis."""
        try:
            # Search area below caption
            search_area = fitz.Rect(
                page.rect.x0, caption_bbox.y1,
                page.rect.x1, min(page.rect.y1, caption_bbox.y1 + 800)
            )
            
            blocks = page.get_text("blocks", clip=search_area, sort=True)
            if not blocks:
                return None
                
            # Find consecutive text blocks that might form a table
            table_blocks = []
            current_y = blocks[0][1]  # y0 of first block
            
            for block in blocks:
                block_bbox = fitz.Rect(block[:4])
                
                # Check if block is close to previous blocks (likely part of table)
                if abs(block_bbox.y0 - current_y) < 50:
                    table_blocks.append(block)
                    current_y = block_bbox.y1
                else:
                    # Check if we've found enough blocks to form a table
                    if len(table_blocks) >= 2:
                        break
                    # Start new table
                    table_blocks = [block]
                    current_y = block_bbox.y1
            
            if table_blocks:
                # Create bounding box from all table blocks
                table_bbox = fitz.Rect(table_blocks[0][:4])
                for block in table_blocks[1:]:
                    table_bbox.include_rect(fitz.Rect(block[:4]))
                return table_bbox
                
        except Exception as e:
            self._log(f"Error in table detection: {e}", "debug")
            
        return None

    def _detect_figure_bbox(self, page: fitz.Page, caption_bbox: fitz.Rect) -> Optional[fitz.Rect]:
        """Enhanced figure detection using multiple visual analysis methods."""
        try:
            # Method 1: Look for visual elements above caption
            visuals = page.get_drawings() + page.get_image_info()
            candidate_bboxes = []
            
            for visual in visuals:
                visual_bbox = fitz.Rect(visual['rect'] if 'rect' in visual else visual['bbox'])
                
                # Check if visual is above caption and reasonably close
                if (visual_bbox.y1 < caption_bbox.y0 and 
                    (caption_bbox.y0 - visual_bbox.y1) < 600):
                    candidate_bboxes.append(visual_bbox)
            
            # Method 2: Look for text blocks that might be part of figure
            search_area = fitz.Rect(
                page.rect.x0, max(page.rect.y0, caption_bbox.y0 - 600),
                page.rect.x1, caption_bbox.y0
            )
            
            blocks = page.get_text("blocks", clip=search_area, sort=True)
            for block in blocks:
                block_bbox = fitz.Rect(block[:4])
                # Check if block might be part of figure (not too small)
                if block_bbox.width > 100 and block_bbox.height > 50:
                    candidate_bboxes.append(block_bbox)
            
            # Combine candidate bounding boxes
            if candidate_bboxes:
                element_bbox = fitz.Rect(candidate_bboxes[0])
                for bbox in candidate_bboxes[1:]:
                    element_bbox.include_rect(bbox)
                return element_bbox
                
        except Exception as e:
            self._log(f"Error in figure detection: {e}", "debug")
            
        return None

    def _is_valid_bbox(self, bbox: fitz.Rect, page: fitz.Page) -> bool:
        """Validate if a bounding box is reasonable."""
        if not bbox or bbox.is_empty:
            return False
            
        # Check minimum size
        if bbox.width < self.min_element_size or bbox.height < self.min_element_size:
            return False
            
        # Check if bbox is within page bounds
        if not page.rect.contains(bbox):
            return False
            
        return True

    def _fallback_bbox_detection(self, page: fitz.Page, caption_bbox: fitz.Rect, 
                                element_type: str) -> Optional[fitz.Rect]:
        """Fallback bounding box detection when primary methods fail."""
        try:
            if element_type == 'table':
                # For tables, use a conservative area below caption
                fallback_y0 = caption_bbox.y1
                fallback_y1 = min(page.rect.y1, caption_bbox.y1 + 400)
                return fitz.Rect(page.rect.x0, fallback_y0, page.rect.x1, fallback_y1)
            else:
                # For figures, use a conservative area above caption
                fallback_y0 = max(page.rect.y0, caption_bbox.y0 - 400)
                fallback_y1 = caption_bbox.y0
                return fitz.Rect(page.rect.x0, fallback_y0, page.rect.x1, fallback_y1)
                
        except Exception as e:
            self._log(f"Error in fallback detection: {e}", "debug")
            return None

    def _extract_element(self, caption: Dict[str, Any]) -> Optional[ExtractedElement]:
        """Extract a single element with comprehensive error handling."""
        try:
            el_type = caption['type']
            el_number = caption['number']
            page = caption['page']
            page_num = page.number
            
            self._log(f"   -> Analyzing {el_type.capitalize()} {el_number} on page {page_num + 1}...")
            
            # Get element bounding box
            element_bbox = self._get_element_bbox(caption)
            if not element_bbox or element_bbox.is_empty:
                self._log(f"      ⚠️ Could not determine valid boundaries for {el_type} {el_number}", "warning")
                return None
            
            # Create padded bounding box
            padded_bbox = fitz.Rect(
                element_bbox.x0 - self.padding,
                element_bbox.y0 - self.padding,
                element_bbox.x1 + self.padding,
                element_bbox.y1 + self.padding
            )
            
            # Intersect with page bounds
            final_bbox = padded_bbox.intersect(page.rect)
            if not final_bbox or final_bbox.is_empty:
                self._log(f"      ⚠️ Bounding box is empty after clipping for {el_type} {el_number}", "warning")
                return None
            
            # Convert to integer rect for pixmap
            final_clip = final_bbox.irect
            
            # Determine output path
            subdir = f"{el_type}s"
            filename = f"{el_type}_{el_number.replace('.', '_')}.png"
            output_path = os.path.join(self.output_dir, subdir, filename)
            
            # Extract the element
            pix = page.get_pixmap(clip=final_clip, dpi=self.dpi)
            pix.save(output_path)
            
            self._log(f"      ✓ Extracted successfully to '{output_path}'")
            
            # Create extracted element object
            element = ExtractedElement(
                element_type=el_type,
                number=el_number,
                page_num=page_num,
                bbox=element_bbox,
                file_path=output_path,
                caption=caption['text'],
                confidence=0.9,  # High confidence for successful extraction
                extraction_method=caption.get('detection_method', 'standard')
            )
            
            self.stats['successful_extractions'] += 1
            return element
            
        except Exception as e:
            error_msg = f"Failed to extract {el_type} {el_number}: {e}"
            self._log(error_msg, "error")
            self.failed_extractions.append({
                'type': el_type,
                'number': el_number,
                'page': page_num + 1,
                'error': str(e),
                'caption': caption.get('text', '')
            })
            self.stats['failed_extractions'] += 1
            return None

    def extract_all(self):
        """Main method to find and extract all tables and figures with comprehensive error handling."""
        try:
            self.doc = fitz.open(self.pdf_path)
            self._log(f"📄 Processing '{self.pdf_path}'...")
            
            # Find all captions
            captions = self._find_captions()
            self._log(f"   Found {len(captions)} potential captions.")
            
            # Extract each element
            for caption in captions:
                element = self._extract_element(caption)
                if element:
                    self.elements.append(element)
            
            # Save metadata and generate report
            self._save_metadata()
            self._generate_extraction_report()
            
            self._log("\n✅ Extraction complete.")
            self._log(f"   Successfully extracted: {self.stats['successful_extractions']}")
            self._log(f"   Failed extractions: {self.stats['failed_extractions']}")
            self._log(f"   Total elements found: {self.stats['total_captions_found']}")
            
        except Exception as e:
            self._log(f"❌ Critical error during extraction: {e}", "error")
            traceback.print_exc()
        finally:
            if self.doc:
                self.doc.close()

    def _save_metadata(self):
        """Saves comprehensive metadata about the extraction process."""
        try:
            metadata = {
                'pdf_file': self.pdf_path,
                'extraction_stats': self.stats,
                'successful_elements': [],
                'failed_extractions': self.failed_extractions,
                'extraction_settings': {
                    'dpi': self.dpi,
                    'padding': self.padding,
                    'min_element_size': self.min_element_size,
                    'strict_mode': self.strict_mode
                }
            }
            
            # Convert successful elements to dictionaries
            for elem in self.elements:
                elem_dict = asdict(elem)
                elem_dict['bbox'] = list(elem_dict['bbox'])
                metadata['successful_elements'].append(elem_dict)
            
            # Save main metadata
            metadata_path = os.path.join(self.output_dir, 'metadata', 'extraction_log.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self._log(f"   📋 Metadata saved to '{metadata_path}'")
            
            # Save failed extractions separately for analysis
            if self.failed_extractions:
                failed_path = os.path.join(self.output_dir, 'failed', 'failed_extractions.json')
                with open(failed_path, 'w', encoding='utf-8') as f:
                    json.dump(self.failed_extractions, f, indent=2)
                self._log(f"   📋 Failed extractions saved to '{failed_path}'")
                
        except Exception as e:
            self._log(f"Error saving metadata: {e}", "error")

    def _generate_extraction_report(self):
        """Generate a human-readable extraction report."""
        try:
            report_path = os.path.join(self.output_dir, 'metadata', 'extraction_report.txt')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("SCIENTIFIC PDF EXTRACTION REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"PDF File: {self.pdf_path}\n")
                f.write(f"Extraction Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("EXTRACTION STATISTICS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Captions Found: {self.stats['total_captions_found']}\n")
                f.write(f"Successful Extractions: {self.stats['successful_extractions']}\n")
                f.write(f"Failed Extractions: {self.stats['failed_extractions']}\n")
                f.write(f"Success Rate: {(self.stats['successful_extractions'] / max(self.stats['total_captions_found'], 1)) * 100:.1f}%\n\n")
                
                f.write("EXTRACTED ELEMENTS:\n")
                f.write("-" * 30 + "\n")
                
                # Group by type
                by_type = defaultdict(list)
                for elem in self.elements:
                    by_type[elem.element_type].append(elem)
                
                for elem_type, elems in by_type.items():
                    f.write(f"\n{elem_type.upper()}S ({len(elems)}):\n")
                    for elem in elems:
                        f.write(f"  - {elem_type.capitalize()} {elem.number} (Page {elem.page_num + 1})\n")
                        f.write(f"    Caption: {elem.caption[:100]}{'...' if len(elem.caption) > 100 else ''}\n")
                        f.write(f"    File: {elem.file_path}\n")
                
                if self.failed_extractions:
                    f.write(f"\nFAILED EXTRACTIONS:\n")
                    f.write("-" * 30 + "\n")
                    for failed in self.failed_extractions:
                        f.write(f"  - {failed['type'].capitalize()} {failed['number']} (Page {failed['page']})\n")
                        f.write(f"    Error: {failed['error']}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("End of Report\n")
                f.write("=" * 60 + "\n")
            
            self._log(f"   📋 Extraction report saved to '{report_path}'")
            
        except Exception as e:
            self._log(f"Error generating report: {e}", "error")

    def get_extraction_summary(self) -> Dict[str, Any]:
        """Return a summary of the extraction results."""
        return {
            'total_elements_found': self.stats['total_captions_found'],
            'successful_extractions': self.stats['successful_extractions'],
            'failed_extractions': self.stats['failed_extractions'],
            'success_rate': (self.stats['successful_extractions'] / max(self.stats['total_captions_found'], 1)) * 100,
            'elements_by_type': defaultdict(int, {elem.element_type: sum(1 for e in self.elements if e.element_type == elem.element_type) for elem in self.elements})
        }