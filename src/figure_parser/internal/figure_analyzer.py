import os
import json
import fitz  # PyMuPDF
from PIL import Image
import io
from dotenv import load_dotenv
import google.generativeai as genai
import langextract as lx
from pathlib import Path
import numpy as np
from skimage import filters, measure, morphology, color, util
import logging

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Get logger for this module without configuring basic logging
logger = logging.getLogger(__name__)

class FigureAnalyzer:
    def __init__(self, output_dir: str = "output", export_json: bool = True):
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.export_json = export_json
        self.supported_image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}
        self.classification_examples = [
            lx.data.ExampleData(
                text="Bar chart showing gene expression levels across different conditions",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="figure_type",
                        extraction_text="plot",
                        attributes={"subtype": "bar_chart", "domain": "genomics"}
                    )
                ]
            ),
            lx.data.ExampleData(
                text="Fluorescence microscopy image of cells with DAPI staining",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="figure_type",
                        extraction_text="microscopy",
                        attributes={"subtype": "fluorescence", "staining": "DAPI"}
                    )
                ]
            ),
            lx.data.ExampleData(
                text="Western blot showing protein bands at different molecular weights",
                extractions=[
                    lx.data.Extraction(
                        extraction_class="figure_type",
                        extraction_text="gel",
                        attributes={"subtype": "western_blot", "analysis": "protein"}
                    )
                ]
            )
        ]

    def find_images_in_folder(self, folder_path):
        """Recursively find all image files in a folder and its subfolders."""
        folder_path = Path(folder_path)
        image_files = []
        
        for file_path in folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_image_extensions:
                # Calculate relative path from the input folder
                relative_path = file_path.relative_to(folder_path)
                image_files.append({
                    'file_path': str(file_path),
                    'relative_path': str(relative_path),
                    'filename': file_path.name,
                    'subfolder': str(relative_path.parent) if relative_path.parent != Path('.') else 'root'
                })
        
        return image_files

    def load_image_from_file(self, file_path):
        """Load an image file and convert to RGB if needed."""
        try:
            with Image.open(file_path) as img:
                # Convert to RGB if needed (some formats like RGBA or grayscale)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return img.copy()  # Return a copy to avoid file handle issues
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            return None

    def process_image_file(self, file_path, relative_path, min_size=100):
        """Process a single image file, splitting into panels if needed."""
        image = self.load_image_from_file(file_path)
        if image is None:
            return None
        
        # Split into panels
        panels = self.split_into_panels(image, min_size=min_size)
        
        processed_panels = []
        for panel_idx, panel in enumerate(panels):
            if panel.width < min_size or panel.height < min_size:
                continue
            
            # Convert panel to bytes for analysis
            buf = io.BytesIO()
            panel.save(buf, format="PNG")
            panel_data = buf.getvalue()
            
            processed_panels.append({
                'panel_index': panel_idx,
                'data': panel_data,
                'width': panel.width,
                'height': panel.height,
                'relative_path': relative_path
            })
        
        return processed_panels

    def analyze_folder(self, folder_path):
        """Analyze all images in a folder and its subfolders."""
        folder_path = Path(folder_path)
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Invalid folder path: {folder_path}")
        
        logger.info(f"Starting folder analysis for: {folder_path}")
        
        # Find all images
        image_files = self.find_images_in_folder(folder_path)
        logger.info(f"Found {len(image_files)} image files")
        
        if not image_files:
            logger.warning("No image files found in the specified folder")
            return {
                'folder_path': str(folder_path),
                'total_images': 0,
                'image_analyses': [],
                'subfolder_summary': {}
            }
        
        # Group images by subfolder
        subfolder_images = {}
        for img_info in image_files:
            subfolder = img_info['subfolder']
            if subfolder not in subfolder_images:
                subfolder_images[subfolder] = []
            subfolder_images[subfolder].append(img_info)
        
        # Process each subfolder
        all_results = []
        subfolder_summary = {}
        
        for subfolder, images in subfolder_images.items():
            logger.info(f"Processing subfolder: {subfolder} ({len(images)} images)")
            subfolder_results = []
            
            for i, img_info in enumerate(images):
                logger.info(f"Processing image {i+1}/{len(images)} in {subfolder}: {img_info['filename']}")
                
                # Process the image file
                panels = self.process_image_file(
                    img_info['file_path'], 
                    img_info['relative_path']
                )
                
                if panels:
                    for panel in panels:
                        # Analyze with AI
                        description = self.detect_figures_with_ai(panel['data'])
                        
                        result = {
                            'subfolder': subfolder,
                            'filename': img_info['filename'],
                            'relative_path': img_info['relative_path'],
                            'panel_index': panel['panel_index'],
                            'dimensions': f"{panel['width']}x{panel['height']}",
                            'ai_description': description
                        }
                        
                        subfolder_results.append(result)
                        all_results.append(result)
            
            # Save subfolder analysis
            if self.export_json:
                subfolder_output_file = self.output_dir / f"{folder_path.stem}_{subfolder}_analysis.json"
                subfolder_result = {
                    'subfolder': subfolder,
                    'total_images': len(subfolder_results),
                    'image_analyses': subfolder_results
                }
                
                with open(subfolder_output_file, 'w') as f:
                    json.dump(subfolder_result, f, indent=2)
                
                subfolder_summary[subfolder] = {
                    'total_images': len(subfolder_results),
                    'output_file': str(subfolder_output_file)
                }
        
        # Save overall analysis
        if self.export_json:
            overall_output_file = self.output_dir / f"{folder_path.stem}_overall_analysis.json"
            overall_result = {
                'folder_path': str(folder_path),
                'total_images': len(all_results),
                'total_files': len(image_files),
                'subfolder_summary': subfolder_summary,
                'image_analyses': all_results
            }
            
            with open(overall_output_file, 'w') as f:
                json.dump(overall_result, f, indent=2)
        
        return {
            'folder_path': str(folder_path),
            'total_images': len(all_results),
            'total_files': len(image_files),
            'subfolder_summary': subfolder_summary,
            'image_analyses': all_results
        }

    def split_into_panels(self, image, min_size=100):
        """Split a figure image into individual panels using segmentation. Only keep panels above min_size."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        gray = color.rgb2gray(np.array(image))
        thresh = filters.threshold_otsu(gray)
        binary = gray < thresh
        cleaned = morphology.remove_small_objects(binary, min_size=500)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=500)
        labeled = measure.label(cleaned)
        regions = measure.regionprops(labeled)
        panels = []
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            width = maxc - minc
            height = maxr - minr
            if width < min_size or height < min_size:
                continue
            panel_img = image.crop((minc, minr, maxc, maxr))
            panels.append(panel_img)
        if not panels:
            return [image]
        return panels

    def extract_images_from_pdf(self, pdf_path, min_size=100, output_dir=None, pdf_stem=None):
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n < 5:
                    img_data = pix.tobytes("png")
                    pil_img = Image.open(io.BytesIO(img_data))
                    if pil_img.width < min_size or pil_img.height < min_size:
                        continue
                    panels = self.split_into_panels(pil_img, min_size=min_size)
                    for panel_idx, panel in enumerate(panels):
                        if panel.width < min_size or panel.height < min_size:
                            continue
                        buf = io.BytesIO()
                        panel.save(buf, format="PNG")
                        panel_data = buf.getvalue()
                        # Save panel image to output folder if output_dir and pdf_stem are provided
                        img_filename = None
                        if output_dir and pdf_stem:
                            img_filename = f"{pdf_stem}_page{page_num+1}_img{img_index}_panel{panel_idx}.png"
                            out_folder = Path(output_dir) / pdf_stem
                            out_folder.mkdir(parents=True, exist_ok=True)
                            with open(out_folder / img_filename, "wb") as f:
                                f.write(panel_data)
                        images.append({
                            'page': page_num + 1,
                            'index': f"{img_index}_{panel_idx}",
                            'data': panel_data,
                            'width': panel.width,
                            'height': panel.height,
                            'filename': img_filename if img_filename else None
                        })
                pix = None
        doc.close()
        return images

    def detect_figures_with_ai(self, image_data):
        try:
            image = Image.open(io.BytesIO(image_data))
            prompt = """
            Analyze this image and identify if it contains scientific figures. For each figure found:
            1. Describe what type of figure it is (plot, graph, microscopy, gel, diagram, table, etc.)
            2. Describe the content and what it shows
            3. Identify if it's a multi-panel figure (has panels A, B, C, etc.)
            4. Note any key findings or patterns visible
            Be specific and scientific in your description.
            """
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    def classify_panels_structured(self, descriptions):
        prompt = """
        Extract figure classification and metadata from scientific figure descriptions.
        Focus on: figure type, scientific domain, key attributes, and content description.
        Be precise and use exact text from the description.
        """
        try:
            result = lx.extract(
                text_or_documents=descriptions,
                prompt_description=prompt,
                examples=self.classification_examples,
                model_id="gemini-2.5-pro"
            )
            return result
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return None

    def analyze_pdf(self, pdf_path):
        pdf_stem = Path(pdf_path).stem
        out_folder = self.output_dir / pdf_stem
        out_folder.mkdir(parents=True, exist_ok=True)
        images = self.extract_images_from_pdf(pdf_path, output_dir=self.output_dir, pdf_stem=pdf_stem)
        logger.info(f"Found {len(images)} images")
        results = []
        descriptions = []
        for i, img in enumerate(images):
            logger.info(f"Analyzing image {i+1}/{len(images)}...")
            description = self.detect_figures_with_ai(img['data'])
            result = {
                'page': img['page'],
                'image_index': img['index'],
                'dimensions': f"{img['width']}x{img['height']}",
                'ai_description': description,
                'filename': img.get('filename')
            }
            results.append(result)
            descriptions.append(description)
        # Save raw analysis (all images)
        if self.export_json:
            raw_output_file = out_folder / f"{pdf_stem}_raw_analysis.json"
            raw_final_result = {
                'pdf_path': str(pdf_path),
                'total_images': len(images),
                'image_analyses': results,
                'structured_extractions': None
            }
            with open(raw_output_file, 'w') as f:
                json.dump(raw_final_result, f, indent=2)
        # Filter for scientific figures
        def is_scientific_figure(desc):
            if not isinstance(desc, str):
                return False
            desc_l = desc.lower()
            # Heuristic: look for keywords
            return (
                'figure' in desc_l or 'plot' in desc_l or 'graph' in desc_l or 'diagram' in desc_l or 'microscopy' in desc_l or 'gel' in desc_l or 'table' in desc_l
            ) and not (
                'no scientific figures' in desc_l or 'does not contain' in desc_l or 'not possible to identify' in desc_l or 'does not qualify' in desc_l or 'does not present' in desc_l or 'does not contain a recognizable scientific figure' in desc_l
            )
        filtered_results = [r for r in results if is_scientific_figure(r['ai_description'])]
        if self.export_json:
            filtered_output_file = out_folder / f"{pdf_stem}_analysis.json"
            filtered_final_result = {
                'pdf_path': str(pdf_path),
                'total_images': len(filtered_results),
                'image_analyses': filtered_results,
                'structured_extractions': None
            }
            with open(filtered_output_file, 'w') as f:
                json.dump(filtered_final_result, f, indent=2)
            return filtered_final_result
        # If not exporting, still return in-memory result
        return {
            'pdf_path': str(pdf_path),
            'total_images': len(filtered_results),
            'image_analyses': filtered_results,
            'structured_extractions': None
        }

# Keep the old class name for backward compatibility
PDFFigureAnalyzer = FigureAnalyzer

# Usage examples
if __name__ == "__main__":
    # Example 1: Analyze a folder with images
    analyzer = FigureAnalyzer(output_dir="output", export_json=True)
    
    # Analyze a folder containing subfolders with images
    try:
        results = analyzer.analyze_folder("/path/to/your/image/folder")
        print(f"Analysis complete! Found {results['total_images']} images across {len(results['subfolder_summary'])} subfolders")
    except Exception as e:
        print(f"Error during analysis: {e}")
    
    # Example 2: Analyze a PDF (original functionality still works)
    try:
        pdf_results = analyzer.analyze_pdf("/path/to/your/document.pdf")
        print(f"PDF analysis complete! Found {pdf_results['total_images']} images")
    except Exception as e:
        print(f"Error during PDF analysis: {e}")