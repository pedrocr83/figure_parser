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

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class PDFFigureAnalyzer:
    def __init__(self, output_dir="output"):
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
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
            print(f"Classification error: {e}")
            return None

    def analyze_pdf(self, pdf_path):
        pdf_stem = Path(pdf_path).stem
        out_folder = self.output_dir / pdf_stem
        out_folder.mkdir(parents=True, exist_ok=True)
        images = self.extract_images_from_pdf(pdf_path, output_dir=self.output_dir, pdf_stem=pdf_stem)
        print(f"Found {len(images)} images")
        results = []
        descriptions = []
        for i, img in enumerate(images):
            print(f"Analyzing image {i+1}/{len(images)}...")
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
        filtered_output_file = out_folder / f"{pdf_stem}_analysis.json"
        filtered_final_result = {
            'pdf_path': str(pdf_path),
            'total_images': len(filtered_results),
            'image_analyses': filtered_results,
            'structured_extractions': None
        }
        with open(filtered_output_file, 'w') as f:
            json.dump(filtered_final_result, f, indent=2)
        # Also return the filtered result for compatibility
        return filtered_final_result 