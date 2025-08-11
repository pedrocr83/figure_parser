import os
import json
import shutil
import google.generativeai as genai
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import logging

# Get logger for this module without configuring basic logging
logger = logging.getLogger(__name__)

class FigureClassifier:
    """
    A class to filter images extracted from scientific PDFs using the Gemini Pro Vision model.
    It identifies and separates genuine figures, tables, and schemas from text-only images.
    """

    def __init__(self, input_dir: str, output_dir_name: str = "filtered_results", api_key: str = None):
        """
        Initializes the FigureClassifier.

        Args:
            input_dir (str): The path to the output directory from figure_extractor.py.
            output_dir_name (str, optional): Name for the filtered results directory. Defaults to "filtered_results".
            api_key (str, optional): The Google AI API key. If None, uses environment variables.
        
        Raises:
            ValueError: If the API key is not found.
            FileNotFoundError: If the input directory or metadata log does not exist.
        """
        self.input_dir = Path(input_dir)
        self.output_dir_name = output_dir_name
        self.api_key = self._get_api_key(api_key)
        
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"The specified input directory does not exist: {self.input_dir}")
            
        self.metadata_path = self.input_dir / 'metadata' / 'extraction_log.json'
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

        self.output_dirs = self._setup_directories()
        self.model = self._initialize_gemini_model()
        logger.info("Classifier initialized successfully.")

    def _get_api_key(self, api_key_arg: str) -> str:
        """Retrieves the Google API key from arguments or environment variables."""
        if api_key_arg:
            return api_key_arg
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable or use the --api_key argument.")
        return api_key

    def _setup_directories(self) -> dict:
        """Creates the necessary directories for storing filtered results."""
        filtered_base = self.input_dir / self.output_dir_name
        dirs = {
            "base": filtered_base,
            "verified": filtered_base / "verified",
            "text_only": filtered_base / "text_only"
        }
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Filtered results will be saved in: {filtered_base}")
        return dirs

    def _initialize_gemini_model(self):
        """Initializes and returns the Gemini Pro Vision model."""
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel('gemini-pro-vision')

    def _classify_image(self, image_path: str) -> str:
        """
        Uses the Gemini Pro Vision model to classify a single image.

        Returns:
            A classification string: "KEEP", "DISCARD", or "ERROR".
        """
        prompt = """
        You are an AI assistant for scientific document analysis. Your task is to determine if the given image is a genuine figure, table, or schematic diagram, or if it is just a block of text that was mistakenly extracted.

        Analyze the provided image.

        Respond with a single word:
        - 'KEEP' if the image contains a figure, graph, table, or schematic.
        - 'DISCARD' if the image is primarily composed of text, such as a paragraph, a list, or an algorithm listing without a visual diagram.
        """
        try:
            image = Image.open(image_path)
            response = self.model.generate_content([prompt, image])
            # Clean up the response to handle potential markdown or extra text
            classification = response.text.strip().upper().replace("`", "")
            if "KEEP" in classification:
                return "KEEP"
            elif "DISCARD" in classification:
                return "DISCARD"
            else:
                logger.warning(f"Unexpected response from Gemini for {Path(image_path).name}: {response.text}. Defaulting to DISCARD.")
                return "DISCARD"
        except Exception as e:
            logger.error(f"Error classifying image {Path(image_path).name}: {e}")
            return "ERROR"

    def run_filter(self):
        """
        Orchestrates the image filtering process from start to finish.
        """
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        successful_elements = metadata.get('successful_elements', [])
        if not successful_elements:
            logger.warning("No successful extractions found in the metadata log to filter.")
            return

        filtered_metadata = []
        logger.info(f"Starting second-pass filtering for {len(successful_elements)} images...")

        for element in tqdm(successful_elements, desc="Analyzing Images"):
            original_path = element.get('file_path')
            if not original_path or not Path(original_path).exists():
                logger.warning(f"Image file not found for element: {element.get('number')}. Skipping.")
                element['filter_status'] = 'File Not Found'
                filtered_metadata.append(element)
                continue

            classification = self._classify_image(original_path)
            element['filter_status'] = classification

            if classification == "KEEP":
                dest_folder = self.output_dirs["verified"]
            elif classification == "DISCARD":
                dest_folder = self.output_dirs["text_only"]
            else:  # Handle ERROR case
                logger.warning(f"Skipping file movement for {Path(original_path).name} due to a classification error.")
                filtered_metadata.append(element)
                continue

            # Maintain original subdirectory structure (e.g., figures, tables)
            element_type_dir = Path(original_path).parent.name
            dest_path_dir = dest_folder / element_type_dir
            dest_path_dir.mkdir(exist_ok=True)
            
            dest_path = dest_path_dir / Path(original_path).name
            shutil.move(original_path, str(dest_path))
            
            element['new_file_path'] = str(dest_path)
            filtered_metadata.append(element)

        # Save the new filtered metadata
        metadata['filtered_elements'] = filtered_metadata
        filtered_log_path = self.output_dirs['base'] / 'filtered_extraction_log.json'
        with open(filtered_log_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Filtering complete! Filtered log saved to: {filtered_log_path}")
        
        # Return summary statistics
        keep_count = sum(1 for elem in filtered_metadata if elem.get('filter_status') == 'KEEP')
        discard_count = sum(1 for elem in filtered_metadata if elem.get('filter_status') == 'DISCARD')
        error_count = sum(1 for elem in filtered_metadata if elem.get('filter_status') == 'ERROR')
        
        return {
            'total_processed': len(filtered_metadata),
            'kept': keep_count,
            'discarded': discard_count,
            'errors': error_count,
            'filtered_log_path': str(filtered_log_path)
        }

# Keep the old class name for backward compatibility
GeminiImageClassifier = FigureClassifier