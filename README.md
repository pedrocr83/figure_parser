# PDF Figure Analyzer

This project provides a simple AI-powered tool to extract and analyze scientific figures from PDF files. It uses Google Gemini Vision and LangExtract to automatically detect, describe, and classify figures (such as plots, microscopy images, gels, etc.) in scientific papers.

## Features
- Extracts all images from a PDF (and splits multi-panel figures)
- Uses AI to describe and classify each figure
- Outputs structured results as JSON (both raw and filtered)
- Saves all extracted images and results in a dedicated folder per PDF
- Minimal setup and usage

## Quick Start Example

1. **Install dependencies** (from the project root, using [uv](https://github.com/astral-sh/uv)):
   ```bash
   uv sync
   ```
   This will install all dependencies specified in your `pyproject.toml`.

2. **Set up your `.env` file** (required for API access):
   - Create a file named `.env` in the project root (the same folder as your README and pyproject.toml).
   - The file should contain the following environment variables:
     ```
     # Required for Google Gemini Vision
     GOOGLE_API_KEY=your_google_gemini_api_key_here

     # Required for LangExtract cloud usage (if you have a key)
     LANGEXTRACT_API_KEY=your_langextract_api_key_here

     # Optional: Use Vertex AI endpoint for Google Generative AI (set to '1' to enable)
     GOOGLE_GENAI_USE_VERTEXAI=0
     ```
   - Replace the values with your actual API keys:
     - `GOOGLE_API_KEY`: Get from [Google AI Studio](https://makersuite.google.com/app/apikey).
     - `LANGEXTRACT_API_KEY`: Get from your LangExtract account (if required for your use case).
     - `GOOGLE_GENAI_USE_VERTEXAI`: Set to `1` if you want to use Vertex AI endpoint (advanced/enterprise use).
   - At minimum, `GOOGLE_API_KEY` and `LANGEXTRACT_API_KEY` are required for the analyzer to access Gemini Vision.

3. **Run the analyzer:**
   ```bash
   python src/figure_parser/main.py data/Raw/236077.pdf --output data
   ```
   - This will create a folder `data/236077` (for `236077.pdf`).
   - All extracted images and results will be saved inside this folder.

## Output Structure

For a PDF named `236077.pdf`, you will get:
```
data/
  236077/
    236077_page1_img0_panel0.png
    236077_page2_img1_panel0.png
    ...
    236077_raw_analysis.json      # All extracted images
    236077_analysis.json          # Only scientific figures (filtered)
```

- `*_raw_analysis.json`: Analysis for **all** extracted images.
- `*_analysis.json`: Only images where a scientific figure was detected.
