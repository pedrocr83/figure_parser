# PDF Figure Analyzer

This project provides a simple AI-powered tool to extract and analyze scientific figures from PDF files. It uses Google Gemini Vision and LangExtract to automatically detect, describe, and classify figures (such as plots, microscopy images, gels, etc.) in scientific papers.

## Features
- Extracts all images from a PDF
- Uses AI to describe and classify each figure
- Outputs structured results as JSON
- Minimal setup and usage

## Quick Start Example

1. **Install dependencies** (from the project root):
   ```bash
   pip install .
   ```

2. **Set up your `.env` file** (see below).

3. **Run the analyzer:**
   ```bash
   python src/figure_parser/main.py path/to/your_paper.pdf
   ```
   Output will be saved in the `output/` directory by default.

## .env Example

Create a file named `.env` in the project root with the following content:

```
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

Replace `your_google_gemini_api_key_here` with your actual API key from Google AI Studio.

---

For more advanced usage and customization, see the `pdf_figure_analyzer_guide.md` in the project root.
