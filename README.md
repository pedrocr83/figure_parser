# Figure Parser

A comprehensive tool for extracting figures and tables from PDFs with optional AI-powered analysis, classification, and filtering.

## Features

- **PDF Figure & Table Extraction**: Uses advanced computer vision techniques to identify and extract figures, tables, and graphs from scientific PDFs
- **AI-Powered Analysis**: Integration with Google's Gemini AI for intelligent figure classification and description
- **AI-Based Filtering**: Automatically separates genuine figures from text-only images using Gemini Pro Vision
- **Multi-format Support**: Handles various image formats and PDF structures
- **Comprehensive Logging**: Detailed logging with configurable verbosity levels
- **Flexible Output**: Configurable DPI, padding, and output formats
- **Docker Support**: Containerized deployment for consistent environments

## Installation

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd figure_parser

# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY for AI analysis
```

### Docker Installation

```bash
# Build the Docker image
docker build -t figure-parser .

# Run with Docker
docker run -v $(pwd):/app figure-parser --pdf document.pdf --output results
```

## Usage

### Basic Extraction

Extract figures and tables from a PDF:

```bash
python src/figure_parser/main.py --pdf document.pdf --output results
```

### Extraction with AI Analysis

Extract and analyze with AI classification:

```bash
python src/figure_parser/main.py --pdf document.pdf --output results --analyze
```

### Extraction with AI Analysis and Filtering

Extract, analyze, and filter to separate genuine figures from text-only images:

```bash
python src/figure_parser/main.py --pdf document.pdf --output results --analyze --filter
```

### Advanced Options

```bash
python src/figure_parser/main.py \
    --pdf document.pdf \
    --output results \
    --dpi 300 \
    --padding 30 \
    --min-size 50 \
    --strict \
    --analyze \
    --filter \
    --verbose
```

## Command Line Arguments

### Required Arguments
- `--pdf`: Path to input PDF file

### Optional Arguments
- `--output`: Output directory (default: "output")
- `--dpi`: DPI for image extraction (default: 200)
- `--padding`: Padding around extracted elements in pixels (default: 20)
- `--min-size`: Minimum element size in pixels (default: 30)
- `--strict`: Enable strict mode for extraction
- `--analyze`: Run AI-based analysis on extracted images
- `--export-json`: Export analysis results to JSON (default: True)
- `--filter`: Run AI-based filtering to separate genuine figures from text-only images
- `--filter-output`: Output directory name for filtered results (default: "filtered_results")
- `--verbose, -v`: Enable verbose logging

## Output Structure

```
output/
├── figures/           # Extracted figure images
├── tables/            # Extracted table images  
├── graphs/            # Extracted graph images
├── metadata/          # Extraction logs and reports
├── failed/            # Failed extraction details
├── [PDF_NAME]/        # AI analysis results (if --analyze used)
│   ├── [PDF_NAME]_raw_analysis.json
│   └── [PDF_NAME]_analysis.json
└── filtered_results/  # AI-filtered results (if --filter used)
    ├── verified/      # Genuine figures, tables, and schemas
    └── text_only/     # Text-only images that were discarded
```

## Examples

### Example 1: Extract Only
```bash
# Extract figures and tables from a research paper
python src/figure_parser/main.py \
    --pdf research_paper.pdf \
    --output extracted_figures \
    --dpi 300
```

### Example 2: Extract and Analyze
```bash
# Extract and get AI analysis with high quality
python src/figure_parser/main.py \
    --pdf research_paper.pdf \
    --output analyzed_figures \
    --dpi 300 \
    --padding 40 \
    --analyze \
    --verbose
```

### Example 3: Extract, Analyze, and Filter
```bash
# Extract, analyze, and filter to get only genuine figures
python src/figure_parser/main.py \
    --pdf research_paper.pdf \
    --output processed_figures \
    --dpi 300 \
    --analyze \
    --filter \
    --filter-output clean_figures
```

### Example 4: Docker Usage
```bash
# Process PDF using Docker container
docker run -v $(pwd):/app figure-parser \
    --pdf research_paper.pdf \
    --output results \
    --analyze \
    --filter
```

### Example 5: Batch Processing
```bash
# Process multiple PDFs
for pdf in papers/*.pdf; do
    python src/figure_parser/main.py \
        --pdf "$pdf" \
        --output "results/$(basename "$pdf" .pdf)" \
        --analyze \
        --filter
done
```

## Architecture

The tool consists of three main components:

1. **ScientificPDFExtractor** (`internal/figure_extrator.py`):
   - Identifies figures, tables, and graphs in PDFs
   - Uses caption detection and visual analysis
   - Extracts high-quality images with configurable settings

2. **FigureAnalyzer** (`internal/figure_analyzer.py`):
   - Provides AI-powered analysis of extracted images
   - Classifies figure types and content
   - Generates structured metadata and descriptions

3. **FigureClassifier** (`internal/figure_classifier.py`):
   - Uses Gemini Pro Vision to filter images
   - Separates genuine figures from text-only images
   - Provides verified and text-only output directories

## Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Required for AI analysis and filtering (Gemini API)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)

### Output Settings
- **DPI**: Higher values produce larger, higher-quality images
- **Padding**: Adds space around extracted elements
- **Min Size**: Filters out very small elements
- **Strict Mode**: More conservative extraction with higher accuracy

## Docker

### Building the Image
```bash
docker build -t figure-parser .
```

### Running with Docker
```bash
# Basic usage
docker run -v $(pwd):/app figure-parser --pdf document.pdf --output results

# With AI analysis and filtering
docker run -v $(pwd):/app figure-parser \
    --pdf document.pdf \
    --output results \
    --analyze \
    --filter

# Override default command
docker run -v $(pwd):/app figure-parser --help
```

### Docker Features
- Based on Python 3.12 slim image
- Includes build tools for dependencies
- Mounts current directory to `/app` for easy file access
- Uses the main script as entrypoint

## Troubleshooting

### Common Issues

1. **No elements extracted**:
   - Check if PDF contains figures/tables
   - Try increasing `--min-size`
   - Use `--verbose` for detailed logging

2. **Poor extraction quality**:
   - Increase `--dpi` for higher resolution
   - Adjust `--padding` for better element boundaries
   - Enable `--strict` mode

3. **AI analysis fails**:
   - Verify `GOOGLE_API_KEY` is set
   - Check internet connection
   - Ensure extracted images exist

4. **AI filtering issues**:
   - Ensure Gemini API key is valid
   - Check API quota limits
   - Verify extracted images are accessible

### Logs
- Check `figure_parser.log` for detailed execution logs
- Use `--verbose` flag for real-time logging
- Review metadata files in output directory
- Check filtered results logs in `filtered_results/` directory

## Development

### Testing
```bash
# Run integration tests
python test_integration.py

# Test with sample PDF
python src/figure_parser/main.py --pdf test.pdf --output test_output --verbose
```

### Code Structure
```
figure_parser/
├── src/figure_parser/
│   ├── internal/
│   │   ├── figure_extrator.py    # PDF extraction logic
│   │   ├── figure_analyzer.py    # AI analysis logic
│   │   └── figure_classifier.py  # AI filtering logic
│   ├── __init__.py
│   └── main.py                   # Main CLI interface
├── Dockerfile                     # Docker containerization
├── requirements.txt               # Python dependencies
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

## TODO: Future Enhancements

### Unit Testing Implementation
- [ ] **Core Module Tests**: Create comprehensive unit tests for `figure_extrator.py`, `figure_analyzer.py`, and `figure_classifier.py`
- [ ] **Mock Testing**: Implement mock objects for external dependencies (Gemini API, file system operations)
- [ ] **Test Coverage**: Aim for >90% code coverage across all modules
- [ ] **Test Data**: Create sample PDFs and expected outputs for regression testing
- [ ] **CI/CD Integration**: Set up automated testing in GitHub Actions or similar

### Benchmarking & Performance
- [ ] **Performance Metrics**: Measure extraction time, memory usage, and CPU utilization
- [ ] **PDF Size Analysis**: Benchmark performance across different PDF sizes (1MB, 10MB, 100MB+)
- [ ] **Image Complexity**: Test performance with varying figure densities and complexities
- [ ] **API Latency**: Measure Gemini API response times and optimize batch processing
- [ ] **Memory Profiling**: Identify memory bottlenecks and optimize large PDF processing

### Scaling & Parallelization
- [ ] **Multi-PDF Processing**: Implement parallel processing for batch PDF operations
- [ ] **Worker Pool**: Create configurable worker pools for concurrent extraction
- [ ] **Async Processing**: Convert I/O operations to async/await for better concurrency
- [ ] **Distributed Processing**: Support for processing across multiple machines/nodes
- [ ] **Queue Management**: Implement job queues for large-scale batch processing
- [ ] **Resource Management**: Add configurable limits for concurrent operations
- [ ] **Progress Tracking**: Enhanced progress bars and status reporting for parallel jobs

### Infrastructure Improvements
- [ ] **Microservices**: Split into separate services for extraction, analysis, and filtering
- [ ] **API Endpoints**: RESTful API for integration with other systems
- [ ] **Database Integration**: Store processing results and metadata in persistent storage
- [ ] **Caching Layer**: Implement caching for repeated operations and API responses
- [ ] **Monitoring**: Add metrics collection and health checks for production deployment

## Dependencies

- **PyMuPDF (fitz)**: PDF processing and image extraction
- **Pillow (PIL)**: Image manipulation and format conversion
- **scikit-image**: Computer vision and image processing
- **opencv-python**: Additional computer vision capabilities
- **google-generativeai**: AI analysis with Gemini
- **langextract**: Structured data extraction
- **numpy**: Numerical operations
- **python-dotenv**: Environment variable management
- **tqdm**: Progress bars for long operations

