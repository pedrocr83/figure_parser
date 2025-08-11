import sys
import argparse
from pathlib import Path
from internal.figure_analyzer import PDFFigureAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Analyze figures in scientific PDFs using AI')
    parser.add_argument('pdf_path', help='Path to the PDF file to analyze')
    parser.add_argument('--output', '-o', default='output', help='Output directory for results (default: output)')
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return 1
    if not pdf_path.suffix.lower() == '.pdf':
        print("Error: Please provide a PDF file")
        return 1
    try:
        analyzer = PDFFigureAnalyzer(output_dir=args.output)
        result = analyzer.analyze_pdf(pdf_path)
        if result:
            print(f"\nSuccessfully analyzed {result['total_images']} images!")
            print(f"Check the '{args.output}' folder for results")
        else:
            print("No analysis results generated")
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
