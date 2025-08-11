import argparse
import sys
from pathlib import Path
import logging

from internal.figure_extrator import ScientificPDFExtractor
from internal.figure_analyzer import FigureAnalyzer
from internal.figure_classifier import FigureClassifier


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('figure_parser.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract figures and tables from PDFs with optional AI analysis and filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract figures and tables only
  python main.py --pdf document.pdf --output results
  
  # Extract and analyze with AI
  python main.py --pdf document.pdf --output results --analyze
  
  # Extract, analyze, and filter with AI
  python main.py --pdf document.pdf --output results --analyze --filter
  
  # Extract with custom settings
  python main.py --pdf document.pdf --output results --dpi 300 --padding 30
  
  # Extract, analyze, and filter with verbose logging
  python main.py --pdf document.pdf --output results --analyze --filter --verbose
        """
    )
    
    # Required arguments
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF file")
    parser.add_argument("--output", type=str, default="output", help="Output directory for extracted elements")
    
    # Extraction options
    parser.add_argument("--dpi", type=int, default=200, help="DPI for image extraction (default: 200)")
    parser.add_argument("--padding", type=int, default=20, help="Padding around extracted elements in pixels (default: 20)")
    parser.add_argument("--min-size", type=int, default=30, help="Minimum element size in pixels (default: 30)")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode for extraction")
    
    # Analysis options
    parser.add_argument("--analyze", action="store_true", help="Run AI-based analysis on extracted images")
    parser.add_argument("--export-json", action="store_true", default=True, help="Export analysis results to JSON (default: True)")
    
    # Filtering options
    parser.add_argument("--filter", action="store_true", help="Run AI-based filtering to separate genuine figures from text-only images")
    parser.add_argument("--filter-output", type=str, default="filtered_results", help="Output directory name for filtered results (default: filtered_results)")
    
    # General options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate input
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.suffix.lower() == '.pdf':
        logger.error(f"File must be a PDF: {pdf_path}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing PDF: {pdf_path}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Step 1: Extract figures and tables using ScientificPDFExtractor
        logger.info("=" * 60)
        logger.info("STEP 1: EXTRACTING FIGURES AND TABLES")
        logger.info("=" * 60)
        
        extractor = ScientificPDFExtractor(
            pdf_path=str(pdf_path),
            output_dir=str(output_dir),
            dpi=args.dpi,
            padding=args.padding,
            min_element_size=args.min_size,
            strict_mode=args.strict,
            enable_logging=True
        )
        
        # Perform extraction
        extractor.extract_all()
        
        # Get extraction summary
        extraction_summary = extractor.get_extraction_summary()
        
        logger.info("\nExtraction Summary:")
        logger.info(f"  Total elements found: {extraction_summary['total_elements_found']}")
        logger.info(f"  Successful extractions: {extraction_summary['successful_extractions']}")
        logger.info(f"  Failed extractions: {extraction_summary['failed_extractions']}")
        logger.info(f"  Success rate: {extraction_summary['success_rate']:.1f}%")
        
        # Show breakdown by type
        for elem_type, count in extraction_summary['elements_by_type'].items():
            logger.info(f"  {elem_type.capitalize()}s: {count}")
        
        # Step 2: Optional AI Analysis
        if args.analyze:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 2: AI-BASED ANALYSIS")
            logger.info("=" * 60)
            
            if extraction_summary['successful_extractions'] == 0:
                logger.warning("No elements were successfully extracted. Skipping analysis.")
            else:
                # Initialize analyzer
                analyzer = FigureAnalyzer(
                    output_dir=str(output_dir),
                    export_json=args.export_json
                )
                
                # Analyze the PDF
                logger.info("Running AI analysis on extracted elements...")
                analysis_result = analyzer.analyze_pdf(str(pdf_path))
                
                logger.info("\nAnalysis Summary:")
                logger.info(f"  Total images analyzed: {analysis_result.get('total_images', 0)}")
                logger.info(f"  Analysis results saved to: {output_dir}")
                
                if args.export_json:
                    logger.info("  JSON export enabled - results saved to output directory")
        
        # Step 3: Optional AI Filtering
        if args.filter:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 3: AI-BASED FILTERING")
            logger.info("=" * 60)
            
            if extraction_summary['successful_extractions'] == 0:
                logger.warning("No elements were successfully extracted. Skipping filtering.")
            else:
                # Initialize classifier
                classifier = FigureClassifier(
                    input_dir=str(output_dir),
                    output_dir_name=args.filter_output
                )
                
                # Run filtering
                logger.info("Running AI-based filtering to separate genuine figures from text-only images...")
                filter_result = classifier.run_filter()
                
                if filter_result:
                    logger.info("\nFiltering Summary:")
                    logger.info(f"  Total processed: {filter_result['total_processed']}")
                    logger.info(f"  Kept (genuine figures): {filter_result['kept']}")
                    logger.info(f"  Discarded (text-only): {filter_result['discarded']}")
                    logger.info(f"  Errors: {filter_result['errors']}")
                    logger.info(f"  Filtered log saved to: {filter_result['filtered_log_path']}")
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir}")
        
        steps_completed = ["Extraction"]
        if args.analyze and extraction_summary['successful_extractions'] > 0:
            steps_completed.append("AI Analysis")
        if args.filter and extraction_summary['successful_extractions'] > 0:
            steps_completed.append("AI Filtering")
        
        logger.info(f"Completed steps: {' → '.join(steps_completed)}")
            
        # Show output file locations
        logger.info("\nOutput files:")
        logger.info(f"  Extracted elements: {output_dir}/figures/, {output_dir}/tables/, {output_dir}/graphs/")
        logger.info(f"  Metadata: {output_dir}/metadata/")
        if args.analyze and args.export_json:
            logger.info(f"  Analysis results: {output_dir}/")
        if args.filter:
            logger.info(f"  Filtered results: {output_dir}/{args.filter_output}/")
        
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error during processing: {e}")
        logger.error("Check the logs for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()
