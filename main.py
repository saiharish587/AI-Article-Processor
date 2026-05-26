import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

from src.chunk_text import chunk_text
from src.summarize import load_summarizer, summarize_chunk
from src.categorize import load_category_classifier, categorize_chunk
from src.detect_spam import load_spam_detector, detect_spam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('article_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ArticleProcessor:
    """Advanced article processing orchestrator with comprehensive features."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the article processor with configuration."""
        self.config = self._load_config(config_path)
        self.models_dir = Path(self.config.get('models_dir', 'models'))
        self.output_dir = Path(self.config.get('output_dir', '.'))

        # Ensure directories exist
        self.models_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # Model cache
        self._models = {}

        logger.info("Article Processor initialized")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'models_dir': 'models',
            'output_dir': '.',
            'max_chunk_tokens': 512,
            'summary_max_length': 50,
            'summary_min_length': 20,
            'device': 0,  # GPU device
            'batch_size': 1,
            'retry_attempts': 3,
            'timeout': 300
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config {config_path}: {e}")

        return default_config

    def _load_models_with_progress(self):
        """Load all models with progress tracking."""
        models_to_load = [
            ('summarizer', 'Loading summarization model...', load_summarizer),
            ('category_classifier', 'Loading classification model...', load_category_classifier),
            ('spam_detector', 'Loading spam detection model...', load_spam_detector)
        ]

        for model_name, message, loader_func in tqdm(models_to_load, desc="Loading AI Models"):
            if model_name not in self._models:
                logger.info(message)
                try:
                    self._models[model_name] = loader_func()
                    logger.info(f"✓ {model_name} loaded successfully")
                except Exception as e:
                    logger.error(f"✗ Failed to load {model_name}: {e}")
                    raise

    def _validate_file(self, file_path: str) -> Path:
        """Validate and return Path object for input file."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        if path.suffix.lower() != '.txt':
            logger.warning(f"File extension is not .txt: {path.suffix}")

        return path

    def _create_output_filename(self, input_file: Path) -> Path:
        """Generate timestamped output filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = input_file.stem
        return self.output_dir / f"processed_{base_name}_{timestamp}.csv"

    def process_article(self, file_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single article with comprehensive analysis.

        Args:
            file_path: Path to the input text file
            output_path: Optional custom output path

        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()

        try:
            # Validate input file
            input_file = self._validate_file(file_path)
            logger.info(f"Processing article: {input_file}")

            # Load models if not already loaded
            if not self._models:
                self._load_models_with_progress()

            # Read and validate content
            with open(input_file, 'r', encoding='utf-8') as f:
                full_content = f.read().strip()

            if not full_content:
                raise ValueError("Input file is empty")

            content_length = len(full_content)
            logger.info(f"Content length: {content_length} characters")

            # Intelligent chunking with progress
            logger.info("Performing intelligent text chunking...")
            chunks = chunk_text(full_content, max_tokens=self.config['max_chunk_tokens'])

            if not chunks:
                raise ValueError("No chunks generated from input text")

            logger.info(f"Generated {len(chunks)} chunks")

            # Multi-level summarization
            summarizer = self._models['summarizer']
            chunk_summaries = []

            logger.info("Generating chunk-level summaries...")
            for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks")):
                try:
                    summary = summarize_chunk(summarizer, chunk)
                    chunk_summaries.append(summary)
                except Exception as e:
                    logger.warning(f"Failed to summarize chunk {i+1}: {e}")
                    chunk_summaries.append(f"[Summary failed for chunk {i+1}]")

            # Generate final article summary
            logger.info("Creating final article summary...")
            combined_summaries = ' '.join(chunk_summaries)

            try:
                final_summary = summarizer(
                    combined_summaries,
                    max_length=self.config['summary_max_length'],
                    min_length=self.config['summary_min_length'],
                    do_sample=False
                )[0]['summary_text']
            except Exception as e:
                logger.warning(f"Final summary generation failed: {e}")
                final_summary = combined_summaries[:200] + "..." if len(combined_summaries) > 200 else combined_summaries

            # Topic categorization
            logger.info("Categorizing article topic...")
            category_classifier = self._models['category_classifier']
            category = categorize_chunk(category_classifier, final_summary)

            # Spam detection
            logger.info("Analyzing content quality...")
            spam_detector = self._models['spam_detector']
            is_spam = detect_spam(spam_detector, final_summary)

            # Prepare results
            processing_time = time.time() - start_time
            result = {
                'title': input_file.stem.replace('_', ' '),
                'file_path': str(input_file.absolute()),
                'final_summary': final_summary,
                'category': category,
                'is_spam': is_spam,
                'num_chunks': len(chunks),
                'content_length': content_length,
                'processing_time_seconds': round(processing_time, 2),
                'chunk_summaries': chunk_summaries,
                'timestamp': datetime.now().isoformat(),
                'config_used': self.config
            }

            # Save results
            output_file = Path(output_path) if output_path else self._create_output_filename(input_file)
            self._save_results(result, output_file)

            logger.info(f"✓ Processing completed in {processing_time:.2f} seconds")
            logger.info(f"✓ Results saved to: {output_file}")

            return result

        except Exception as e:
            logger.error(f"✗ Processing failed: {e}")
            raise

    def _save_results(self, result: Dict[str, Any], output_file: Path):
        """Save processing results to CSV and JSON formats."""
        # Prepare CSV data (exclude complex objects)
        csv_data = {
            k: v for k, v in result.items()
            if k not in ['chunk_summaries', 'config_used', 'timestamp']
        }
        csv_data['chunk_summaries'] = json.dumps(result['chunk_summaries'])

        # Save CSV
        df = pd.DataFrame([csv_data])
        df.to_csv(output_file, index=False)

        # Save detailed JSON
        json_file = output_file.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved as CSV: {output_file}")
        logger.info(f"Detailed results saved as JSON: {json_file}")

    def list_available_files(self, directory: str = '.') -> list:
        """List all available .txt files in a directory."""
        path = Path(directory)
        txt_files = list(path.glob('*.txt'))
        return [str(f) for f in sorted(txt_files)]

    def batch_process(self, file_paths: list, output_dir: Optional[str] = None) -> list:
        """Process multiple articles in batch mode."""
        results = []

        for file_path in tqdm(file_paths, desc="Batch processing"):
            try:
                result = self.process_article(file_path, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append({'file_path': file_path, 'error': str(e)})

        return results


def create_argument_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Article Processor - Advanced article summarization and classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py article.txt                           # Process single file
  python main.py article.txt -o results.csv           # Custom output
  python main.py --batch *.txt                        # Batch process
  python main.py --list                               # List available files
  python main.py --config config.json article.txt     # Use custom config
        """
    )

    parser.add_argument('input', nargs='?', help='Input text file to process')

    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-c', '--config', help='Configuration file path')
    parser.add_argument('-b', '--batch', nargs='*', help='Batch process multiple files')
    parser.add_argument('-l', '--list', action='store_true', help='List available .txt files')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bars')

    return parser


def interactive_mode(processor: ArticleProcessor):
    """Run interactive mode for file selection."""
    print("\n🤖 AI Article Processor - Interactive Mode")
    print("=" * 50)

    # Show current directory and available files
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")

    available_files = processor.list_available_files()
    if available_files:
        print(f"\n📄 Available .txt files ({len(available_files)}):")
        for i, file in enumerate(available_files, 1):
            print(f"  {i}. {Path(file).name}")
    else:
        print("\n⚠️  No .txt files found in current directory")
        print("💡 Run 'python scrape_article.py' first to create content files")
        return

    # Get user input
    while True:
        try:
            choice = input("\n🎯 Enter file number, filename, or 'q' to quit: ").strip()

            if choice.lower() in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                return

            # Handle number input
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_files):
                    selected_file = available_files[idx]
                else:
                    print(f"❌ Invalid number. Please enter 1-{len(available_files)}")
                    continue
            else:
                # Handle filename input
                if os.path.isabs(choice):
                    selected_file = choice
                else:
                    selected_file = os.path.join(current_dir, choice)

            # Process the file
            print(f"\n🚀 Processing: {Path(selected_file).name}")
            result = processor.process_article(selected_file)

            # Display results
            print("\n" + "="*60)
            print("📊 PROCESSING RESULTS")
            print("="*60)
            print(f"📝 Title: {result['title']}")
            print(f"🏷️  Category: {result['category']}")
            print(f"🚨 Spam: {'Yes' if result['is_spam'] else 'No'}")
            print(f"📄 Chunks: {result['num_chunks']}")
            print(f"⏱️  Time: {result['processing_time_seconds']}s")
            print(f"\n📋 Summary: {result['final_summary']}")
            print("="*60)

            break

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
            return
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.exception("Processing error")


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Disable progress bars if requested
    if args.no_progress:
        tqdm.disable = True

    try:
        # Initialize processor
        processor = ArticleProcessor(args.config)

        # Handle different modes
        if args.list:
            # List mode
            files = processor.list_available_files()
            if files:
                print("📄 Available .txt files:")
                for file in files:
                    print(f"  • {file}")
            else:
                print("⚠️  No .txt files found")

        elif args.batch:
            # Batch mode
            print(f"🚀 Starting batch processing of {len(args.batch)} files...")
            results = processor.batch_process(args.batch)
            successful = sum(1 for r in results if 'error' not in r)
            print(f"✅ Batch processing complete: {successful}/{len(results)} successful")

        elif args.input:
            # Single file mode
            processor.process_article(args.input, args.output)

        else:
            # Interactive mode
            interactive_mode(processor)

    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Fatal error")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()