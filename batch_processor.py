import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add src to path
sys.path.append('src')

from scrape_article import UniversalArticleScraper
from main import ArticleProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BatchArticleProcessor:
    """Batch processing orchestrator for multiple articles."""

    def __init__(self, config_path: str = None, max_workers: int = 2):
        """Initialize batch processor."""
        self.config_path = config_path
        self.max_workers = max_workers

        # Initialize components
        self.scraper = UniversalArticleScraper()
        self.processor = ArticleProcessor(config_path)

        logger.info("Batch Article Processor initialized")

    def load_urls_from_file(self, file_path: str) -> List[str]:
        """Load URLs from a text file."""
        urls = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        urls.append(line)
            logger.info(f"Loaded {len(urls)} URLs from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load URLs from {file_path}: {e}")
            raise

        return urls

    def scrape_and_process_single(self, url: str, output_dir: str = None) -> Dict[str, Any]:
        """Scrape and process a single URL."""
        result = {
            'url': url,
            'success': False,
            'scraping_time': 0,
            'processing_time': 0,
            'error': None,
            'data': None
        }

        try:
            # Scrape article
            scrape_start = time.time()
            success, content = self.scraper.scrape_article(url)
            result['scraping_time'] = time.time() - scrape_start

            if not success:
                result['error'] = f"Scraping failed: {content}"
                return result

            # Save scraped content temporarily
            temp_file = f"temp_{hash(url)}.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(content)

            try:
                # Process the article
                process_start = time.time()
                processed_result = self.processor.process_article(temp_file, output_dir)
                result['processing_time'] = time.time() - process_start

                result['success'] = True
                result['data'] = processed_result

            finally:
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        except Exception as e:
            result['error'] = str(e)
            logger.exception(f"Error processing {url}")

        return result

    def process_batch(self, urls: List[str], output_dir: str = None,
                     use_threads: bool = False) -> List[Dict[str, Any]]:
        """
        Process multiple URLs in batch mode.

        Args:
            urls: List of URLs to process
            output_dir: Output directory for results
            use_threads: Whether to use multi-threading

        Returns:
            List of processing results
        """
        logger.info(f"Starting batch processing of {len(urls)} URLs")
        logger.info(f"Output directory: {output_dir or 'default'}")
        logger.info(f"Multi-threading: {'enabled' if use_threads else 'disabled'}")

        results = []

        if use_threads and len(urls) > 1:
            # Multi-threaded processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self.scrape_and_process_single, url, output_dir)
                    for url in urls
                ]

                for future in tqdm(as_completed(futures), total=len(urls), desc="Processing URLs"):
                    try:
                        result = future.result(timeout=600)  # 10 minute timeout
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Thread execution error: {e}")
                        results.append({
                            'url': 'unknown',
                            'success': False,
                            'error': str(e)
                        })
        else:
            # Sequential processing
            for url in tqdm(urls, desc="Processing URLs"):
                result = self.scrape_and_process_single(url, output_dir)
                results.append(result)

        # Generate summary
        successful = sum(1 for r in results if r['success'])
        total_scraping_time = sum(r['scraping_time'] for r in results if r['success'])
        total_processing_time = sum(r['processing_time'] for r in results if r['success'])

        logger.info("=" * 60)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total URLs: {len(urls)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {len(urls) - successful}")
        logger.info(f"Average scraping time: {total_scraping_time / max(successful, 1):.2f}s")
        logger.info(f"Average processing time: {total_processing_time / max(successful, 1):.2f}s")
        logger.info(f"Total time: {total_scraping_time + total_processing_time:.2f}s")

        return results

    def save_batch_results(self, results: List[Dict[str, Any]], output_file: str):
        """Save batch processing results to file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Batch results saved to: {output_file}")

            # Also save summary CSV
            csv_file = output_file.replace('.json', '_summary.csv')
            summary_data = []
            for result in results:
                summary_data.append({
                    'url': result['url'],
                    'success': result['success'],
                    'scraping_time': result.get('scraping_time', 0),
                    'processing_time': result.get('processing_time', 0),
                    'error': result.get('error', ''),
                    'title': result.get('data', {}).get('title', '') if result.get('data') else '',
                    'category': result.get('data', {}).get('category', '') if result.get('data') else '',
                    'is_spam': result.get('data', {}).get('is_spam', '') if result.get('data') else ''
                })

            import pandas as pd
            df = pd.DataFrame(summary_data)
            df.to_csv(csv_file, index=False)
            logger.info(f"Summary CSV saved to: {csv_file}")

        except Exception as e:
            logger.error(f"Failed to save batch results: {e}")

def create_argument_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Batch Article Processor - Process multiple articles simultaneously",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_processor.py urls.txt                          # Process URLs from file
  python batch_processor.py url1 url2 url3                   # Process individual URLs
  python batch_processor.py urls.txt --threads 4             # Multi-threaded processing
  python batch_processor.py urls.txt --output results/       # Custom output directory
        """
    )

    parser.add_argument('urls', nargs='+', help='URLs to process or file containing URLs')
    parser.add_argument('-o', '--output', help='Output directory for results')
    parser.add_argument('-t', '--threads', type=int, default=2, help='Number of threads for parallel processing')
    parser.add_argument('--no-threads', action='store_true', help='Disable multi-threading')
    parser.add_argument('-c', '--config', help='Configuration file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

    return parser

def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize batch processor
        processor = BatchArticleProcessor(
            config_path=args.config,
            max_workers=args.threads if not args.no_threads else 1
        )

        # Load URLs
        urls = []
        for url_arg in args.urls:
            if os.path.isfile(url_arg):
                # Load from file
                urls.extend(processor.load_urls_from_file(url_arg))
            else:
                # Direct URL
                urls.append(url_arg)

        if not urls:
            print("❌ No URLs provided or file is empty")
            sys.exit(1)

        # Remove duplicates while preserving order
        seen = set()
        urls = [x for x in urls if not (x in seen or seen.add(x))]

        print(f"🚀 Starting batch processing of {len(urls)} URLs...")

        # Process batch
        use_threads = not args.no_threads and len(urls) > 1
        results = processor.process_batch(urls, args.output, use_threads)

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"batch_results_{timestamp}.json"
        processor.save_batch_results(results, output_file)

        # Print summary
        successful = sum(1 for r in results if r['success'])
        print("\n✅ Batch processing completed!")
        print(f"   Successful: {successful}/{len(urls)}")
        print(f"   Results saved to: {output_file}")

        if successful < len(urls):
            print(f"   Failed URLs: {len(urls) - successful}")
            # Could list failed URLs here if needed

    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Fatal error")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()