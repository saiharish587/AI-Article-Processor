#!/usr/bin/env python3
"""
AI Article Scraper - Universal Web Content Extractor
Advanced web scraping tool with intelligent content extraction and error handling.

Author: AI Assistant
Version: 2.0.0
"""

import requests
from bs4 import BeautifulSoup
import os
import sys
import argparse
import logging
import time
import json
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import re
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingConfig:
    """Configuration for web scraping operations."""
    user_agents: List[str] = None
    timeout: int = 30
    max_retries: int = 3
    delay_between_requests: float = 1.0
    min_content_length: int = 500
    max_content_length: int = 100000

    def __post_init__(self):
        if self.user_agents is None:
            self.user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ]

class UniversalArticleScraper:
    """Advanced web scraper with intelligent content extraction."""

    def __init__(self, config: Optional[ScrapingConfig] = None):
        """Initialize the scraper with configuration."""
        self.config = config or ScrapingConfig()
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        # Content extraction patterns for different site types
        self.content_patterns = {
            'wikipedia': self._extract_wikipedia_content,
            'news': self._extract_news_content,
            'blog': self._extract_blog_content,
            'generic': self._extract_generic_content
        }

        logger.info("Universal Article Scraper initialized")

    def _rotate_user_agent(self):
        """Rotate user agent to avoid detection."""
        import random
        self.session.headers['User-Agent'] = random.choice(self.config.user_agents)

    def _make_request(self, url: str, retry_count: int = 0) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and error handling."""
        try:
            self._rotate_user_agent()

            response = self.session.get(
                url,
                timeout=self.config.timeout,
                allow_redirects=True
            )

            response.raise_for_status()

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logger.warning(f"Unexpected content type: {content_type}")
                return None

            return response

        except requests.exceptions.RequestException as e:
            if retry_count < self.config.max_retries:
                wait_time = (retry_count + 1) * self.config.delay_between_requests
                logger.warning(f"Request failed (attempt {retry_count + 1}): {e}")
                logger.info(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                return self._make_request(url, retry_count + 1)
            else:
                logger.error(f"Request failed after {self.config.max_retries} attempts: {e}")
                return None

    def _detect_site_type(self, url: str, soup: BeautifulSoup) -> str:
        """Detect the type of website for appropriate content extraction."""
        domain = urlparse(url).netloc.lower()

        # Wikipedia detection
        if 'wikipedia.org' in domain:
            return 'wikipedia'

        # News site detection
        news_indicators = [
            soup.find('meta', {'property': 'og:type', 'content': 'article'}),
            soup.find('article'),
            soup.find(class_=re.compile(r'article|story|news')),
            soup.find('time') or soup.find('date'),
            any(tag.get('class', []) for tag in soup.find_all() if 'article' in ' '.join(tag.get('class', [])).lower())
        ]

        if any(news_indicators):
            return 'news'

        # Blog detection
        blog_indicators = [
            soup.find('meta', {'property': 'og:type', 'content': 'blog'}),
            soup.find(class_=re.compile(r'post|entry|blog')),
            soup.find('article') and soup.find('aside'),
        ]

        if any(blog_indicators):
            return 'blog'

        return 'generic'

    def _extract_wikipedia_content(self, soup: BeautifulSoup) -> str:
        """Extract content from Wikipedia pages with improved robustness."""
        # Find main content div
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            logger.warning("Could not find mw-content-text div")
            return ""

        paragraphs = []

        # Extract paragraphs from main content
        for p in content_div.find_all('p'):
            # Skip if paragraph is inside unwanted elements
            parent_classes = []
            for parent in p.parents:
                if hasattr(parent, 'get') and parent.get('class'):
                    parent_classes.extend(parent.get('class', []))

            # Skip references, navigation, and other unwanted content
            skip_classes = ['reflist', 'references', 'navbox', 'vertical-navbox', 'toc']
            if any(cls in ' '.join(parent_classes) for cls in skip_classes):
                continue

            text = p.get_text().strip()
            # More lenient paragraph filtering for Wikipedia
            if len(text) > 20:  # Reduced from 50 to 20 for Wikipedia
                paragraphs.append(text)

        # If we have limited paragraph content, try to get more from other sections
        if len(paragraphs) < 5:
            logger.warning("Limited paragraph content, trying additional extraction")

            # Try to get content from definition lists (dl/dt/dd)
            for dl in content_div.find_all('dl'):
                for dd in dl.find_all('dd'):
                    text = dd.get_text().strip()
                    if len(text) > 50:
                        paragraphs.append(text)

            # Try to get content from list items that contain substantial text
            for li in content_div.find_all('li'):
                # Skip navigation lists
                if li.find_parent('nav') or li.find_parent(class_=re.compile(r'nav|menu')):
                    continue
                text = li.get_text().strip()
                # Only include list items that look like substantial content
                if len(text) > 100 and not text.startswith('[') and not text.startswith('^'):
                    paragraphs.append(text)

        # Try to extract from sections that might contain additional content
        additional_content = []
        for heading in content_div.find_all(['h2', 'h3', 'h4']):
            # Get content after headings
            section_content = []
            sibling = heading.find_next_sibling()

            while sibling and sibling.name not in ['h2', 'h3', 'h4']:
                if sibling.name == 'p':
                    text = sibling.get_text().strip()
                    if len(text) > 30:
                        section_content.append(text)
                elif sibling.name == 'ul':
                    # Extract list items
                    for li in sibling.find_all('li'):
                        text = li.get_text().strip()
                        if len(text) > 50:
                            section_content.append(f"• {text}")
                sibling = sibling.find_next_sibling()

            if section_content:
                additional_content.extend(section_content)

        # Combine main paragraphs with additional content
        all_content = paragraphs + additional_content

        # Remove duplicates while preserving order
        seen = set()
        unique_content = []
        for item in all_content:
            if item not in seen:
                unique_content.append(item)
                seen.add(item)

        content = '\n\n'.join(unique_content)

        if len(content) < 300:
            logger.warning("Wikipedia-specific extraction yielded insufficient content, trying generic extraction")
            return self._extract_generic_content(soup)

        return content

    def _extract_news_content(self, soup: BeautifulSoup) -> str:
        """Extract content from news articles."""
        # Try multiple selectors for article content
        selectors = [
            'article',
            '[class*="article"]',
            '[class*="story"]',
            '[class*="content"]',
            'main',
            '.post-content',
            '.entry-content'
        ]

        for selector in selectors:
            content_element = soup.select_one(selector)
            if content_element:
                # Extract paragraphs from the content
                paragraphs = []
                for p in content_element.find_all('p'):
                    text = p.get_text().strip()
                    if len(text) > 30:  # Filter out short/meaningless paragraphs
                        paragraphs.append(text)

                if paragraphs:
                    return '\n\n'.join(paragraphs)

        # Fallback to generic extraction
        return self._extract_generic_content(soup)

    def _extract_blog_content(self, soup: BeautifulSoup) -> str:
        """Extract content from blog posts."""
        # Similar to news but with blog-specific selectors
        selectors = [
            '.post-content',
            '.entry-content',
            '.blog-content',
            'article .content',
            '[class*="post"] p'
        ]

        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                text = ' '.join([elem.get_text().strip() for elem in elements])
                if len(text) > 200:
                    return text

        return self._extract_generic_content(soup)

    def _extract_generic_content(self, soup: BeautifulSoup) -> str:
        """Generic content extraction for unknown site types."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()

        # Find the main content area (largest text block)
        text_blocks = []
        for element in soup.find_all(['p', 'div', 'article', 'section']):
            text = element.get_text().strip()
            if len(text) > 100:  # Only consider substantial text blocks
                text_blocks.append((element, text))

        if not text_blocks:
            return ""

        # Sort by text length and return the largest block
        text_blocks.sort(key=lambda x: len(x[1]), reverse=True)
        return text_blocks[0][1]

    def _clean_content(self, content: str) -> str:
        """Clean and normalize extracted content."""
        if not content:
            return ""

        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

        # Remove special characters but keep basic punctuation
        content = re.sub(r'[^\w\s.,!?-]', ' ', content)

        # Fix spacing issues
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\s*([.,!?])', r'\1', content)

        # Remove very short lines that are likely noise
        lines = [line.strip() for line in content.split('\n') if len(line.strip()) > 20]
        content = '\n\n'.join(lines)

        return content.strip()

    def _validate_content(self, content: str, url: str = "") -> bool:
        """Validate extracted content quality."""
        if not content:
            return False

        content_length = len(content)

        # More lenient validation for Wikipedia pages (they can be very long)
        is_wikipedia = 'wikipedia.org' in url.lower()

        min_length = 200 if is_wikipedia else self.config.min_content_length  # 200 for wiki, 500 for others
        max_length = 500000 if is_wikipedia else self.config.max_content_length  # 500K for wiki, 100K for others

        # Check length constraints
        if content_length < min_length:
            logger.warning(f"Content too short: {content_length} characters (min: {min_length})")
            return False

        if content_length > max_length:
            logger.warning(f"Content too long: {content_length} characters (max: {max_length})")
            logger.info("Article is very long - this is normal for comprehensive Wikipedia pages")
            # For extremely long content, we'll allow it but log a warning
            # The processing pipeline will handle chunking anyway

        # Check for meaningful content (not just navigation/menu text)
        words = content.split()
        min_words = 20 if is_wikipedia else 50  # More lenient for Wikipedia

        if len(words) < min_words:
            logger.warning(f"Content has too few words: {len(words)} (min: {min_words})")
            return False

        # Check for excessive repetition (spam indicator) - more lenient for wiki
        unique_words = set(words)
        min_unique_ratio = 0.15 if is_wikipedia else 0.3  # More lenient for Wikipedia (encyclopedic content can have repetition)

        unique_ratio = len(unique_words) / len(words)
        if unique_ratio < min_unique_ratio:
            logger.warning(f"Content appears to have excessive repetition (unique ratio: {unique_ratio:.2f})")
            return False

        return True

    def scrape_article(self, url: str) -> Tuple[bool, str]:
        """
        Scrape article content from a given URL.

        Args:
            url: The URL to scrape

        Returns:
            Tuple of (success: bool, content: str)
        """
        logger.info(f"Starting scrape: {url}")

        # Validate URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False, "Invalid URL format"
        except Exception as e:
            return False, f"URL parsing error: {e}"

        # Make the request
        response = self._make_request(url)
        if not response:
            return False, "Failed to fetch URL after retries"

        try:
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Detect site type and extract content
            site_type = self._detect_site_type(url, soup)
            logger.info(f"Detected site type: {site_type}")

            extractor = self.content_patterns.get(site_type, self._extract_generic_content)
            raw_content = extractor(soup)

            # If Wikipedia extraction failed or gave very little content, try generic extraction
            if site_type == 'wikipedia' and (not raw_content or len(raw_content) < 300):
                logger.warning("Wikipedia-specific extraction yielded insufficient content, trying generic extraction")
                generic_content = self._extract_generic_content(soup)
                if len(generic_content) > len(raw_content or ""):
                    logger.info("Using generic extraction as fallback")
                    raw_content = generic_content

            # Debug information
            logger.debug(f"Raw content length: {len(raw_content)}")
            logger.debug(f"First 200 chars: {raw_content[:200] if raw_content else 'None'}")

            # Clean and validate content
            cleaned_content = self._clean_content(raw_content)

            logger.debug(f"Cleaned content length: {len(cleaned_content)}")
            logger.debug(f"Word count: {len(cleaned_content.split()) if cleaned_content else 0}")

            if not self._validate_content(cleaned_content, url):
                logger.warning(f"Raw content length: {len(raw_content)}")
                logger.warning(f"Cleaned content length: {len(cleaned_content)}")
                logger.warning(f"Word count: {len(cleaned_content.split()) if cleaned_content else 0}")
                return False, "Extracted content failed validation"

            logger.info(f"Successfully extracted {len(cleaned_content)} characters")
            return True, cleaned_content

        except Exception as e:
            logger.error(f"Content extraction error: {e}")
            return False, f"Content extraction failed: {e}"

    def scrape_to_file(self, url: str, output_path: Optional[str] = None) -> bool:
        """
        Scrape article and save to file.

        Args:
            url: URL to scrape
            output_path: Optional custom output path

        Returns:
            Success status
        """
        success, content = self.scrape_article(url)

        if not success:
            logger.error(f"Scraping failed: {content}")
            return False

        # Generate filename from URL
        if not output_path:
            domain = urlparse(url).netloc
            path_parts = urlparse(url).path.strip('/').split('/')
            title = path_parts[-1] if path_parts else 'article'
            title = re.sub(r'[^\w\-_]', '_', title)  # Sanitize filename
            output_path = f"{title}.txt"

        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save content
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"Article saved to: {output_file.absolute()}")
            logger.info(f"Content length: {len(content)} characters")
            return True

        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            return False

def create_argument_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Universal Article Scraper - Advanced web content extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scrape_article.py https://en.wikipedia.org/wiki/AI
  python scrape_article.py https://example.com/article -o custom_name.txt
  python scrape_article.py --batch urls.txt
  python scrape_article.py --interactive
        """
    )

    parser.add_argument('url', nargs='?', help='URL to scrape')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-b', '--batch', help='Batch file with URLs (one per line)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--config', help='Configuration file path')

    return parser

def interactive_mode(scraper: UniversalArticleScraper):
    """Run interactive scraping mode."""
    print("\n🕷️  Universal Article Scraper - Interactive Mode")
    print("=" * 55)

    while True:
        try:
            url = input("\n🌐 Enter article URL (or 'q' to quit): ").strip()

            if url.lower() in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break

            if not url:
                continue

            # Validate URL format
            if not url.startswith(('http://', 'https://')):
                print("❌ Please enter a valid URL starting with http:// or https://")
                continue

            print(f"\n🚀 Scraping: {url}")
            success = scraper.scrape_to_file(url)

            if success:
                print("✅ Article scraped successfully!")
            else:
                print("❌ Scraping failed. Check the logs for details.")

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_mode(scraper: UniversalArticleScraper, batch_file: str):
    """Process multiple URLs from a batch file."""
    try:
        with open(batch_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if not urls:
            print("❌ No valid URLs found in batch file")
            return

        print(f"🚀 Starting batch processing of {len(urls)} URLs...")

        successful = 0
        for url in tqdm(urls, desc="Scraping articles"):
            if scraper.scrape_to_file(url):
                successful += 1

        print(f"✅ Batch processing complete: {successful}/{len(urls)} successful")

    except FileNotFoundError:
        print(f"❌ Batch file not found: {batch_file}")
    except Exception as e:
        print(f"❌ Batch processing error: {e}")

def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load configuration if provided
        config = None
        if args.config:
            # Could implement config loading here
            pass

        # Initialize scraper
        scraper = UniversalArticleScraper(config)

        # Handle different modes
        if args.batch:
            batch_mode(scraper, args.batch)
        elif args.interactive or not args.url:
            interactive_mode(scraper)
        else:
            # Single URL mode
            success = scraper.scrape_to_file(args.url, args.output)
            if success:
                print("✅ Article scraped successfully!")
            else:
                print("❌ Scraping failed!")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Fatal error")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()