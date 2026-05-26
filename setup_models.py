import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import time
from tqdm import tqdm
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_setup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for a single AI model."""
    name: str
    source: str
    description: str
    size_gb: float
    expected_hash: Optional[str] = None
    required: bool = True

    @property
    def cache_key(self) -> str:
        """Generate cache key for this model."""
        return f"{self.source.replace('/', '_').replace('-', '_')}"

class AIModelManager:
    """Advanced AI model management system."""

    def __init__(self, models_dir: str = "models", config_file: Optional[str] = None):
        """Initialize the model manager."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)

        # Model registry
        self.models = self._get_model_registry()

        # Cache and metadata
        self.cache_file = self.models_dir / "model_cache.json"
        self.metadata_file = self.models_dir / "model_metadata.json"

        # Load existing cache
        self.cache = self._load_cache()
        self.metadata = self._load_metadata()

        logger.info(f"Model Manager initialized - Models directory: {self.models_dir}")

    def _get_model_registry(self) -> Dict[str, ModelConfig]:
        """Get the registry of all required models."""
        return {
            'summarizer': ModelConfig(
                name='DistilBART Summarizer',
                source='sshleifer/distilbart-cnn-12-6',
                description='Text summarization model for generating concise summaries',
                size_gb=1.2,
                required=True
            ),
            'tokenizer': ModelConfig(
                name='BERT Tokenizer',
                source='bert-base-uncased',
                description='Tokenization model for accurate text chunking',
                size_gb=0.4,
                required=True
            ),
            'classifier': ModelConfig(
                name='BART MNLI Classifier',
                source='facebook/bart-large-mnli',
                description='Zero-shot topic classification model',
                size_gb=1.6,
                required=True
            ),
            'sentiment': ModelConfig(
                name='RoBERTa Sentiment',
                source='cardiffnlp/twitter-roberta-base-sentiment-latest',
                description='Sentiment analysis for spam detection',
                size_gb=1.4,
                required=True
            )
        }

    def _load_cache(self) -> Dict[str, Dict]:
        """Load model cache information."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        """Save model cache information."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _load_metadata(self) -> Dict[str, Dict]:
        """Load model metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {}

    def _save_metadata(self):
        """Save model metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _calculate_directory_size(self, path: Path) -> float:
        """Calculate directory size in GB."""
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024**3)  # Convert to GB

    def _verify_model(self, model_key: str) -> bool:
        """Verify if a model is properly downloaded and cached."""
        if model_key not in self.cache:
            return False

        model_info = self.cache[model_key]
        model_path = self.models_dir / model_info.get('local_path', '')

        # Check if directory exists
        if not model_path.exists():
            return False

        # Check size (approximate)
        actual_size = self._calculate_directory_size(model_path)
        expected_size = self.models[model_key].size_gb

        # Allow 20% variance in size
        size_tolerance = expected_size * 0.2
        if abs(actual_size - expected_size) > size_tolerance:
            logger.warning(f"Size mismatch for {model_key}: expected {expected_size:.1f}GB, got {actual_size:.1f}GB")
            return False

        return True

    def _download_with_progress(self, model_key: str) -> bool:
        """Download a model with progress tracking."""
        model_config = self.models[model_key]

        logger.info(f"Downloading {model_config.name}...")
        logger.info(f"Source: {model_config.source}")
        logger.info(f"Estimated size: {model_config.size_gb:.1f}GB")

        try:
            # Import here to avoid dependency issues
            from transformers import pipeline, AutoTokenizer

            start_time = time.time()

            # Create progress-aware download
            with tqdm(total=100, desc=f"Downloading {model_config.name}", unit="%") as pbar:

                # Download based on model type
                if model_key == 'summarizer':
                    model = pipeline("summarization", model=model_config.source, device=-1)
                elif model_key == 'tokenizer':
                    tokenizer = AutoTokenizer.from_pretrained(model_config.source)
                elif model_key == 'classifier':
                    model = pipeline("zero-shot-classification", model=model_config.source, device=-1)
                elif model_key == 'sentiment':
                    model = pipeline("text-classification", model=model_config.source, device=-1)
                else:
                    raise ValueError(f"Unknown model type: {model_key}")

                pbar.update(100)

            # Calculate actual download time and size
            download_time = time.time() - start_time
            model_path = self.models_dir / model_config.cache_key

            # Update cache
            self.cache[model_key] = {
                'name': model_config.name,
                'source': model_config.source,
                'local_path': str(model_path),
                'downloaded_at': time.time(),
                'download_time_seconds': download_time,
                'size_gb': model_config.size_gb
            }

            # Update metadata
            self.metadata[model_key] = {
                'config': {
                    'name': model_config.name,
                    'source': model_config.source,
                    'description': model_config.description,
                    'size_gb': model_config.size_gb
                },
                'download_info': self.cache[model_key],
                'verification_status': 'downloaded'
            }

            self._save_cache()
            self._save_metadata()

            logger.info(f"✓ {model_config.name} downloaded successfully in {download_time:.1f}s")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to download {model_config.name}: {e}")
            return False

    def setup_models(self, force_redownload: bool = False, skip_optional: bool = False) -> bool:
        """
        Setup all required models with intelligent caching.

        Args:
            force_redownload: Force re-download of all models
            skip_optional: Skip optional models

        Returns:
            Success status
        """
        logger.info("Starting AI Model Setup...")
        logger.info("=" * 50)

        # Set Hugging Face cache directory
        os.environ['HF_HOME'] = str(self.models_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(self.models_dir)

        total_size = sum(model.size_gb for model in self.models.values())
        logger.info(f"Total estimated size: {total_size:.1f}GB")

        # Check available disk space
        try:
            import shutil
            disk_usage = shutil.disk_usage(self.models_dir)
            available_gb = disk_usage.free / (1024**3)

            if available_gb < total_size * 1.2:  # 20% buffer
                logger.warning(f"Low disk space: {available_gb:.1f}GB available, {total_size:.1f}GB needed")
                if not self._confirm_action("Continue anyway?"):
                    return False
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

        successful_downloads = 0
        failed_downloads = []

        # Process each model
        for model_key, model_config in self.models.items():
            logger.info(f"\nProcessing {model_config.name}...")

            # Check if already downloaded and valid
            if not force_redownload and self._verify_model(model_key):
                logger.info(f"✓ {model_config.name} already available")
                successful_downloads += 1
                continue

            # Download the model
            if self._download_with_progress(model_key):
                successful_downloads += 1
            else:
                failed_downloads.append(model_key)
                if model_config.required:
                    logger.error(f"Required model {model_config.name} failed to download")
                    return False

        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("MODEL SETUP SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total models: {len(self.models)}")
        logger.info(f"Successful: {successful_downloads}")
        logger.info(f"Failed: {len(failed_downloads)}")

        if failed_downloads:
            logger.warning(f"Failed models: {', '.join(failed_downloads)}")

        if successful_downloads == len(self.models):
            logger.info("🎉 All models downloaded successfully!")
            logger.info(f"Models stored in: {self.models_dir.absolute()}")

            # Create verification script
            self._create_verification_script()
            return True
        else:
            logger.error("❌ Some models failed to download")
            return False

    def _confirm_action(self, message: str) -> bool:
        """Get user confirmation for an action."""
        try:
            response = input(f"{message} (y/N): ").strip().lower()
            return response in ['y', 'yes']
        except KeyboardInterrupt:
            return False

    def _create_verification_script(self):
        """Create a verification script for downloaded models."""
        verify_script = self.models_dir.parent / "verify_models.py"

        script_content = '''#!/usr/bin/env python3
"""
Model Verification Script
Verify that all downloaded models are working correctly.
"""

import sys
import os
sys.path.append('src')

def verify_models():
    """Verify all models can be loaded and used."""
    try:
        from src.summarize import load_summarizer
        from src.categorize import load_category_classifier
        from src.detect_spam import load_spam_detector

        print("🔍 Verifying models...")

        # Test summarizer
        print("Testing summarizer...")
        summarizer = load_summarizer()
        test_text = "This is a test article about artificial intelligence and machine learning."
        result = summarizer(test_text, max_length=20, min_length=10)
        print("✓ Summarizer working")

        # Test classifier
        print("Testing classifier...")
        classifier = load_category_classifier()
        categories = ["Technology", "Science", "Sports"]
        result = classifier(test_text, categories)
        print("✓ Classifier working")

        # Test spam detector
        print("Testing spam detector...")
        detector = load_spam_detector()
        result = detector(test_text)
        print("✓ Spam detector working")

        print("🎉 All models verified successfully!")
        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    verify_models()
'''

        try:
            with open(verify_script, 'w') as f:
                f.write(script_content)
            logger.info(f"Verification script created: {verify_script}")
        except Exception as e:
            logger.warning(f"Failed to create verification script: {e}")

    def list_models(self):
        """List all models and their status."""
        print("\n🤖 AI Models Status")
        print("=" * 60)

        for model_key, model_config in self.models.items():
            status = "✓ Available" if self._verify_model(model_key) else "✗ Missing"
            size_info = f"{model_config.size_gb:.1f}GB"

            print(f"{model_config.name:<25} {status:<12} {size_info:<8} {model_config.description}")

        print("=" * 60)

    def cleanup_models(self, confirm: bool = True) -> bool:
        """Clean up downloaded models."""
        if confirm and not self._confirm_action("This will delete all downloaded models. Continue?"):
            return False

        try:
            import shutil
            shutil.rmtree(self.models_dir)
            self.models_dir.mkdir()
            self.cache.clear()
            self.metadata.clear()
            self._save_cache()
            self._save_metadata()
            logger.info("✓ All models cleaned up")
            return True
        except Exception as e:
            logger.error(f"✗ Cleanup failed: {e}")
            return False

def create_argument_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Model Setup Manager - Advanced model download & management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_models.py                    # Download all models
  python setup_models.py --force           # Force re-download
  python setup_models.py --list            # List model status
  python setup_models.py --cleanup         # Remove all models
  python setup_models.py --verify          # Verify installations
        """
    )

    parser.add_argument('--force', action='store_true', help='Force re-download of all models')
    parser.add_argument('--list', action='store_true', help='List model status')
    parser.add_argument('--cleanup', action='store_true', help='Remove all downloaded models')
    parser.add_argument('--verify', action='store_true', help='Verify model installations')
    parser.add_argument('--models-dir', default='models', help='Models directory path')
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
        # Initialize manager
        manager = AIModelManager(args.models_dir)

        # Handle different operations
        if args.list:
            manager.list_models()
        elif args.cleanup:
            success = manager.cleanup_models()
            if success:
                print("✅ Models cleaned up successfully")
            else:
                print("❌ Cleanup failed")
                sys.exit(1)
        elif args.verify:
            # Run verification
            verify_script = Path(args.models_dir).parent / "verify_models.py"
            if verify_script.exists():
                print("🔍 Running model verification...")
                os.system(f"python {verify_script}")
            else:
                print("❌ Verification script not found. Run setup first.")
                sys.exit(1)
        else:
            # Default: setup models
            success = manager.setup_models(force_redownload=args.force)
            if success:
                print("\n🎉 Model setup completed successfully!")
                print("You can now run the main processing scripts.")
            else:
                print("\n❌ Model setup failed!")
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