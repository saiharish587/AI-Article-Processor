# AI Article Processor

AI Article Processor is a Python project for scraping article text from web pages, splitting long text into sentence-safe chunks, summarizing the content, classifying the topic, and checking the result for low-quality or spam-like text.

The project is built around three main steps:

1. Scrape an article from a URL and save it as a text file.
2. Process that text with Hugging Face models to generate a summary and classification.
3. Save the results as both CSV and JSON.

## What this project does

- Extracts readable article content from web pages.
- Breaks long text into chunks without cutting sentences in half.
- Generates summaries for each chunk and a final article summary.
- Classifies the article into a broad category.
- Runs a sentiment-style quality check that is used as a spam flag.
- Supports single-file, interactive, and batch-style workflows.

## Project structure

```text
DLPROJECT-master/
├── main.py                 # Main article processing entry point
├── scrape_article.py       # Web scraper that saves article text to .txt files
├── batch_processor.py      # Batch URL scraper + processor
├── setup_models.py         # Model download and cache management
├── config.json             # Configuration template
├── processed_*.json        # Example output file
└── src/
    ├── chunk_text.py       # Sentence-aware chunking
    ├── summarize.py        # Summary model wrapper
    ├── categorize.py      # Topic classification wrapper
    └── detect_spam.py     # Spam/quality check wrapper
```

## How it works

### 1. Scraping

`scrape_article.py` fetches a URL, detects the page type, extracts the main text, cleans it, and saves it as a `.txt` file.

### 2. Chunking

`src/chunk_text.py` uses NLTK sentence tokenization and a BERT tokenizer to keep chunks under a token limit without splitting sentences.

### 3. Summarization

`src/summarize.py` loads the `sshleifer/distilbart-cnn-12-6` model and summarizes each chunk.

### 4. Classification

`src/categorize.py` uses `facebook/bart-large-mnli` to pick the best label from a fixed set of broad categories such as Technology, Science, Politics, and Health.

### 5. Spam or quality flag

`src/detect_spam.py` uses a RoBERTa sentiment pipeline and treats the negative label as a spam-like signal.

### 6. Output

`main.py` saves the final result as:

- a CSV file with the main fields
- a JSON file with the full result, including chunk summaries

## Requirements

This project targets Python 3.8 or newer.

It uses these libraries:

- `requests`
- `beautifulsoup4`
- `pandas`
- `tqdm`
- `nltk`
- `transformers`
- `torch`

The first run may also need NLTK sentence data such as `punkt`.

## Installation

1. Clone or open the project folder.
2. Create and activate a virtual environment.
3. Install the Python dependencies used by the scripts.

Example:

```bash
pip install requests beautifulsoup4 pandas tqdm nltk transformers torch
```

If NLTK tokenization is missing, download the tokenizer data:

```bash
python -c "import nltk; nltk.download('punkt')"
```

## First-time model setup

Run the model setup script to download and cache the Hugging Face models:

```bash
python setup_models.py
```

This script manages the summarizer, classifier, and sentiment model cache under the local `models/` directory.

## Usage

### 1. Scrape a URL into a text file

```bash
python scrape_article.py https://en.wikipedia.org/wiki/Artificial_intelligence
```

Optional output file:

```bash
python scrape_article.py https://example.com/article -o article.txt
```

### 2. Process a saved text file

```bash
python main.py article.txt
```

Optional output path:

```bash
python main.py article.txt -o results.csv
```

### 3. List local text files

```bash
python main.py --list
```

### 4. Process multiple local text files

```bash
python main.py --batch article1.txt article2.txt
```

### 5. Interactive mode

If you run `main.py` without arguments, it opens an interactive file picker in the terminal.

```bash
python main.py
```

### 6. Batch process URLs

`batch_processor.py` can scrape a list of URLs or accept URLs directly.

```bash
python batch_processor.py urls.txt
python batch_processor.py https://example.com/a https://example.com/b
```

## Configuration

`main.py` accepts a `--config` argument, but it expects a simple JSON object with flat keys such as:

```json
{
  "models_dir": "models",
  "output_dir": ".",
  "max_chunk_tokens": 512,
  "summary_max_length": 50,
  "summary_min_length": 20,
  "device": 0,
  "batch_size": 1,
  "retry_attempts": 3,
  "timeout": 300
}
```

The bundled `config.json` is a broader template, but the current `main.py` loader only reads top-level keys like the example above.

## Output files

When processing a file, the project writes:

- `processed_<name>_<timestamp>.csv`
- `processed_<name>_<timestamp>.json`

The JSON output includes the full result dictionary, including chunk summaries and the configuration used.

## Notes and limitations

- The summarization and classification models are downloaded from Hugging Face and may take time on the first run.
- The code defaults to GPU-style model loading in a few places, so a CUDA-capable setup is ideal.
- `main.py` works on local `.txt` files, not raw URLs. Use `scrape_article.py` first if you start from a web page.
- The project is designed for article-style content, not arbitrary documents or heavily structured pages.

## Logging

The scripts create log files in the project root:

- `article_processor.log`
- `scraper.log`
- `batch_processor.log`
- `model_setup.log`

These logs are useful when scraping fails, a model cannot load, or output generation errors occur.

## Troubleshooting

If processing fails, check the following first:

1. The input file exists and contains text.
2. The required Python packages are installed.
3. The Hugging Face models were downloaded successfully.
4. NLTK sentence tokenization data is available.
5. The source article was scraped cleanly and contains enough meaningful text.

If a web page cannot be scraped, the site may block automated requests or may not expose article content in a simple HTML structure.

## Example workflow

```bash
python setup_models.py
python scrape_article.py https://en.wikipedia.org/wiki/Artificial_intelligence -o ai_article.txt
python main.py ai_article.txt
```

## License

No license file is currently included in the repository.