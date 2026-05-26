import os
from transformers import pipeline

def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", truncation=True, device=0)

def summarize_chunk(summarizer, chunk):
    return summarizer(chunk, max_length=50, min_length=10, do_sample=False)[0]['summary_text']