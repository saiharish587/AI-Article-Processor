import os
from transformers import pipeline

def load_category_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

def categorize_chunk(classifier, chunk):
    candidate_labels = ["Technology", "Science", "History", "Art", "Sports", "Politics", "Education", "Health", "Business", "Entertainment"]
    result = classifier(chunk[:512], candidate_labels=candidate_labels)
    return result['labels'][0]  # Top predicted label