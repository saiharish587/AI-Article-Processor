import os
from transformers import pipeline

def load_spam_detector():
    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0)

def detect_spam(detector, chunk):
    label = detector(chunk[:512])[0]['label']
    return label == 'LABEL_0'  # Assuming LABEL_0 is negative/spam