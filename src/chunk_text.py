import os
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

def chunk_text(text, max_tokens=512, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                # If a single sentence is too long, split it (though unlikely for sentences)
                chunks.append(sentence)
        else:
            current_chunk += " " + sentence
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks