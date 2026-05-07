"""Summarization model loading and chunked inference."""

import torch
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
)


def load_bart(model_name="facebook/bart-large-cnn", device="cuda"):
    """Load BART model and tokenizer."""
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def load_t5(model_path="google-t5/t5-base", device="cuda"):
    """Load T5 model and tokenizer (works for base or fine-tuned)."""
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer


def chunk_text(text, tokenizer, max_chunk_tokens=1000, overlap_tokens=50):
    """Split text into overlapping chunks fitting model context."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_chunk_tokens
        chunks.append(tokenizer.decode(tokens[start:end], skip_special_tokens=True))
        start += max_chunk_tokens - overlap_tokens
    return chunks


def summarize_chunk(text, model, tokenizer, model_type="bart",
                    max_summary_length=150, min_summary_length=40,
                    no_repeat_ngram_size=3, device="cuda"):
    """Summarize a chunk. Handles BART and T5 (with task prefix)."""
    if model_type == "t5":
        text = "summarize: " + text
        max_input = 512
    else:
        max_input = 1024

    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_input).to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_summary_length,
            min_length=min_summary_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_long_text(text, model, tokenizer, model_type="bart", device="cuda"):
    """Chunk a long document, summarize each chunk, concatenate."""
    max_chunk_tokens = 1000 if model_type == "bart" else 400
    chunks = chunk_text(text, tokenizer, max_chunk_tokens=max_chunk_tokens)
    chunk_summaries = [
        summarize_chunk(chunk, model, tokenizer, model_type=model_type, device=device)
        for chunk in chunks
    ]
    return ' '.join(chunk_summaries), chunk_summaries
