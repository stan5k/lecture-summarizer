"""Lecture Summarizer — Gradio MVP

Showcases the fine-tuned T5-base model trained on arxiv-summarization,
with two modes:
1. Pre-computed: instant display of summaries for 3 demo lectures (<1s response)
2. Live: paste a YouTube URL and watch the pipeline process in real-time (~1-3 min)

Usage:
    python app.py
"""

import json
import os
import re
import time
import gradio as gr

# ---- Configuration ----
PROJECT_ROOT = PROJECT_DIR # Colab specific: __file__ is not defined when using exec()
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

LECTURES = {
    "Machine Learning — But what is a neural network? (3Blue1Brown)": "ml_neural_network",
    "History — Neoliberalism and the End of History": "history_neoliberalism",
    "Psychology — Why Having More Doesn\'t Make You Happier": "psych_wellbeing",
}

LECTURE_URLS = {
    "ml_neural_network": "https://www.youtube.com/watch?v=aircAruvnKk",
    "history_neoliberalism": "https://www.youtube.com/watch?v=oIp0_rAEMIs",
    "psych_wellbeing": "https://www.youtube.com/watch?v=_5qGVpy4-mc",
}

PRECOMPUTED_FILE = "t5_finetuned_v2_summaries.json"


# ---- Pre-computed summaries (loaded at startup) ----

def load_precomputed():
    filepath = os.path.join(PROCESSED_DIR, PRECOMPUTED_FILE)
    try:
        with open(filepath) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found")
        return {}


def load_transcript(lecture_id):
    path = os.path.join(PROCESSED_DIR, f"{lecture_id}.txt")
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return "Transcript not available."


PRECOMPUTED = load_precomputed()


def get_precomputed_summary(lecture_label):
    """Return pre-computed fine-tuned T5 summary for a lecture."""
    if not lecture_label:
        return "Please select a lecture.", "", "", ""
    
    lecture_id = LECTURES[lecture_label]
    record = PRECOMPUTED.get(lecture_id)
    
    if record is None:
        return f"No pre-computed summary for {lecture_label}.", "", "", ""
    
    summary = record["full_summary"]
    word_count = record["summary_word_count"]
    n_chunks = record["n_chunks"]
    model_id = record["model"]
    
    transcript = load_transcript(lecture_id)
    input_words = len(transcript.split())
    compression = (word_count / input_words * 100) if input_words > 0 else 0
    
    metadata = (
        f"**Model**: {model_id}  \n"
        f"**Input length**: {input_words:,} words  \n"
        f"**Output length**: {word_count} words  \n"
        f"**Compression**: {compression:.1f}%  \n"
        f"**Chunks processed**: {n_chunks}"
    )
    
    url = LECTURE_URLS.get(lecture_id, "")
    url_md = f"[Watch original lecture on YouTube]({url})" if url else ""
    
    return summary, metadata, url_md, transcript[:500] + "..."


# ---- Live processing (lazy-loaded fine-tuned T5) ----

_LIVE = {"loaded": False}


def _ensure_model_loaded():
    """Lazy-load the fine-tuned T5 on first live request."""
    if _LIVE["loaded"]:
        return None
    
    print("Loading fine-tuned T5 for live processing...")
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    
    ft_path = os.path.join(MODELS_DIR, "t5_finetuned_v2_final")
    if not os.path.exists(ft_path):
        return f"Fine-tuned model not found at {ft_path}. Live processing unavailable."
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _LIVE["device"] = device
    _LIVE["torch"] = torch
    _LIVE["tokenizer"] = T5Tokenizer.from_pretrained(ft_path)
    _LIVE["model"] = T5ForConditionalGeneration.from_pretrained(ft_path).to(device).eval()
    _LIVE["loaded"] = True
    print(f"Model loaded on {device}")
    return None  # success


def _extract_video_id(url):
    match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11}).*', url)
    return match.group(1) if match else None


def _chunk_text(text, tokenizer, max_chunk_tokens=400, overlap_tokens=30):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_chunk_tokens
        chunks.append(tokenizer.decode(tokens[start:end], skip_special_tokens=True))
        start += max_chunk_tokens - overlap_tokens
    return chunks


def _summarize(text):
    """Summarize text using the loaded fine-tuned T5."""
    torch = _LIVE["torch"]
    device = _LIVE["device"]
    tokenizer = _LIVE["tokenizer"]
    model = _LIVE["model"]
    
    chunks = _chunk_text(text, tokenizer, max_chunk_tokens=400)
    summaries = []
    for chunk in chunks:
        prefixed = "summarize: " + chunk
        inputs = tokenizer(prefixed, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150, min_length=40,
                num_beams=4, length_penalty=2.0,
                early_stopping=True, no_repeat_ngram_size=3,
            )
        summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return ' '.join(summaries), len(chunks)


def process_live_url(url, progress=gr.Progress()):
    """Run the fine-tuned T5 pipeline live on a user-provided YouTube URL."""
    if not url or not url.strip():
        return "Please enter a YouTube URL.", ""
    
    progress(0.05, desc="Validating URL...")
    video_id = _extract_video_id(url)
    if not video_id:
        return "Could not parse a valid YouTube video ID from that URL.", ""
    
    try:
        progress(0.1, desc="Loading model (first run takes ~30 seconds)...")
        load_error = _ensure_model_loaded()
        if load_error:
            return load_error, ""
        
        progress(0.3, desc="Fetching transcript from YouTube...")
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id)
        transcript = ' '.join(s.text for s in fetched)
        transcript = re.sub(r'\s+', ' ', transcript.replace('\n', ' ')).strip()
        word_count = len(transcript.split())
        
        if word_count < 100:
            return f"Transcript too short ({word_count} words). Need at least 100 words.", ""
        
        progress(0.5, desc=f"Summarizing {word_count:,} words...")
        start = time.time()
        summary, n_chunks = _summarize(transcript)
        elapsed = time.time() - start
        
        progress(1.0, desc="Done")
        
        summary_words = len(summary.split())
        compression = (summary_words / word_count * 100) if word_count > 0 else 0
        
        info = (
            f"**Input**: {word_count:,} words from video `{video_id}`  \n"
            f"**Output**: {summary_words} words ({n_chunks} chunks processed)  \n"
            f"**Compression**: {compression:.1f}%  \n"
            f"**Processing time**: {elapsed:.1f} seconds"
        )
        
        return info, summary
    
    except Exception as e:
        return f"Error during processing: {type(e).__name__}: {e}", ""


# ---- Gradio interface ----

with gr.Blocks(title="Lecture Summarizer") as demo:
    gr.Markdown("# Lecture Summarizer")
    gr.Markdown(
        "A capstone project demonstrating a T5 model fine-tuned on the arxiv-summarization "
        "dataset (3 epochs, ~4,300 examples) applied to educational lecture content. "
        "Use the tabs below to view pre-computed summaries (instant) or process a new "
        "YouTube URL (~1-3 minutes)."
    )
    
    with gr.Tab("Pre-computed Summaries"):
        gr.Markdown(
            "### How to use\n"
            "Pick one of the three demo lectures and click **Show Summary**. "
            "These were generated by the fine-tuned T5 model and are cached for instant retrieval."
        )
        
        lecture_dropdown = gr.Dropdown(
            choices=list(LECTURES.keys()),
            label="Lecture",
            value=list(LECTURES.keys())[0],
        )
        submit_btn = gr.Button("Show Summary", variant="primary")
        
        with gr.Row():
            with gr.Column(scale=2):
                summary_output = gr.Textbox(label="Summary", lines=12)
            with gr.Column(scale=1):
                metadata_output = gr.Markdown()
                url_output = gr.Markdown()
        
        with gr.Accordion("Original transcript (first 500 chars)", open=False):
            transcript_preview = gr.Textbox(label="Transcript preview", lines=6)
        
        submit_btn.click(
            fn=get_precomputed_summary,
            inputs=[lecture_dropdown],
            outputs=[summary_output, metadata_output, url_output, transcript_preview],
        )
    
    with gr.Tab("Try Your Own URL (Live)"):
        gr.Markdown(
            "### Live processing mode\n"
            "Paste any YouTube URL with auto-generated captions. The fine-tuned T5 model "
            "will process the transcript and produce a summary.\n\n"
            "**Expected time: 1-3 minutes** depending on lecture length. "
            "First run loads the model (~30 seconds extra)."
        )
        
        url_input = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        live_btn = gr.Button("Process URL", variant="primary")
        live_info = gr.Markdown()
        live_summary = gr.Textbox(label="Generated Summary", lines=12)
        
        live_btn.click(
            fn=process_live_url,
            inputs=[url_input],
            outputs=[live_info, live_summary],
        )
    
    with gr.Tab("About"):
        gr.Markdown("""
## About this project

A capstone exploring the effects of domain-specific fine-tuning on summarization quality.

**The model**: `T5-base` fine-tuned on `ccdv/arxiv-summarization` (3 epochs, lr 1e-4, ~4,300 filtered examples).

**Pipeline**: YouTube transcript → cleaning → overlapping chunking (400 tokens, 30 overlap) → per-chunk summarization with no_repeat_ngram_size=3 → concatenation.

**Findings from full evaluation** (in project notebooks):
- The fine-tuned model produces significantly longer, more content-rich summaries than off-the-shelf BART or T5 baselines (highest recall in both ROUGE and BERTScore).
- It also exhibits domain-skew failures: arxiv-trained model handles STEM content well but produces repetition and boilerplate fixation on humanities content (e.g., the History lecture).

**Pre-computed mode**: shows summaries cached from the fine-tuned model — instant response.
**Live mode**: runs the full pipeline on any YouTube URL with available captions.

See the project repository for full evaluation, error analysis, and the comparison with baseline models.
""")


if __name__ == "__main__":
    demo.launch()
