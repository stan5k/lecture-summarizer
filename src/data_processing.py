"""Data processing: transcript fetching, cleaning, filtering."""

import re
import json
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11}).*', url)
    return match.group(1) if match else None


def fetch_transcript(url):
    """Fetch a transcript from a YouTube URL.

    Returns a list of dicts with 'text', 'start', 'duration' keys.
    """
    api = YouTubeTranscriptApi()
    video_id = extract_video_id(url)
    fetched = api.fetch(video_id)
    return [
        {"text": s.text, "start": s.start, "duration": s.duration}
        for s in fetched
    ]


def clean_transcript(segments):
    """Join transcript segments into clean running text."""
    raw_text = ' '.join(seg['text'] for seg in segments)
    cleaned = raw_text.replace('\n', ' ')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def filter_arxiv_example(example,
                         min_article_words=200,
                         max_article_words=15000,
                         min_abstract_words=50,
                         max_abstract_words=400,
                         max_compression_ratio=0.3):
    """Filter for arxiv-summarization examples used in fine-tuning."""
    article_words = len(example['article'].split())
    abstract_words = len(example['abstract'].split())

    if article_words < min_article_words or article_words > max_article_words:
        return False
    if abstract_words < min_abstract_words or abstract_words > max_abstract_words:
        return False
    if abstract_words / article_words > max_compression_ratio:
        return False
    return True
