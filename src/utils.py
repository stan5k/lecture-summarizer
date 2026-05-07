"""Shared utilities: file I/O and metadata."""

import json
from datetime import datetime


def save_json(data, path):
    """Save dict to JSON file with indentation."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path):
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def make_summary_record(lecture_metadata, model_name, config,
                        chunk_summaries, full_summary):
    """Build a standardized record for a summary result."""
    return {
        "lecture_metadata": lecture_metadata,
        "model": model_name,
        "config": config,
        "n_chunks": len(chunk_summaries),
        "chunk_summaries": chunk_summaries,
        "full_summary": full_summary,
        "summary_word_count": len(full_summary.split()),
        "timestamp": datetime.now().isoformat(),
    }
