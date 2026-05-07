# Data Documentation

## Sources

Three YouTube lectures spanning different academic domains:

1. **Machine Learning**: "But what is a neural network?" (3Blue1Brown, 18:39)
   - URL: https://www.youtube.com/watch?v=aircAruvnKk
2. **History**: "Neoliberalism and the End of History - Part 3" (15:06)
   - URL: https://www.youtube.com/watch?v=oIp0_rAEMIs
3. **Psychology**: "Why Having More Doesn't Make You Happier" (Yale Science of Well-Being, 26:34)
   - URL: https://www.youtube.com/watch?v=_5qGVpy4-mc

## Directory contents

- `raw/` — Original auto-generated transcripts from YouTube (JSON with timestamps)
- `processed/` — Cleaned plain-text transcripts ready for summarization

## Regenerating data

Run `notebooks/01_eda.ipynb` cells 1-3 to re-fetch transcripts from YouTube.
Note: YouTube transcripts may change over time as auto-generation models update.
