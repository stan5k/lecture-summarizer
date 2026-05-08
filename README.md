# Lecture Summarization with Fine-tuned T5

A capstone project comparing pre-trained and fine-tuned summarization models
on educational lecture content across multiple academic domains.

## Problem Statement

Some of the leading summarization models (BART, T5) are trained
from a large corpus of data which include new articles. Educational lectures differ from this
distribution in terms of technical terminology, length, structure, and conceptual
density. This project characterizes how fine-tuning T5 on academic content
affects summary quality across diverse lecture domains.

## Dataset

The [ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization) is a text database with over 200k entries of
abstract-article pairs. Of that dataset, a randomized subset (5k abstract-article pairs) was used as training data.
The data was cleaned by removing outliers such as those entries that had abstract with the value '0'. The data was
additionally filtered. When comparing the number of words in the article to that of the abstract, entries were excluded where the 
number of words in the abstract divided by the number of words in the article was over 0.3. At the end of the
data preparation pipeline, the outliers accounted for <3% of the traning subset.

Note: The size of the traning data (i.e. ~5k filtered examples)is small when compared to the
size of the data that was available (~200k). This is a noted limitation and covered in future
work.

## Methodology

We compare three model configurations on three educational lectures
(Machine Learning, History, Psychology):
1. **BART-large-CNN** — domain-tuned baseline (news summarization)
2. **T5-base** — general-purpose multi-task baseline
3. **T5-base fine-tuned** — same model, but fine-tuned with ccdv/arxiv-summarization
   (3 epochs, lr=1e-4, ~5K filtered examples) --> located [here](https://drive.google.com/file/d/1n3CHzND1BRMAwcpSKgFVSqspAothKu9N/view?usp=share_link)

Inputs are YouTube lecture transcripts retrieved via `youtube-transcript-api`.
Long transcripts are processed via overlapping chunking and concatenation.

## Results

- **Wordiness of Fine-tuned T5**: Consistently longer text summaries were
  genereated by the Fine-tuned T5 model which lead to a better reacall, but
  worse precision (see below)
- **Precision-recall trade-off**: BART produces concise summaries with high
  precision but lower content coverage. Fine-tuned T5 produces verbose
  summaries with high content coverage but low precision.
- **Domain skew**: Fine-tuning on STEM-heavy arxiv content improved in-domain
  performance (ML lecture) but degraded out-of-domain performance
  (History lecture exhibited boilerplate fixation and repetition).
- **Metric agreement**: Both ROUGE (lexical) and BERTScore (semantic)
  confirmed the precision-recall trade-off, suggesting it reflects a real
  difference in summarization style rather than a metric artifact.

## MVP

This repository contains a Gradio application powered by `app.py`.

## Setup instructions


Clone the repository and navigate into the project folder:

```bash
git clone <your-repo-url>
cd <your-repo-name>

```bash
pip install -r requirements.txt
python app.py
```

Gradio will print a local URL (typically http://127.0.0.1:7860) and a temporary public URL valid for 72 hours.

### Two demo modes

**Pre-computed mode** (Pre-computed Summaries tab): instant display of summaries already generated for our three demo lectures. Response time <1 second.

**Live mode** (Try Your Own URL tab): paste any YouTube URL with auto-generated captions. Takes 1-3 minutes depending on lecture length. First run loads the model (~30 seconds extra).

> **Note**: Live mode requires the fine-tuned model files at `models/t5_finetuned_v2_final/`. These are not included in the repository due to size. To regenerate them, run `notebooks/03_experiments.ipynb`.


## Credits

- Claude and ChatGPT were both used to generate code and to trouble shoot some coding errors.
  The author(me) designed the data analysis and made the conclusions. 
- arxiv-summarization dataset: `ccdv/arxiv-summarization`
- Pre-trained models: `facebook/bart-large-cnn`, `google-t5/t5-base`
- Lecture sources: 3Blue1Brown (YouTube), MIT OpenCourseWare,
  Yale Online (Science of Well-Being)

## Future work

- Larger, mixed-domain fine-tuning data (humanities + STEM) and more than 5k
- Hierarchical summarization (summarize the summaries). Could help the user experience
- Larger lecture evaluation set with human-written references (instead of Claude summarized)

