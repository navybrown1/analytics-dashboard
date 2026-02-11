# Document Intelligence Dashboard

A local, Python-based analytics dashboard that turns static DOCX files into interactive data products. Created for STA 9708 LN3.1 Rules of Probability.

## Quick Start

### 1) Install Dependencies

Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### 2) Place Document

Place the file `STA 9708 LN3.1 Rules of Probability 2-10-2026.docx` in this folder.
If you don't have it, the app will ask you to upload one.

### 3) Run App

```bash
streamlit run app.py
```

## Features

- Automated Structure Parsing: Splits documents into sections without manual tagging.
- Entity Extraction: Identifies mathematical terms, dates, and proper nouns using regex and heuristics.
- Interactive Graph: Visualizes how concepts link different sections of the document.
- Requirements Checklist: Auto-detects rules, definitions, and must-statements for compliance checks.
- Grounded Q&A: A TF-IDF search engine that answers questions using only the document text.

## Design Notes

### Extraction Pipeline

The `doc_processor.py` module uses a heuristic approach:

- Sectioning: Detects headers based on line length (<60 chars), formatting (heading/all-caps), and numbering (1., 2.).
- Entities: Uses regex patterns specific to probability/math (for example `P(A)`, Union, Intersection), plus capitalized phrase detection.
- Q&A: Uses Scikit-Learn `TfidfVectorizer` to convert sections into vectors and finds highest cosine similarity to a query.

### Confidence and Risk

- Risk score: Calculated from frequency of negative/uncertainty words (`not`, `unlikely`, `error`, `fail`) in each section.
- Q&A confidence: Returns not found if cosine similarity is below `0.1`.

## Project Structure

- `app.py`: Main Streamlit UI code.
- `doc_processor.py`: Backend logic for parsing and NLP.
- `requirements.txt`: Python package list.
