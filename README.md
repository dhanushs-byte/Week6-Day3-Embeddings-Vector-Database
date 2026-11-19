# Week 6 Day 3 — Embeddings + Vector Databases

Python implementation for both uploaded Week 6 Day 3 PDFs. It uses a **local JSON index**, not a database or hosted vector database, and does not require an API key.

## Tasks covered

### First half — Chunking
- Extract text from a 10-page PDF using PyMuPDF.
- Fixed-size token chunking with overlap (256/32 and 512/50).
- Sentence-based chunking that keeps whole sentences where possible.
- RecursiveCharacterTextSplitter comparison (512/50).
- Demonstrates how a fixed-size boundary can split a sentence.
- Tests chunk sizes 128, 256, 512 and 1024.
- Retrieves top 3 chunks for 10 questions, measures whether the correct page appears in the top 3, saves a CSV and plot.

### Second half — Metadata and stable upserts
- Stores source, page, section, created date, chunk index, word count, doc_type and ingested_at.
- Detects larger-font headings using PyMuPDF `get_text("dict")`.
- Saves JSON Lines and prints three examples.
- `describe_chunk()` creates citation text.
- Uses deterministic UUID5 chunk IDs.
- Re-indexes unchanged content without increasing the count.
- Demonstrates an edited chunk replacing the same ID.
- Demonstrates deleting chunks by source metadata.
- Includes a SHA-256 file-hash check.

## Project structure

```text
Week6-Day3-Embeddings-Vector-Database/
├── data/
│   └── README.md
├── outputs/
├── create_sample_pdf.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run in VS Code (Windows PowerShell)

```powershell
# 1. Open this project folder in VS Code
# 2. Create virtual environment
python -m venv .venv

# 3. Activate it
.venv\Scripts\Activate.ps1

# 4. Install packages
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Create a 10-page demo PDF (if you do not have your own)
python create_sample_pdf.py

# 6. Run both assignment halves
python main.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model from Hugging Face. No OpenAI key or other API token is needed. Internet is required only for the initial model download; after the model is cached, it can run offline.

## Use your own PDF

Place a 10-page PDF at `data/sample.pdf`, replacing the generated demo file, then run:

```powershell
python main.py
```

The 10 retrieval questions in `main.py` are written for the included demonstration PDF. If you replace it with a different PDF, edit the `questions` list and expected page numbers inside `run_retrieval_experiment()` so each question's answer is on the specified page.

## Output files

After running, check `outputs/`:
- `chunks.jsonl` — one metadata record per line.
- `local_vector_index.json` — local vector index containing stable IDs, metadata, text and embeddings.
- `hit_rate_results.csv` — hit-rate measurements for each chunk size.
- `hit_rate_vs_chunk_size.png` — hit-rate chart.
- `file_hashes.json` — SHA-256 hashes for the bonus check.

These generated outputs are ignored by Git so the repository stays small.

## Important notes

- The assignment's sample expected chunk counts (74, 38 and 36) depend on the exact PDF text and tokenizer. Your counts may differ; do not hard-code those numbers.
- The expected upsert example (120 points) is illustrative. Your point count depends on the PDF and chunking.
- This project uses a JSON file as a simple local vector store to demonstrate stable IDs, upsert and deletion. It is not a production vector database such as FAISS, Chroma or Qdrant.
- For section detection, the included heuristic treats text with font size >= 15 as a heading. Adjust the threshold if your PDF uses different typography.
