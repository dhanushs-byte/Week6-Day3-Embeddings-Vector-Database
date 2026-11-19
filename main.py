from pathlib import Path
from datetime import datetime, date, timezone
import hashlib
import json
import re
import uuid

import fitz
import numpy as np
import tiktoken
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
PDF_PATH = DATA_DIR / "sample.pdf"
CHUNKS_FILE = OUTPUT_DIR / "chunks.jsonl"
INDEX_FILE = OUTPUT_DIR / "local_vector_index.json"
FILE_HASHES_FILE = OUTPUT_DIR / "file_hashes.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

OUTPUT_DIR.mkdir(exist_ok=True)
ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def extract_pdf_pages(pdf_path: Path):
    """Return one dict per page, preserving the page number and detected headings."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_number, page in enumerate(doc, start=1):
        page_text = page.get_text("text").strip()
        blocks = page.get_text("dict").get("blocks", [])
        headings = []
        for block in blocks:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(span.get("text", "") for span in spans).strip()
                if not line_text:
                    continue
                max_size = max((span.get("size", 0) for span in spans), default=0)
                # Heuristic: larger text than ordinary body text is a heading.
                if max_size >= 15 and len(line_text) < 120:
                    headings.append(line_text)
        pages.append({
            "page": page_number,
            "text": page_text,
            "headings": headings,
        })
    doc.close()
    return pages


def split_fixed_tokens(text: str, size: int, overlap: int):
    """Fixed-size token chunks with overlap. The returned strings are decoded tokens."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be > 0 and 0 <= overlap < size")
    tokens = ENCODING.encode(text)
    chunks = []
    step = size - overlap
    for start in range(0, len(tokens), step):
        piece = tokens[start:start + size]
        if not piece:
            break
        chunks.append(ENCODING.decode(piece))
        if start + size >= len(tokens):
            break
    return chunks


def split_sentence_tokens(text: str, max_tokens: int):
    """Group complete sentences; a long single sentence is split as a fallback."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    result, current = [], []
    for sentence in sentences:
        if not sentence:
            continue
        if count_tokens(sentence) > max_tokens:
            if current:
                result.append(" ".join(current))
                current = []
            result.extend(split_fixed_tokens(sentence, max_tokens, 0))
            continue
        candidate = " ".join(current + [sentence])
        if current and count_tokens(candidate) > max_tokens:
            result.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        result.append(" ".join(current))
    return result


def split_recursive(text: str, chunk_size=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def build_chunk_records(pdf_path: Path):
    pages = extract_pdf_pages(pdf_path)
    records = []
    chunk_index = 0
    for page in pages:
        page_text = page["text"]
        if not page_text:
            continue
        # Paragraph chunks are used for metadata/upsert demo; retain page-level provenance.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]
        if not paragraphs:
            paragraphs = [page_text]
        section = page["headings"][0] if page["headings"] else "Unclassified"
        for paragraph in paragraphs:
            # Keep metadata chunks reasonably sized.
            for piece in split_fixed_tokens(paragraph, 256, 32):
                chunk_index += 1
                records.append({
                    "source": pdf_path.name,
                    "page": page["page"],
                    "section": section,
                    "created_date": date.today().isoformat(),
                    "chunk_index": chunk_index,
                    "word_count": len(piece.split()),
                    "doc_type": pdf_path.suffix.lstrip(".").upper(),
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "text": piece,
                })
    return records


def describe_chunk(chunk: dict) -> str:
    return f'{chunk["source"]}, page {chunk["page"]}, {chunk["section"]}'


def stable_chunk_id(chunk: dict) -> str:
    key = f'{chunk["source"]}:{chunk["page"]}:{chunk["chunk_index"]}'
    return str(uuid.uuid5(NAMESPACE, key))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(records, path: Path):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_index():
    return load_json(INDEX_FILE, {})


def upsert_records(records, model, index):
    """Local JSON index: deterministic IDs overwrite existing entries."""
    texts = [r["text"] for r in records]
    vectors = model.encode(texts, normalize_embeddings=True).tolist() if texts else []
    replaced = 0
    for record, vector in zip(records, vectors):
        chunk_id = stable_chunk_id(record)
        old = index.get(chunk_id)
        if old and old.get("text") != record["text"]:
            replaced += 1
        index[chunk_id] = {
            "id": chunk_id,
            "source": record["source"],
            "page": record["page"],
            "section": record["section"],
            "text": record["text"],
            "embedding": vector,
        }
    return index, replaced


def remove_source(index, source_name):
    before = len(index)
    index = {key: value for key, value in index.items() if value.get("source") != source_name}
    return index, before - len(index)


def retrieve_top3(question, chunks, model):
    query_vector = model.encode([question], normalize_embeddings=True)
    vectors = np.array([c["embedding"] for c in chunks])
    scores = cosine_similarity(query_vector, vectors)[0]
    top_indices = np.argsort(scores)[::-1][:3]
    return [chunks[i] for i in top_indices]


def run_chunking_experiment(pages):
    full_text = "\n\n".join(p["text"] for p in pages)
    fixed_256 = split_fixed_tokens(full_text, 256, 32)
    fixed_512 = split_fixed_tokens(full_text, 512, 50)
    sentence = split_sentence_tokens(full_text, 512)
    recursive = split_recursive(full_text, 512, 50)

    print("\nTASK 1 — CHUNKING IMPLEMENTATIONS")
    print(f"Fixed 256/32: {len(fixed_256)} chunks")
    print(f"Fixed 512/50: {len(fixed_512)} chunks")
    print(f"Sentence chunks (max 512 tokens): {len(sentence)} chunks")
    print(f"Recursive 512/50: {len(recursive)} chunks")

    # Demonstrate a sentence being cut by a fixed token boundary.
    demo = ("This is a deliberately long example sentence that demonstrates how "
            "fixed-size chunking can split a sentence in the middle. " * 20)
    demo_chunks = split_fixed_tokens(demo, 25, 0)
    print("\nFixed-size boundary example:")
    print("Chunk 1 ending:", repr(demo_chunks[0][-70:]))
    print("Chunk 2 beginning:", repr(demo_chunks[1][:70]))
    print("A sentence may continue across the boundary because fixed chunking follows token count.")


def run_retrieval_experiment(pdf_path, pages, model):
    """Ten questions are paired with the page where their answer is located."""
    questions = [
        ("What does the introduction explain?", 1),
        ("How do users sign in to their account?", 2),
        ("How can a user reset a forgotten password?", 3),
        ("Where can users download monthly invoices?", 4),
        ("How can users update an expired payment method?", 5),
        ("Where can a subscription and renewal date be reviewed?", 6),
        ("How should personal information be protected?", 7),
        ("What should users do when a page does not load?", 8),
        ("What information should a support request include?", 9),
        ("Why are stable identifiers useful when maintaining documents?", 10),
    ]

    print("\nTASK 2 — CHUNK SIZE RETRIEVAL EXPERIMENT")
    rows = []
    for size in [128, 256, 512, 1024]:
        all_chunks = []
        for page in pages:
            for text in split_fixed_tokens(page["text"], size, min(50, size // 4)):
                if text.strip():
                    all_chunks.append({
                        "page": page["page"],
                        "text": text,
                    })
        texts = [c["text"] for c in all_chunks]
        embeddings = model.encode(texts, normalize_embeddings=True)
        for c, emb in zip(all_chunks, embeddings):
            c["embedding"] = emb.tolist()

        hits = 0
        for question, expected_page in questions:
            top3 = retrieve_top3(question, all_chunks, model)
            if any(item["page"] == expected_page for item in top3):
                hits += 1
        hit_rate = hits / len(questions)
        rows.append((size, hits, len(questions), hit_rate))
        print(f"Chunk size {size:4}: hit rate {hits}/{len(questions)} = {hit_rate:.0%}")

    with (OUTPUT_DIR / "hit_rate_results.csv").open("w", encoding="utf-8") as f:
        f.write("chunk_size,hits,total_questions,hit_rate\n")
        for size, hits, total, rate in rows:
            f.write(f"{size},{hits},{total},{rate:.4f}\n")

    plt.figure(figsize=(8, 5))
    plt.plot([r[0] for r in rows], [r[3] for r in rows], marker="o")
    plt.xlabel("Chunk size (tokens)")
    plt.ylabel("Hit rate (correct page in top 3)")
    plt.title("Retrieval hit rate vs chunk size")
    plt.xticks([128, 256, 512, 1024])
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hit_rate_vs_chunk_size.png", dpi=150)
    plt.close()
    print("Saved outputs/hit_rate_results.csv and outputs/hit_rate_vs_chunk_size.png")
    print("Recommendation: start with 256 or 512 tokens; choose the size with the best measured hit rate,")
    print("then inspect answer context. Smaller chunks improve focus; larger chunks preserve more context.")


def run_metadata_and_upsert(pdf_path, model):
    print("\nTASK 3 — METADATA EXTRACTION")
    records = build_chunk_records(pdf_path)
    write_jsonl(records, CHUNKS_FILE)
    print(f"Created {len(records)} metadata chunks at {CHUNKS_FILE}")
    print("Three JSONL examples:")
    for record in records[:3]:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    if records:
        print("Citation example:", describe_chunk(records[0]))

    print("\nTASK 4 — STABLE IDS AND UPSERTS")
    index = load_index()
    index, replaced1 = upsert_records(records, model, index)
    save_json(INDEX_FILE, index)
    print(f"Run 1: {len(index)} points | {replaced1} replaced")

    index = load_index()
    index, replaced2 = upsert_records(records, model, index)
    save_json(INDEX_FILE, index)
    print(f"Run 2 (no change): {len(index)} points | {replaced2} replaced")

    # Simulate editing one paragraph in one existing chunk.
    edited = [dict(r) for r in records]
    if edited:
        edited[0]["text"] += " Updated during the re-indexing demonstration."
    index, replaced3 = upsert_records(edited, model, index)
    save_json(INDEX_FILE, index)
    print(f"Run 3 (one text edit): {len(index)} points | {replaced3} replaced")
    print("Note: the ID is based on source/page/chunk_index, so editing the text overwrites the same ID.")

    # Demonstrate deletion by source metadata.
    index, deleted = remove_source(index, "__removed_document__.pdf")
    save_json(INDEX_FILE, index)
    print(f"Removed-file demo: deleted {deleted} chunks for __removed_document__.pdf")
    print("\nWhy random IDs create duplicates (4 lines):")
    print("1. A random ID changes every time the same chunk is processed.")
    print("2. The index cannot recognize that the new chunk already exists.")
    print("3. Re-indexing inserts another record instead of updating the old one.")
    print("4. Stable IDs let upsert replace the existing record and prevent duplicates.")

    hashes = load_json(FILE_HASHES_FILE, {})
    current_hash = file_sha256(pdf_path)
    previous_hash = hashes.get(pdf_path.name)
    if previous_hash == current_hash:
        print("Bonus: file hash unchanged — this file can be skipped on a future ingestion run.")
    else:
        print("Bonus: file hash changed/new — process this file and save its hash.")
        hashes[pdf_path.name] = current_hash
        save_json(FILE_HASHES_FILE, hashes)


def main():
    if not PDF_PATH.exists():
        print("PDF not found:", PDF_PATH)
        print("Run `python create_sample_pdf.py` to create a 10-page demo PDF,")
        print("or place your own 10-page PDF at data/sample.pdf.")
        return

    pages = extract_pdf_pages(PDF_PATH)
    print(f"Loaded {len(pages)} pages from {PDF_PATH.name}")
    if len(pages) != 10:
        print("WARNING: The assignment asks for a 10-page PDF. Your PDF has", len(pages), "pages.")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    run_chunking_experiment(pages)
    run_retrieval_experiment(PDF_PATH, pages, model)
    run_metadata_and_upsert(PDF_PATH, model)
    print("\nCompleted. Check the outputs folder.")


if __name__ == "__main__":
    main()
