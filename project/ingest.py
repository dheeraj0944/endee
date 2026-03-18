"""
ingest.py — The "indexing" pipeline
====================================
What this file does (in plain English):
  1. Reads a PDF file page by page
  2. Splits the text into small overlapping chunks (like cutting a book into flashcards)
  3. Converts each chunk into an embedding (a list of 384 numbers) using a free AI model
  4. Stores every chunk + its embedding into Endee (our vector database)

Run this ONCE per document you want the chatbot to know about.
"""

import os
import re
from PyPDF2 import PdfReader                          # reads PDF files
from sentence_transformers import SentenceTransformer # converts text → embeddings (numbers)
from endee import Endee, Precision


# ──────────────────────────────────────────────
# STEP 1 — Connect to Endee (running on Docker)
# ──────────────────────────────────────────────

def get_endee_index(index_name: str = "knowledge_base"):
    client = Endee()
    client.set_base_url("http://localhost:8080/api/v1")
    try:
        client.create_index(
            name=index_name,
            dimension=384,
            space_type="cosine",
            precision=Precision.INT8
        )
        print(f"📦 Created new Endee index: '{index_name}'")
    except Exception:
        print(f"✅ Using existing Endee index: '{index_name}'")
    return client.get_index(name=index_name)


# ──────────────────────────────────────────────
# STEP 2 — Read and extract text from a PDF
# ──────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Open a PDF and extract all its text as one big string.
    Each page's text is joined with a newline.
    """
    print(f"📄 Reading PDF: {pdf_path}")
    reader = PdfReader(pdf_path)

    full_text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:                          # some pages might be images with no text
            full_text += page_text + "\n"
            print(f"   Page {page_num + 1} extracted ({len(page_text)} chars)")

    print(f"✅ Total text extracted: {len(full_text)} characters")
    return full_text


# ──────────────────────────────────────────────
# STEP 3 — Split text into chunks
# ──────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split a long text into smaller overlapping pieces.

    WHY CHUNK?
      Embedding models have a token limit (~512 tokens).
      Also, a full document averages out all topics — poor retrieval.
      Small focused chunks = precise retrieval.

    WHY OVERLAP?
      If we cut strictly at 500 chars, a sentence might get split in half.
      Overlap (100 chars) means each chunk shares some content with the next,
      so we never lose context at the boundaries.

    Example with chunk_size=10, overlap=3:
      Text:  "Hello world this is a test"
      Chunk1: "Hello worl"
      Chunk2: "rld this i"   ← starts 3 chars back from where chunk1 ended
      Chunk3: "s is a tes"
    """
    # Clean up excessive whitespace first
    text = re.sub(r'\s+', ' ', text).strip()

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we're not at the end, try to cut at a sentence boundary (. ! ?)
        if end < len(text):
            # Look for the last sentence-ending punctuation before 'end'
            last_period = max(
                text.rfind('. ', start, end),
                text.rfind('! ', start, end),
                text.rfind('? ', start, end)
            )
            if last_period > start + (chunk_size // 2):  # only use it if it's past the halfway point
                end = last_period + 1                     # include the period in this chunk

        chunk = text[start:end].strip()
        if chunk:                                         # skip empty chunks
            chunks.append(chunk)

        start = end - overlap                             # move back by 'overlap' chars for next chunk
        if start >= len(text):
            break

    print(f"✅ Split into {len(chunks)} chunks (size≈{chunk_size} chars, overlap={overlap})")
    return chunks


# ──────────────────────────────────────────────
# STEP 4 — Convert chunks to embeddings
# ──────────────────────────────────────────────

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Use a free sentence-transformer model to turn each text chunk
    into a list of 384 numbers (an embedding vector).

    all-MiniLM-L6-v2 is:
      - Free and runs locally (no API key needed)
      - Fast — encodes hundreds of chunks in seconds
      - 384 dimensions — small but very accurate for similarity search
      - The most popular model for RAG projects
    """
    print("🧠 Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')    # downloads ~90MB on first run

    print(f"🔢 Embedding {len(chunks)} chunks...")
    embeddings = model.encode(
        chunks,
        show_progress_bar=True,   # shows a progress bar in the terminal
        batch_size=32             # process 32 chunks at a time (faster than one-by-one)
    )

    # model.encode() returns a numpy array — convert to plain Python lists for Endee
    embeddings_list = [emb.tolist() for emb in embeddings]
    print(f"✅ Created {len(embeddings_list)} embeddings, each with {len(embeddings_list[0])} dimensions")
    return embeddings_list


# ──────────────────────────────────────────────
# STEP 5 — Store everything in Endee
# ──────────────────────────────────────────────

def upsert_to_endee(index, chunks: list[str], embeddings: list[list[float]], source_name: str):
    """
    Store each chunk + its embedding into Endee.

    Each item we store has:
      - id     : unique identifier (e.g. "myfile_chunk_0")
      - vector : the 384-number embedding
      - meta   : extra info we want back when we retrieve it
                 (the actual text, source filename, chunk number)

    WHY STORE THE TEXT IN META?
      Endee only stores vectors for fast search.
      But when we retrieve the top-5 similar vectors, we need the
      actual text to show the user and pass to the LLM.
      Storing it in metadata means we get it back automatically.
    """
    print(f"📤 Uploading {len(chunks)} chunks to Endee...")

    # Build a list of vector items to upsert
  
    vector_items = []
    for i in range(len(chunks)):
        chunk = str(chunks[i])
        embedding = embeddings[i]
        vector_items.append({
            "id": f"{source_name}_chunk_{i}",
            "vector": embedding,
            "meta": {"text": chunk, "source": source_name, "chunk_index": i}
        })

    # Endee supports batch upserts — send all chunks in one call (much faster)
    BATCH_SIZE = 100
    for i in range(0, len(vector_items), BATCH_SIZE):
        batch = vector_items[i:i + BATCH_SIZE]
        index.upsert(batch)
        print(f"   Uploaded batch {i//BATCH_SIZE + 1} ({len(batch)} chunks)")

    print(f"✅ All {len(chunks)} chunks stored in Endee!")


# ──────────────────────────────────────────────
# MAIN FUNCTION — tie everything together
# ──────────────────────────────────────────────

def ingest_pdf(pdf_path: str, index_name: str = "knowledge_base"):
    """
    Full pipeline: PDF → chunks → embeddings → Endee
    Call this function with any PDF path to index it.
    """
    # Get a clean filename to use as source identifier (e.g. "my_document.pdf" → "my_document")
    source_name = os.path.splitext(os.path.basename(pdf_path))[0]
    source_name = re.sub(r'[^a-zA-Z0-9_-]', '_', source_name)  # make it safe for IDs

    print(f"\n{'='*50}")
    print(f"🚀 Starting ingestion for: {pdf_path}")
    print(f"{'='*50}\n")

    # Connect to Endee
    index = get_endee_index(index_name)

    # Step 1: Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("❌ No text found in PDF. It might be a scanned image PDF.")
        return

    # Step 2: Chunk the text
    chunks = split_into_chunks(text, chunk_size=800, overlap=150)

    # Step 3: Embed the chunks
    embeddings = embed_chunks(chunks)

    # Step 4: Store in Endee
    upsert_to_endee(index, chunks, embeddings, source_name)

    print(f"\n🎉 Done! '{source_name}' is now searchable in your chatbot.\n")
    return len(chunks)


# Run this file directly to test ingestion
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py path/to/your/file.pdf")
    else:
        ingest_pdf(sys.argv[1])