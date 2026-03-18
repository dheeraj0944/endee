# DocMind — RAG Knowledge Base Chatbot

An AI-powered chatbot that lets you upload any PDF and ask questions in natural language. Built using **RAG (Retrieval Augmented Generation)** with **[Endee](https://github.com/endee-io/endee)** as the vector database.

---

## Project Overview

Most LLMs don't know anything about your documents. DocMind solves this using RAG:

1. Upload a PDF — it gets split into chunks, embedded into vectors, and stored in Endee
2. Ask a question — it gets embedded using the same model
3. Endee finds the most similar chunks using cosine similarity search
4. An LLM reads those chunks and generates a grounded, cited answer

The result: accurate answers from your own content, with full source transparency.

---

## System Design

**Indexing Pipeline** (runs once per document)

```
PDF File
  └── PyPDF2          → extract raw text
      └── Chunker     → split into 500-char overlapping chunks
          └── Embedder (all-MiniLM-L6-v2) → 384-dim vectors
              └── Endee (vector DB) → store vectors + metadata
```

**Query Pipeline** (runs on every question)

```
User Question
  └── Embedder (all-MiniLM-L6-v2) → 384-dim vector  [same model!]
      └── Endee.query(vector, top_k=5) → top similar chunks
          └── RAG Prompt (context + question)
              └── Llama 3.1 8B via Groq → grounded answer + citations
```

---

## How Endee is Used

Endee is the core vector database that powers semantic search in this project.

**Creating the index**
```python
from endee import Endee, Precision

client = Endee()
client.set_base_url("http://localhost:8080/api/v1")
client.create_index(
    name="knowledge_base",
    dimension=384,            # matches all-MiniLM-L6-v2 output
    space_type="cosine",      # cosine similarity for text comparison
    precision=Precision.INT8  # 4x memory compression
)
```

**Storing chunk embeddings**
```python
index = client.get_index("knowledge_base")
index.upsert([
    {
        "id": "doc_chunk_0",
        "vector": [0.023, -0.412, ...],  # 384 floats
        "meta": {
            "text": "the actual chunk text",
            "source": "my_document",
            "chunk_index": 0
        }
    }
])
```

**Retrieving similar chunks**
```python
results = index.query(vector=question_embedding, top_k=5)
for result in results:
    print(result["similarity"])       # cosine similarity score
    print(result["meta"]["text"])     # the chunk text
```

---

## Tech Stack

| Component   | Technology              | Why                                     |
|-------------|-------------------------|-----------------------------------------|
| Vector DB   | Endee (Docker)          | Self-hosted, HNSW search, simple SDK    |
| Embeddings  | all-MiniLM-L6-v2        | Free, local, 384-dim, no API key needed |
| LLM         | Llama 3.1 8B via Groq   | Free API, fast inference                |
| PDF Parsing | PyPDF2                  | Lightweight text extraction             |
| UI          | Streamlit               | Clean chat interface                    |
| Language    | Python 3.10+            |                                         |

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Docker Desktop installed and running
- Groq API key — free at [console.groq.com](https://console.groq.com)

### Step 1 — Clone this repo
```bash
git clone https://github.com/YOUR_USERNAME/endee.git
cd endee/project
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Set up your API key
```bash
cp .env.example .env
```
Open `.env` and add:
```
GROQ_API_KEY=your_key_here
```

### Step 5 — Start Endee with Docker
```bash
docker run -d -p 8080:8080 endeeio/endee-server:latest
```

### Step 6 — Run the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 7 — Use it
1. Upload any PDF using the sidebar
2. Wait for the indexed confirmation
3. Ask questions in the chat

---

## Project Structure

```
project/
├── app.py            Streamlit UI — chat, analytics, history, system design
├── ingest.py         Indexing pipeline: PDF to chunks to embeddings to Endee
├── query.py          Query pipeline: question to Endee to LLM to answer
├── requirements.txt  Python dependencies
├── .env.example      Template for API keys
├── .gitignore        Excludes .env from git
└── README.md         This file
```

---

## Key Concepts Demonstrated

- **Embeddings** — text converted to 384-dim vectors where similar meaning equals close vectors
- **Chunking** — documents split into overlapping 500-char pieces for precise retrieval
- **Vector similarity search** — cosine similarity via Endee's HNSW index
- **RAG** — retrieved chunks injected into LLM prompt to ground answers in your documents
- **Confidence scoring** — similarity scores surfaced to the user for answer transparency
- **Multi-PDF support** — multiple documents indexed in the same Endee index

---

## Future Improvements

- Re-ranking with a cross-encoder model on top-K results
- Hybrid search combining dense vectors with BM25 sparse search
- RAGAS evaluation metrics for retrieval quality measurement
- Streaming LLM responses for better UX
- Multi-user support with namespaced Endee indexes
