"""
query.py — The "retrieval + generation" pipeline
=================================================
What this file does (in plain English):
  1. Takes the user's question as input
  2. Converts that question into an embedding (same model as ingest.py)
  3. Asks Endee: "which stored chunks are most similar to this question?"
  4. Takes the top 5 most relevant chunks
  5. Builds a prompt: "Here's context from the document. Answer this question."
  6. Sends it to Google Gemini (free!) and returns the answer

This runs on EVERY question the user asks.
"""

import os
from sentence_transformers import SentenceTransformer
from endee import Endee
# Google Gemini — pip install google-generativeai


# ──────────────────────────────────────────────
# Singleton pattern — load model only once
# ──────────────────────────────────────────────
# Loading the embedding model takes ~2 seconds.
# We don't want to reload it on every question.
# This pattern keeps one instance alive across all calls.

_embedding_model = None

def get_embedding_model():
    """Load the embedding model once and reuse it."""
    global _embedding_model
    if _embedding_model is None:
        print("🧠 Loading embedding model...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


_endee_index = None

def get_endee_index(index_name: str = "knowledge_base"):
    """Connect to Endee once and reuse the connection."""
    global _endee_index
    if _endee_index is None:
        client = Endee()
        client.set_base_url("http://localhost:8080/api/v1")
        _endee_index = client.get_index(name=index_name)
    return _endee_index


# ──────────────────────────────────────────────
# STEP 1 — Embed the user's question
# ──────────────────────────────────────────────

def embed_question(question: str) -> list[float]:
    """
    Convert the user's question into a 384-dimensional vector.

    CRITICAL: We use the SAME model as in ingest.py.
    If you embed the documents with model A and the question with model B,
    the vectors live in different spaces and comparison is meaningless.
    Same model = same vector space = valid comparison.
    """
    model = get_embedding_model()
    embedding = model.encode(question)
    return embedding.tolist()


# ──────────────────────────────────────────────
# STEP 2 — Search Endee for similar chunks
# ──────────────────────────────────────────────

def retrieve_relevant_chunks(
    question_embedding: list[float],
    top_k: int = 5,
    similarity_threshold: float = 0.10
) -> list[dict]:
    """
    Query Endee with the question vector and get back the most similar chunks.

    Parameters:
      top_k              — how many chunks to retrieve (5 is a good default)
      similarity_threshold — ignore chunks below this score (0.3 = 30% similar)
                             too low → irrelevant noise, too high → misses valid results

    Returns a list of dicts like:
      [
        {"text": "...", "source": "my_doc", "score": 0.87},
        {"text": "...", "source": "my_doc", "score": 0.74},
        ...
      ]
    """
    index = get_endee_index()

    # This is the core Endee query — find top_k nearest vectors to our question
    results = index.query(
        vector=question_embedding,
        top_k=top_k
    )

    # Filter out low-similarity results and extract the text from metadata
    chunks = []
    for result in results:
        score = result["similarity"]    # cosine similarity score (0 to 1)

        if score < similarity_threshold:
            continue                 # skip this chunk — not relevant enough

        chunks.append({
            "text": result["meta"].get("text", ""),
            "source": result["meta"].get("source", "unknown"),
            "chunk_index": result["meta"].get("chunk_index", 0),
            "score": round(score, 3)                        # round for display
        })

    return chunks


# ──────────────────────────────────────────────
# STEP 3 — Build a prompt with retrieved context
# ──────────────────────────────────────────────

def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the final prompt we'll send to the LLM.

    RAG prompt structure:
      - Tell the LLM it's a helpful assistant
      - Give it the retrieved context (the relevant chunks)
      - Ask the actual question
      - Tell it to only use the provided context (prevents hallucination)

    This is called "prompt stuffing" — we're putting the document
    knowledge into the prompt rather than fine-tuning the model.
    """
    if not chunks:
        # No relevant chunks found — tell the LLM it doesn't have info
        return f"""You are a helpful assistant. 
        
The user asked: "{question}"

Unfortunately, I could not find any relevant information in the uploaded documents to answer this question. 
Please let the user know that this topic is not covered in their documents, and suggest they upload a relevant document."""

    # Build the context section from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i} from '{chunk['source']}' — relevance score: {chunk['score']}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful AI assistant. Answer the user's question directly and confidently based on the provided document excerpts.

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Give a direct, confident answer in 2-3 sentences
- Summarize what the document is about based on all excerpts together
- Don't say "I cannot determine" — synthesize the excerpts and give your best answer
- Mention the key topics you see across the excerpts
- Keep it concise and clear"""

    return prompt


# ──────────────────────────────────────────────
# STEP 4 — Call the LLM to generate an answer
# ──────────────────────────────────────────────

def generate_answer(prompt: str) -> str:
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ GROQ_API_KEY not found.")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ──────────────────────────────────────────────
# MAIN FUNCTION — full RAG pipeline in one call
# ──────────────────────────────────────────────

def ask(question: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline:
      question → embed → retrieve from Endee → build prompt → LLM → answer

    Returns a dict with:
      - answer   : the LLM's response
      - chunks   : the retrieved source chunks (for transparency)
      - question : the original question
    """
    print(f"\n❓ Question: {question}")

    # Step 1: Embed the question
    print("🔢 Embedding question...")
    question_embedding = embed_question(question)

    # Step 2: Find similar chunks in Endee
    print("🔍 Searching Endee for relevant chunks...")
    chunks = retrieve_relevant_chunks(question_embedding, top_k=top_k)
    print(f"   Found {len(chunks)} relevant chunks")

    if chunks:
        for i, c in enumerate(chunks, 1):
            print(f"   Chunk {i}: score={c['score']} | {c['text'][:80]}...")

    # Step 3: Build the prompt
    prompt = build_prompt(question, chunks)

    # Step 4: Generate the answer
    print("🤖 Generating answer with Gemini...")
    answer = generate_answer(prompt)

    print(f"✅ Answer generated ({len(answer)} chars)")

    return {
        "question": question,
        "answer": answer,
        "chunks": chunks,          # we return these so the UI can show them
        "chunks_found": len(chunks)
    }


# Test the pipeline from the command line
if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    result = ask(question)
    print(f"\n📝 ANSWER:\n{result['answer']}")