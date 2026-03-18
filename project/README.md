\# 📚 DocMind — RAG Knowledge Base Chatbot



An AI-powered chatbot that lets you upload any PDF and ask questions in natural language. Built using \*\*RAG (Retrieval Augmented Generation)\*\* with \*\*\[Endee](https://github.com/endee-io/endee)\*\* as the vector database.



\## 🎯 Project Overview

Upload PDFs → chunks get embedded → stored in Endee → ask questions → Endee retrieves relevant chunks → LLM generates grounded answers.



\## 🏗️ System Design

```

PDF → PyPDF2 → Chunker (500 chars) → all-MiniLM-L6-v2 → Endee (vector DB)

Question → all-MiniLM-L6-v2 → Endee.query(top\_k=5) → Llama 3.1 → Answer

```



\## 🗄️ How Endee is Used

\- `create\_index(name, dimension=384, space\_type="cosine", precision=INT8)`

\- `index.upsert(\[{id, vector, meta}])` — stores chunk embeddings

\- `index.query(vector, top\_k=5)` — finds most similar chunks



\## 🛠️ Tech Stack

| Component | Technology |

|---|---|

| Vector DB | Endee (Docker) |

| Embeddings | all-MiniLM-L6-v2 (384 dims) |

| LLM | Llama 3.1 8B via Groq |

| UI | Streamlit |



\## 🚀 Setup

1\. Install Docker and run: `docker run -d -p 8080:8080 endeeio/endee-server:latest`

2\. `pip install -r requirements.txt`

3\. Copy `.env.example` to `.env` and add your `GROQ\_API\_KEY`

4\. `streamlit run app.py`

5\. Upload a PDF and start asking questions!



\## 📁 Project Structure

```

project/

├── app.py           # Streamlit UI

├── ingest.py        # PDF → chunks → embeddings → Endee

├── query.py         # Question → Endee → LLM → answer

├── requirements.txt

├── .env.example

└── README.md

```



\## 🔮 Future Improvements

\- Re-ranking with cross-encoder

\- Hybrid search (BM25 + dense vectors)

\- RAGAS evaluation metrics

\- Streaming responses

