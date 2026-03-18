"""
app.py — DocMind RAG Chatbot
Features: beautiful pipeline viz, suggested questions, dark mode,
          chat export, search history, confidence meter, multi-PDF
"""

import os
import json
import tempfile
import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ingest import ingest_pdf
from query import ask

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="DocMind — RAG Knowledge Base", page_icon="🧠", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "ingested_files": [],
    "total_chunks": 0,
    "last_retrieval": None,
    "total_queries": 0,
    "search_history": [],
    "suggested_questions": [],
    "dark_mode": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Theme vars ────────────────────────────────────────────────────────────────
dm        = st.session_state.dark_mode
bg        = "#1a1a2e" if dm else "#ffffff"
bg2       = "#16213e" if dm else "#f8f9fc"
card_bg   = "#0f3460" if dm else "#ffffff"
text      = "#e0e0e0" if dm else "#1a1a2e"
text2     = "#a0a0b0" if dm else "#666666"
border    = "#334455" if dm else "#e8e8e8"
accent    = "#667eea"

st.markdown(f"""
<style>
    body, .stApp {{ background-color: {bg}; color: {text}; }}
    .main-header {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px 24px; border-radius: 12px; margin-bottom: 20px;
    }}
    .main-header h1 {{ color: white; margin: 0; font-size: 26px; }}
    .main-header p  {{ color: rgba(255,255,255,0.85); margin: 4px 0 0; font-size: 13px; }}
    .metric-card {{
        background: {card_bg}; border: 1px solid {border};
        border-radius: 10px; padding: 14px 18px; text-align: center;
    }}
    .metric-val   {{ font-size: 26px; font-weight: 700; color: {accent}; }}
    .metric-label {{ font-size: 11px; color: {text2}; text-transform: uppercase; letter-spacing:.05em; margin-top:2px; }}
    .chunk-card {{
        background: {bg2}; border-left: 4px solid {accent};
        border-radius: 0 8px 8px 0; padding: 12px 16px;
        margin: 8px 0; font-size: 13px; color: {text}; line-height: 1.6;
    }}
    .chunk-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
    .chunk-source {{ font-size:11px; color:{text2}; font-weight:600; text-transform:uppercase; }}
    .score-pill   {{ font-size:11px; font-weight:700; padding:3px 10px; border-radius:99px; color:white; }}
    .score-high   {{ background:#22c55e; }}
    .score-medium {{ background:#f59e0b; }}
    .score-low    {{ background:#94a3b8; }}
    .file-chip {{
        display:inline-block; background:#eef2ff; color:{accent};
        border-radius:99px; padding:4px 12px; font-size:12px; font-weight:600; margin:3px;
    }}
    .history-item {{
        background:{card_bg}; border:1px solid {border};
        border-radius:8px; padding:10px 14px; margin:6px 0; font-size:13px;
    }}
    .history-q    {{ font-weight:600; color:{text}; margin-bottom:4px; }}
    .history-meta {{ font-size:11px; color:{text2}; }}
    /* pipeline card */
    .pip-card {{
        border-radius: 12px; padding: 10px 8px; text-align: center;
        border: 2px solid; transition: all 0.2s;
    }}
    .pip-dot {{
        width:30px; height:30px; border-radius:50%;
        font-size:13px; font-weight:700; color:white;
        display:flex; align-items:center; justify-content:center; margin:0 auto 6px;
    }}
    .pip-label {{ font-size:12px; font-weight:600; margin-bottom:2px; }}
    .pip-sub   {{ font-size:10px; }}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_confidence(top_score):
    if top_score >= 0.5: return "High",   "#22c55e", 85
    if top_score >= 0.3: return "Medium", "#f59e0b", 55
    return "Low", "#94a3b8", 25

def generate_suggestions(filename):
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: return []
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content":
                f"Document: '{filename}'. Give exactly 4 short questions a user might ask. "
                f"Return ONLY a JSON array of 4 strings, nothing else."}],
            max_tokens=200
        )
        t = resp.choices[0].message.content.strip()
        s, e = t.find("["), t.rfind("]") + 1
        if s != -1 and e > s:
            return json.loads(t[s:e])
    except:
        pass
    return ["What is this document about?", "What are the main findings?",
            "What methods were used?", "What are the key conclusions?"]

def export_chat_text():
    lines = [f"DocMind Export — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
             f"Docs: {', '.join(st.session_state.ingested_files)}\n", "="*60+"\n"]
    for m in st.session_state.messages:
        role = "You" if m["role"] == "user" else "DocMind"
        lines.append(f"\n{role}:\n{m['content']}\n")
        if m.get("chunks"):
            lines.append(f"  [{len(m['chunks'])} chunks, top: {max(c['score'] for c in m['chunks']):.3f}]\n")
    return "\n".join(lines)

def render_pipeline(slot, stage="idle", chunks_found=0):
    """
    stage: 'running' | 'done' | 'idle'
    Uses pure Streamlit columns — no raw HTML injection.
    """
    if not st.session_state.get("show_pipeline_toggle", True):
        return

    steps = [
        ("🔢", "Embed",    "all-MiniLM-L6-v2"),
        ("🗄️",  "Search",   "Endee cosine ANN"),
        ("📎", "Retrieve", f"{chunks_found} chunks" if stage=="done" else "top-K chunks"),
        ("🤖", "Generate", "Llama 3.1 · Groq"),
    ]

    with slot.container():
        st.caption("⚡ RAG Pipeline")
        c1,a1,c2,a2,c3,a3,c4 = st.columns([4,1,4,1,4,1,4])
        cols   = [c1, c2, c3, c4]
        arrows = [a1, a2, a3]

        for i, (icon, label, sub) in enumerate(steps):
            with cols[i]:
                if stage == "done":
                    st.success(f"✅ **{label}**  \n{sub}")
                elif stage == "running" and i == 0:
                    st.info(f"⟳ **{label}**  \n{sub}")
                elif stage == "running":
                    st.markdown(f"""
                    <div style="background:{bg2};border:1px solid {border};border-radius:8px;
                         padding:10px;text-align:center;font-size:12px;color:{text2}">
                         <b>{icon} {label}</b><br><span style="font-size:10px">{sub}</span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:{bg2};border:1px solid {border};border-radius:8px;
                         padding:10px;text-align:center;font-size:12px;color:{text2}">
                         <b>{icon} {label}</b><br><span style="font-size:10px">{sub}</span>
                    </div>""", unsafe_allow_html=True)

        for a in arrows:
            with a:
                color = "#22c55e" if stage == "done" else border
                st.markdown(f"<div style='text-align:center;padding-top:14px;"
                            f"font-size:18px;color:{color}'>→</div>",
                            unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    col_logo, col_dm = st.columns([3,1])
    with col_logo:
        st.markdown("### 🧠 DocMind")
        st.caption("RAG · Endee Vector DB · LLM")
    with col_dm:
        dm_toggle = st.toggle("🌙", value=st.session_state.dark_mode, key="dm_toggle")
        if dm_toggle != st.session_state.dark_mode:
            st.session_state.dark_mode = dm_toggle
            st.rerun()

    st.markdown("---")
    st.markdown("**📤 Upload Documents**")
    uploaded_files = st.file_uploader("Upload one or more PDFs",
                                      type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.ingested_files:
                with st.spinner(f"Indexing '{uf.name}'..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uf.getvalue())
                        tmp_path = tmp.name
                    try:
                        n = ingest_pdf(tmp_path)
                        st.session_state.ingested_files.append(uf.name)
                        st.session_state.total_chunks += n
                        st.success(f"✅ {uf.name} — {n} chunks")
                        with st.spinner("✨ Generating suggestions..."):
                            st.session_state.suggested_questions = generate_suggestions(uf.name)
                    except Exception as e:
                        st.error(f"❌ {e}")
                    finally:
                        os.unlink(tmp_path)

    if st.session_state.ingested_files:
        st.markdown("---")
        st.markdown("**📁 Indexed Documents**")
        for fname in st.session_state.ingested_files:
            st.markdown(f'<span class="file-chip">📄 {fname}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Settings**")
    top_k        = st.slider("Chunks to retrieve (top-K)", 1, 10, 5)
    show_pipeline = st.toggle("Show RAG pipeline", value=True, key="show_pipeline_toggle")

    if st.session_state.messages:
        st.markdown("---")
        st.markdown("**💾 Export Chat**")
        st.download_button("📥 Download .txt", data=export_chat_text(),
            file_name=f"docmind_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain", use_container_width=True)

    st.markdown("---")
    st.markdown("**🔧 Tech Stack**")
    st.caption("🗄️ Endee vector DB  (localhost:8080)")
    st.caption("🧠 all-MiniLM-L6-v2  (384 dims)")
    st.caption("📏 Cosine similarity · HNSW")
    st.caption("🤖 Llama 3.1 8B via Groq")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages       = []
        st.session_state.last_retrieval = None
        st.session_state.total_queries  = 0
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🧠 DocMind — RAG Knowledge Base</h1>
  <p>Upload PDFs · Ask questions · Get answers grounded in your documents · Powered by Endee Vector Database</p>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "🕐 History", "🏗️ System Design"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    # Metrics row
    if st.session_state.ingested_files:
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><div class="metric-val">{len(st.session_state.ingested_files)}</div><div class="metric-label">Docs Indexed</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="metric-val">{st.session_state.total_chunks}</div><div class="metric-label">Chunks in Endee</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><div class="metric-val">{st.session_state.total_queries}</div><div class="metric-label">Queries Run</div></div>', unsafe_allow_html=True)
        with c4:
            ls = f"{st.session_state.last_retrieval['top_score']:.2f}" if st.session_state.last_retrieval else "—"
            st.markdown(f'<div class="metric-card"><div class="metric-val">{ls}</div><div class="metric-label">Last Top Score</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Welcome screen
    if not st.session_state.ingested_files:
        st.info("👈 Upload a PDF in the sidebar to get started!")
        st.markdown("""
**How it works:**
1. 📄 Upload PDF → chunked into ~500 char pieces → embedded to 384-dim vectors
2. 🗄️ Each chunk stored in **Endee** vector database (running in Docker)
3. ❓ Ask a question → embedded with the **same** model
4. 🔍 Endee finds the most similar chunks via cosine similarity (HNSW)
5. 🤖 Llama 3.1 reads those chunks → generates a grounded, cited answer
        """)

    # Suggested questions
    if st.session_state.suggested_questions and st.session_state.ingested_files:
        st.markdown("**✨ Suggested questions:**")
        cols = st.columns(2)
        for i, q in enumerate(st.session_state.suggested_questions):
            with cols[i % 2]:
                if st.button(f"💬 {q}", key=f"sq_{i}", use_container_width=True):
                    st.session_state["pending_question"] = q
                    st.rerun()
        st.markdown("---")

    # ── Chat history display ──────────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):

            # Show completed pipeline snapshot for past assistant messages
            if message["role"] == "assistant" and message.get("pipeline_done"):
                pip_slot = st.empty()
                render_pipeline(pip_slot, stage="done",
                                chunks_found=message.get("chunks_found", 0))

            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("chunks"):
                chunks    = message["chunks"]
                top_score = max(c['score'] for c in chunks)
                _, conf_color, conf_pct = get_confidence(top_score)

                # Confidence bar
                st.markdown(f"""
                <div style="margin:8px 0 12px">
                  <span style="font-size:12px;color:{text2}">Answer confidence: </span>
                  <span style="font-size:12px;font-weight:700;color:{conf_color}">{get_confidence(top_score)[0]}</span>
                  <span style="font-size:11px;color:{text2}"> · top score {top_score:.3f}</span>
                  <div style="background:{border};border-radius:4px;height:6px;width:100%;margin-top:4px">
                    <div style="background:{conf_color};width:{conf_pct}%;height:6px;border-radius:4px"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

                with st.expander(f"🔍 {len(chunks)} chunks retrieved from Endee"):
                    for i, chunk in enumerate(chunks, 1):
                        sc = "score-high" if chunk['score']>0.5 else "score-medium" if chunk['score']>0.3 else "score-low"
                        st.markdown(
                            f'<div class="chunk-card"><div class="chunk-header">'
                            f'<span class="chunk-source">📄 {chunk["source"]} · chunk #{chunk["chunk_index"]}</span>'
                            f'<span class="score-pill {sc}">sim: {chunk["score"]}</span></div>'
                            f'{chunk["text"]}</div>', unsafe_allow_html=True)

    # ── New question ──────────────────────────────────────────────────────
    pending    = st.session_state.pop("pending_question", None)
    user_input = st.chat_input("Ask anything about your documents...")
    question   = pending or user_input

    if question:
        if not st.session_state.ingested_files:
            st.warning("Please upload a PDF first!")
            st.stop()
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not found in .env file.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            pip_slot = st.empty()

            # Show pipeline — step 1 active
            if show_pipeline:
                render_pipeline(pip_slot, stage="running")

            with st.spinner("Searching Endee and generating answer..."):
                try:
                    result = ask(question, top_k=top_k)
                    answer = result["answer"]
                    chunks = result["chunks"]

                    # Update pipeline — all done
                    if show_pipeline:
                        render_pipeline(pip_slot, stage="done", chunks_found=len(chunks))

                    st.markdown(answer)

                    if chunks:
                        top_score = max(c['score'] for c in chunks)
                        _, conf_color, conf_pct = get_confidence(top_score)

                        # Confidence bar
                        st.markdown(f"""
                        <div style="margin:10px 0 12px">
                          <span style="font-size:12px;color:{text2}">Answer confidence: </span>
                          <span style="font-size:12px;font-weight:700;color:{conf_color}">{get_confidence(top_score)[0]}</span>
                          <span style="font-size:11px;color:{text2}"> · top score {top_score:.3f}</span>
                          <div style="background:{border};border-radius:4px;height:6px;width:100%;margin-top:4px">
                            <div style="background:{conf_color};width:{conf_pct}%;height:6px;border-radius:4px"></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                        st.session_state.last_retrieval = {
                            "top_score": top_score, "chunks": chunks, "question": question
                        }

                        with st.expander(f"🔍 {len(chunks)} chunks retrieved from Endee (top: {top_score:.3f})"):
                            for i, chunk in enumerate(chunks, 1):
                                sc = "score-high" if chunk['score']>0.5 else "score-medium" if chunk['score']>0.3 else "score-low"
                                st.markdown(
                                    f'<div class="chunk-card"><div class="chunk-header">'
                                    f'<span class="chunk-source">Rank #{i} · 📄 {chunk["source"]} · chunk #{chunk["chunk_index"]}</span>'
                                    f'<span class="score-pill {sc}">sim: {chunk["score"]}</span></div>'
                                    f'{chunk["text"]}</div>', unsafe_allow_html=True)

                        conf_label, conf_color, _ = get_confidence(top_score)
                        st.session_state.search_history.append({
                            "question":    question,
                            "answer":      answer[:200] + ("..." if len(answer) > 200 else ""),
                            "top_score":   top_score,
                            "confidence":  conf_label,
                            "conf_color":  conf_color,
                            "chunks_found": len(chunks),
                            "time":        datetime.datetime.now().strftime("%H:%M:%S"),
                            "docs":        ", ".join(set(c['source'] for c in chunks))
                        })

                    st.session_state.total_queries += 1
                    st.session_state.messages.append({
                        "role":         "assistant",
                        "content":      answer,
                        "chunks":       chunks,
                        "chunks_found": len(chunks) if chunks else 0,
                        "pipeline_done": True
                    })

                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Last Retrieval — What Endee Found")
    if not st.session_state.last_retrieval:
        st.info("Ask a question in the Chat tab to see analytics here.")
    else:
        r = st.session_state.last_retrieval
        chunks, top_score = r["chunks"], r["top_score"]
        conf_label, conf_color, conf_pct = get_confidence(top_score)

        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(f'<div class="metric-card"><div class="metric-val">{len(chunks)}</div><div class="metric-label">Chunks Retrieved</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{conf_color}">{conf_label}</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><div class="metric-val">{top_score:.3f}</div><div class="metric-label">Top Similarity</div></div>', unsafe_allow_html=True)

        st.markdown(f"**Question:** {r['question']}")
        st.markdown("---")
        st.markdown("**Similarity scores per chunk:**")
        import pandas as pd
        df = pd.DataFrame([{"Chunk": f"#{c['chunk_index']} ({c['source']})",
                            "Similarity": round(c['score'], 4)} for c in chunks])
        st.bar_chart(df.set_index("Chunk"))
        st.markdown("---")
        for i, chunk in enumerate(chunks, 1):
            sc = "score-high" if chunk['score']>0.5 else "score-medium" if chunk['score']>0.3 else "score-low"
            st.markdown(
                f'<div class="chunk-card"><div class="chunk-header">'
                f'<span class="chunk-source">Rank #{i} · 📄 {chunk["source"]} · #{chunk["chunk_index"]}</span>'
                f'<span class="score-pill {sc}">sim: {chunk["score"]}</span></div>'
                f'{chunk["text"]}</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEARCH HISTORY
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🕐 Search History")
    if not st.session_state.search_history:
        st.info("Your search history will appear here as you ask questions.")
    else:
        history_text = "\n\n".join([
            f"[{h['time']}] Q: {h['question']}\nA: {h['answer']}\nConfidence: {h['confidence']} ({h['top_score']:.3f})"
            for h in reversed(st.session_state.search_history)
        ])
        st.download_button("📥 Export History", data=history_text,
                           file_name="search_history.txt", mime="text/plain")
        st.markdown(f"**{len(st.session_state.search_history)} searches**")
        st.markdown("---")
        for h in reversed(st.session_state.search_history):
            col_h, col_btn = st.columns([5, 1])
            with col_h:
                st.markdown(f"""
                <div class="history-item">
                  <div class="history-q">❓ {h['question']}</div>
                  <div class="history-meta">
                    🕐 {h['time']} &nbsp;·&nbsp;
                    <span style="color:{h['conf_color']};font-weight:600">{h['confidence']}</span> confidence
                    &nbsp;·&nbsp; {h['chunks_found']} chunks &nbsp;·&nbsp; 📄 {h['docs']}
                  </div>
                </div>""", unsafe_allow_html=True)
            with col_btn:
                if st.button("↩ Re-run", key=f"rerun_{h['time']}"):
                    st.session_state["pending_question"] = h['question']
                    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — SYSTEM DESIGN
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🏗️ System Architecture")

    st.markdown("""
| Component | Technology | Role |
|---|---|---|
| Vector DB | **Endee** (Docker) | Stores + retrieves 384-dim vectors via HNSW |
| Embeddings | **all-MiniLM-L6-v2** | Converts text → 384 floats locally, free |
| LLM | **Llama 3.1 8B** via Groq | Generates grounded answers from retrieved context |
| PDF parsing | **PyPDF2** | Extracts raw text page by page |
| UI | **Streamlit** | Chat interface, analytics, history |
    """)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Indexing pipeline")
        st.code("""
PDF
 └─ PyPDF2 → raw text
     └─ Chunker (500 chars, 100 overlap)
         └─ all-MiniLM-L6-v2 → 384-dim vector
             └─ Endee.upsert([{id, vector, meta}])
                 index: knowledge_base
                 space: cosine, INT8, HNSW
        """, language="text")

    with col2:
        st.markdown("#### Query pipeline")
        st.code("""
User question
 └─ all-MiniLM-L6-v2 → 384-dim vector
     └─ Endee.query(vector, top_k=5)
         └─ top-K chunks (similarity ≥ 0.10)
             └─ RAG prompt (context + question)
                 └─ Llama 3.1 → answer + citations
        """, language="text")

    st.markdown("---")
    st.markdown("#### Key design decisions")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Why Endee?**
- Self-hosted Docker, no cloud costs
- HNSW = O(log n) ANN search
- INT8 precision = 4x memory saving
- Simple SDK: create → upsert → query

**Why cosine similarity?**
- Magnitude-independent text comparison
- Works equally for short and long chunks
- Industry standard for semantic search

**Why chunk size 500, overlap 100?**
- Fits embedding model's 512-token limit
- Overlap preserves cross-boundary context
- Small = focused and precise retrieval
        """)
    with c2:
        st.markdown("""
**Why all-MiniLM-L6-v2?**
- Free, 100% local, no API key needed
- 384 dims = fast + accurate enough
- Most popular model for RAG projects

**Why RAG over fine-tuning?**
- Zero training cost, no GPU needed
- New documents available instantly
- Answers are verifiable via source chunks
- Works on any document, any topic

**Why confidence meter?**
- Makes retrieval quality transparent
- Tells user when to rephrase
- Shows understanding of the system
        """)