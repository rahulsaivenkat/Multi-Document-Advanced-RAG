"""PaperMind AI — Modern Streamlit Frontend Interface.

Communicates with the FastAPI backend to provide multi-format document ingestion,
interactive conversational RAG, and verifiable citation exploration.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PaperMind AI — Multi-Format RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

# Supported file extensions for Streamlit file uploader
SUPPORTED_EXTENSIONS = ["pdf", "txt", "md", "docx", "csv", "html", "htm", "json", "py"]

# -----------------------------------------------------------------------------
# Custom Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global App Aesthetics */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .citation-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        border-radius: 4px;
        font-size: 0.92rem;
    }
    .citation-tag {
        font-weight: 700;
        color: #1d4ed8;
        background-color: #dbeafe;
        padding: 2px 6px;
        border-radius: 4px;
        margin-right: 6px;
    }
    .location-badge {
        font-weight: 600;
        color: #047857;
        background-color: #d1fae5;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 4px;
    }
    .diag-box {
        background-color: #f1f5f9;
        padding: 0.6rem 0.8rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL


# -----------------------------------------------------------------------------
# API Client Helpers
# -----------------------------------------------------------------------------
def get_backend_health(base_url: str) -> Optional[Dict[str, Any]]:
    """Fetch health and component info from FastAPI backend."""
    try:
        with httpx.Client(timeout=4.0) as client:
            res = client.get(f"{base_url}/health")
            if res.status_code == 200:
                return res.json()
    except Exception:
        pass
    return None


def fetch_documents(base_url: str) -> List[Dict[str, Any]]:
    """Fetch the indexed document inventory from FastAPI."""
    try:
        with httpx.Client(timeout=6.0) as client:
            res = client.get(f"{base_url}/documents")
            if res.status_code == 200:
                return res.json().get("documents", [])
    except Exception:
        pass
    return []


def upload_files(base_url: str, uploaded_files) -> Optional[Dict[str, Any]]:
    """Upload multiple files to FastAPI /documents/upload."""
    files_payload = []
    for f in uploaded_files:
        files_payload.append(("files", (f.name, f.getvalue(), f.type or "application/octet-stream")))

    try:
        with httpx.Client(timeout=60.0) as client:
            res = client.post(f"{base_url}/documents/upload", files=files_payload)
            if res.status_code == 200:
                return res.json()
            else:
                st.error(f"Upload failed (HTTP {res.status_code}): {res.text}")
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
    return None


def delete_document(base_url: str, doc_id: str) -> bool:
    """Request document deletion from FastAPI backend."""
    try:
        with httpx.Client(timeout=10.0) as client:
            res = client.delete(f"{base_url}/documents/{doc_id}")
            return res.status_code == 200
    except Exception:
        return False


def query_rag(
    base_url: str,
    question: str,
    conversation_history: List[Dict[str, str]],
    document_ids: Optional[List[str]] = None,
    rerank_top_k: int = 5
) -> Optional[Dict[str, Any]]:
    """Send user question to /chat endpoint on FastAPI backend."""
    # Convert session messages to backend schema
    chat_history = []
    for m in conversation_history:
        if m["role"] in ["user", "assistant"]:
            chat_history.append({"role": m["role"], "content": m["content"]})

    payload = {
        "question": question,
        "conversation_history": chat_history[-6:] if chat_history else None,
        "document_ids": document_ids if document_ids else None,
        "rerank_top_k": rerank_top_k
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            res = client.post(f"{base_url}/chat", json=payload)
            if res.status_code == 200:
                return res.json()
            else:
                st.error(f"Chat request failed (HTTP {res.status_code}): {res.text}")
    except Exception as e:
        st.error(f"Connection error to backend: {e}")
    return None


# -----------------------------------------------------------------------------
# Sidebar: System Controls & Document Manager
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Backend Status")
    
    # Backend URL configuration
    backend_url = st.text_input("FastAPI Backend URL", value=st.session_state.backend_url)
    st.session_state.backend_url = backend_url.rstrip("/")

    health_info = get_backend_health(st.session_state.backend_url)

    if health_info:
        st.success("🟢 FastAPI Backend Connected", icon="✅")
        groq_badge = "🟢 Configured" if health_info.get("groq_configured") else "🟡 Using Local Fallback"
        
        with st.expander("Component Details", expanded=False):
            st.markdown(f"**App Version:** `{health_info.get('version')}`")
            st.markdown(f"**Embeddings:** `{health_info.get('embedding_model')}`")
            st.markdown(f"**Reranker:** `{health_info.get('cross_encoder_model')}`")
            st.markdown(f"**Groq LLM:** {groq_badge}")
            st.markdown(f"**Total Docs:** `{health_info.get('indexed_documents_count')}`")
            st.markdown(f"**Total Chunks:** `{health_info.get('indexed_chunks_count')}`")
    else:
        st.error("🔴 Backend Offline or Unreachable", icon="⚠️")
        st.info("Start the FastAPI server in PowerShell:\n```powershell\npython -m uvicorn app.main:app --reload\n```")

    st.markdown("---")

    # Document Upload Section
    st.markdown("### 📄 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload files (.pdf, .txt, .md, .docx, .csv, .html, .json, .py)",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Ingest Documents", use_container_width=True, type="primary"):
            with st.spinner("Parsing, chunking, and dual-indexing into ChromaDB + BM25..."):
                result = upload_files(st.session_state.backend_url, uploaded_files)
                if result:
                    for item in result.get("uploaded", []):
                        fname = item.get("original_filename")
                        status = item.get("status")
                        chunks = item.get("indexed_chunks")
                        if status == "success":
                            st.success(f"**{fname}**: Successfully indexed {chunks} chunks.")
                        elif status == "warning":
                            st.warning(f"**{fname}**: {item.get('warnings', [''])[0]}")
                        else:
                            st.error(f"**{fname}**: {item.get('warnings', ['Failed'])[0]}")
                    st.rerun()

    st.markdown("---")

    # Document Inventory
    st.markdown("### 📚 Indexed Documents")
    documents = fetch_documents(st.session_state.backend_url)
    selected_doc_ids = []

    if documents:
        st.caption(f"{len(documents)} document(s) active in persistent storage")
        
        # Document Filter Multiselect
        doc_options = {d["document_id"]: f"{d['source_filename']} ({d['file_type']})" for d in documents}
        selected_doc_ids = st.multiselect(
            "Filter search scope (optional):",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
            help="Select specific documents to restrict retrieval, or leave empty to query across all."
        )

        # Inventory Table / Deletion Expander
        with st.expander("Manage Stored Files"):
            for d in documents:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{d['source_filename']}**  \n`{d['chunk_count']} chunks` | `{d['file_type']}`")
                with col2:
                    if st.button("🗑️", key=f"del_{d['document_id']}", help="Delete from vector & keyword stores"):
                        if delete_document(st.session_state.backend_url, d["document_id"]):
                            st.toast(f"Deleted {d['source_filename']}")
                            st.rerun()
                        else:
                            st.error("Delete failed")
    else:
        st.info("No documents indexed yet. Upload documents above to begin!")

    st.markdown("---")

    # Retrieval Controls
    st.markdown("### 🔍 Search Settings")
    rerank_k = st.slider("Top Evidence Chunks (Rerank K)", min_value=1, max_value=10, value=5)

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# Main Application Content: Chat & Evidence Attribution
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🧠 PaperMind AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Format RAG with Hybrid BM25 + Semantic Search, CrossEncoder Reranking & Verifiable Citations</div>',
    unsafe_allow_html=True
)

# Display starter tips if conversation is empty
if not st.session_state.messages:
    st.markdown("""
    #### 💡 Getting Started
    1. **Upload Documents** in the sidebar (supported formats: PDF, Text, Markdown, Word DOCX, CSV, HTML, JSON, Python).
    2. **Ask Questions** across your documents in natural language.
    3. **Verify Citations**: Every answer is grounded in retrieved chunks with exact page numbers, line ranges, row numbers, or JSON paths.
    """)

# Render historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render citations if attached to assistant message
        if msg.get("citations"):
            with st.expander(f"📑 Verified Sources ({len(msg['citations'])})", expanded=False):
                for c in msg["citations"]:
                    st.markdown(
                        f"""<div class="citation-card">
                            <span class="citation-tag">[{c['citation_id']}]</span>
                            <strong>{c['source_filename']}</strong> 
                            <span class="location-badge">{c['readable_location']}</span>
                            <br><small style="color: #4b5563;">{c['excerpt']}</small>
                        </div>""",
                        unsafe_allow_html=True
                    )

        # Render diagnostics if present
        if msg.get("diagnostics"):
            diag = msg["diagnostics"]
            with st.expander("🛠️ Retrieval Diagnostics", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Dense Semantic Matches", diag.get("semantic_results_count", 0))
                    st.metric("BM25 Keyword Matches", diag.get("bm25_results_count", 0))
                with col_b:
                    st.metric("RRF Fused Candidates", diag.get("rrf_candidates_count", 0))
                    st.metric("Final Reranked Chunks", diag.get("reranked_results_count", 0))
                
                if diag.get("query_transformed"):
                    st.info(f"**Query Rewritten for Search:** `{diag.get('transformed_query')}`")


# -----------------------------------------------------------------------------
# Chat Input & RAG Execution
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question across your documents..."):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query with backend
    with st.chat_message("assistant"):
        with st.spinner("Searching with BM25 + dense vectors and reranking candidates..."):
            response = query_rag(
                base_url=st.session_state.backend_url,
                question=prompt,
                conversation_history=st.session_state.messages[:-1],
                document_ids=selected_doc_ids,
                rerank_top_k=rerank_k
            )

        if response:
            answer = response.get("answer", "No response generated.")
            citations = response.get("citations", [])
            insufficient = response.get("insufficient_evidence", False)
            diagnostics = response.get("diagnostics")

            # Display answer
            if insufficient:
                st.warning("⚠️ **Notice:** The uploaded documents do not contain sufficient evidence to answer this question.")
            st.markdown(answer)

            # Display citations expander
            if citations:
                with st.expander(f"📑 Verified Sources ({len(citations)})", expanded=True):
                    for c in citations:
                        st.markdown(
                            f"""<div class="citation-card">
                                <span class="citation-tag">[{c['citation_id']}]</span>
                                <strong>{c['source_filename']}</strong> 
                                <span class="location-badge">{c['readable_location']}</span>
                                <br><small style="color: #4b5563;">{c['excerpt']}</small>
                            </div>""",
                            unsafe_allow_html=True
                        )

            # Display diagnostics expander
            if diagnostics:
                with st.expander("🛠️ Retrieval Diagnostics", expanded=False):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Dense Semantic Matches", diagnostics.get("semantic_results_count", 0))
                        st.metric("BM25 Keyword Matches", diagnostics.get("bm25_results_count", 0))
                    with col_b:
                        st.metric("RRF Fused Candidates", diagnostics.get("rrf_candidates_count", 0))
                        st.metric("Final Reranked Chunks", diagnostics.get("reranked_results_count", 0))

                    if diagnostics.get("query_transformed"):
                        st.info(f"**Query Rewritten for Search:** `{diagnostics.get('transformed_query')}`")

            # Save assistant message to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "diagnostics": diagnostics,
                "insufficient_evidence": insufficient
            })
