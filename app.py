import tempfile
import os
import time
import pdfplumber
import docx2txt
import chardet
from typing import Tuple, List
import io
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model
    summarizer_tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-small")
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small")
    return embed_model, summarizer_model, summarizer_tokenizer


# =========================
# Model Loading (cached)
# =========================


# =========================
# Model Loading (cached)
# =========================
@st.cache_resource
def load_models():
    # Open-source, CPU-friendly models
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    sum_tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    sum_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return embed_model, sum_model, sum_tok


embed_model, sum_model, sum_tok = load_models()


# =========================
# Utilities
# =========================
def timed(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)


def read_txt(file_bytes: bytes) -> Tuple[str, str]:
    """Read text with encoding detection. Returns (text, encoding_used)."""
    guess = chardet.detect(file_bytes).get("encoding") or "utf-8"
    try:
        return file_bytes.decode(guess), guess
    except UnicodeDecodeError:
        # Safe fallback that never errors
        return file_bytes.decode("latin1"), "latin1"


def read_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX byte stream using a temporary file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        text = docx2txt.process(tmp_path) or ""
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return text


def read_pdf(file_bytes: bytes) -> str:
    """Extract text from a (text-based) PDF. Scanned PDFs need OCR (not included)."""
    parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t:
                parts.append(t)
    return "\n".join(parts).strip()


def load_resume_text(uploaded_file) -> Tuple[str, str]:
    """
    Returns (text, info) for a Streamlit UploadedFile.
    info includes encoding or parser used for display/debug.
    """
    raw = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        text, enc = read_txt(raw)
        return text, f"txt ({enc})"
    elif name.endswith(".docx"):
        return read_docx(raw), "docx"
    elif name.endswith(".pdf"):
        return read_pdf(raw), "pdf"
    else:
        # Unknown extension: try robust text fallback
        text, enc = read_txt(raw)
        return text, f"fallback-txt ({enc})"


# =========================
# Summarization
# =========================
def summarize_fit(job_desc: str, resume_text: str, max_new_tokens: int = 60) -> Tuple[str, str]:
    """
    Generate:
      - detailed_summary: 2–3 lines explaining fit
      - one_liner: a single concise sentence explaining fit
    Uses FLAN-T5 (open source).
    """
    # Detailed 2–3 lines
    detailed_prompt = (
        "You are a recruiter. In 2–3 lines, explain why this resume fits the job.\n"
        f"Job: {job_desc}\nResume: {resume_text}\nAnswer:"
    )
    detailed_inputs = sum_tok(
        detailed_prompt, return_tensors="pt", truncation=True, max_length=512)
    detailed_outputs = sum_model.generate(
        **detailed_inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True
    )
    detailed_summary = sum_tok.decode(
        detailed_outputs[0], skip_special_tokens=True)

    # One-liner
    one_liner_prompt = (
        "Give exactly one concise sentence on why this candidate is suitable for the job.\n"
        f"Job: {job_desc}\nResume: {resume_text}\nAnswer:"
    )
    one_inputs = sum_tok(one_liner_prompt, return_tensors="pt",
                         truncation=True, max_length=512)
    one_outputs = sum_model.generate(
        **one_inputs,
        max_new_tokens=30,
        num_beams=4,
        early_stopping=True
    )
    one_liner = sum_tok.decode(one_outputs[0], skip_special_tokens=True)

    return detailed_summary, one_liner


# =========================
# UI
# =========================
st.title("🔎 Candidate Recommendation Engine (Open Source)")
st.caption(
    "Embeddings: sentence-transformers/all-MiniLM-L6-v2 • Summaries: google/flan-t5-small")

job_desc = st.text_area(
    "📝 Paste Job Description",
    height=180,
    placeholder="e.g., NLP/ML engineer with Python, PyTorch, Transformers, and MLOps…"
)

uploaded_files = st.file_uploader(
    "📂 Upload resumes (.txt, .docx, .pdf) — one or many",
    type=["txt", "docx", "pdf"],
    accept_multiple_files=True
)

requested_top_n = st.slider("How many top candidates to display?", 1, 10, 5)

if st.button("Match Candidates"):
    if not job_desc:
        st.error("Please enter a job description.")
        st.stop()
    if not uploaded_files:
        st.error("Please upload at least one resume (.txt, .docx, .pdf).")
        st.stop()

    # Read & normalize resumes
    names: List[str] = []
    texts: List[str] = []
    infos: List[str] = []
    for uf in uploaded_files:
        text, info = load_resume_text(uf)
        # Collapse excessive whitespace to keep prompts compact
        text = " ".join(text.split())
        names.append(uf.name)
        texts.append(text)
        infos.append(info)

    top_n = min(requested_top_n, len(texts))

    # ---- Timed: embedding steps ----
    (job_vec,), t_job = timed(embed_model.encode, [job_desc])
    resume_vecs, t_resumes = timed(embed_model.encode, texts)

    # ---- Timed: cosine similarity ----
    sims, t_sim = timed(lambda a, b: cosine_similarity(
        [a], b).flatten(), job_vec, resume_vecs)

    order = np.argsort(sims)[::-1][:top_n]

    st.subheader("🎯 Top Matches")
    for i in order:
        cand_name = names[i]
        score = float(sims[i])
        parser_info = infos[i]

        # ---- Timed: summaries for this candidate ----
        (detailed_summary, one_liner), t_sum = timed(
            summarize_fit, job_desc, texts[i])

        with st.container():
            st.markdown(
                f"**👤 Candidate:** `{cand_name}`  •  _parsed via {parser_info}_")
            st.markdown(f"**📊 Cosine Similarity:** `{score:.4f}`")
            st.markdown(f"**📝 Detailed Summary:** {detailed_summary}")
            st.markdown(f"**💡 One-liner Reason:** {one_liner}")
            with st.expander("⏱ Timing for this candidate"):
                st.write({
                    "embedding_job_sec": round(t_job, 4),
                    "embedding_all_resumes_sec": round(t_resumes, 4),
                    "cosine_similarity_sec": round(t_sim, 4),
                    "summary_generation_sec": round(t_sum, 4),
                })
            st.markdown("---")

# -----------------------------
# Cosine Similarity Formula
# -----------------------------
with st.expander("📐 Formula: Cosine Similarity (with quick example)"):
    st.markdown(
        r"""
**Cosine similarity** between vectors \(A\) and \(B\):  
\[
\cos(\theta) = \frac{A \cdot B}{\|A\|\,\|B\|}
\]

**Toy example:**  
\(A=[1,2],\; B=[2,3]\)  
- Dot: \(1\cdot2 + 2\cdot3 = 8\)  
- Norms: \(\|A\|=\sqrt{5},\; \|B\|=\sqrt{13}\)  
- \(\cos(\theta)=\frac{8}{\sqrt{5}\sqrt{13}} \approx 0.993\)
"""
    )
