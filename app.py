
import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
import yaml
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Evidence-Based RAG Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 2.35rem;
        font-weight: 750;
        letter-spacing: -0.8px;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.6rem;
    }

    .metric-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 12px;
        padding: 18px 20px;
        min-height: 112px;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 750;
        line-height: 1.25;
    }

    .section-heading {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
    }

    .citation-chip {
        display: inline-block;
        background: rgba(59, 130, 246, 0.10);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 6px;
        padding: 4px 8px;
        margin: 3px 4px 3px 0;
        font-size: 0.82rem;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 9px;
        font-weight: 650;
        min-height: 44px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔬 Evidence-Based RAG Research Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Query research papers using hybrid retrieval, cross-encoder
        reranking, evidence sufficiency checks, and source verification.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. LOAD CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent

try:
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
except Exception as error:
    st.error(f"Could not load config.yaml: {error}")
    st.stop()


def get_setting(*paths, default=None):
    """Return the first available setting from a list of dotted paths."""
    for path in paths:
        value = config
        found = True

        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                found = False
                break
            value = value[key]

        if found and value is not None:
            return value

    return default


EMBEDDING_MODEL = get_setting(
    "models.embedding",
    "embedding_model",
    "embedding.model",
    default="text-embedding-3-small",
)

GENERATION_MODEL = get_setting(
    "models.generation",
    "generation_model",
    "llm.model",
    default="gpt-4.1-mini",
)

RERANKER_MODEL = get_setting(
    "models.reranker",
    "reranker_model",
    "retrieval.reranker_model",
    default="cross-encoder/ms-marco-MiniLM-L-6-v2",
)

DENSE_K = int(
    get_setting("retrieval.dense_k", "dense_k", default=10)
)

BM25_K = int(
    get_setting("retrieval.bm25_k", "bm25_k", default=10)
)

RERANK_TOP_K = int(
    get_setting(
        "retrieval.rerank_top_k",
        "retrieval.top_k",
        "rerank_top_k",
        default=5,
    )
)


# ============================================================
# 5. API KEY CONFIGURATION
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.caption("Connect your OpenAI API key to run research queries.")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Enter your API key",
        help="Your key is used to make OpenAI requests during this session.",
    )

    st.caption(
        "Use your own API key. API usage may incur charges under your account."
    )

if not api_key:
    st.info(
        "Enter your OpenAI API key in the sidebar to start researching."
    )
    st.stop()

try:
    client = OpenAI(api_key=api_key)
except Exception as error:
    st.error("Could not initialize the OpenAI client. Check your API key.")
    st.exception(error)
    st.stop()


# ============================================================
# 6. LOAD CORPUS AND FAISS INDEX
# ============================================================

@st.cache_resource
def load_corpus_and_index(base_dir):
    with open(
        Path(base_dir) / "chunks.json",
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    index = faiss.read_index(str(Path(base_dir) / "evidence.index"))

    if len(chunks) != index.ntotal:
        raise ValueError(
            f"Corpus/index mismatch: {len(chunks)} chunks but "
            f"{index.ntotal} vectors."
        )

    return chunks, index


try:
    chunks, index = load_corpus_and_index(str(BASE_DIR))
except Exception as error:
    st.error(
        "Could not load the research corpus. Make sure chunks.json "
        "and evidence.index are in the same folder as app.py."
    )
    st.exception(error)
    st.stop()


def get_chunk_text(chunk):
    """Support common text field names in the saved chunk objects."""
    return str(
        chunk.get("text")
        or chunk.get("content")
        or chunk.get("page_content")
        or ""
    )


def get_chunk_source(chunk):
    """Return the source filename stored in a chunk."""
    return str(
        chunk.get("source")
        or chunk.get("filename")
        or chunk.get("file_name")
        or chunk.get("metadata", {}).get("source")
        or "Unknown source"
    )


def get_chunk_page(chunk):
    """Return the PDF page number stored in a chunk."""
    metadata = chunk.get("metadata", {})

    return (
        chunk.get("page")
        or chunk.get("page_number")
        or metadata.get("page")
        or metadata.get("page_number")
        or "Unknown"
    )


# ============================================================
# 7. BUILD BM25 INDEX AND LOAD RERANKER
# ============================================================

@st.cache_resource
def build_bm25_index(corpus_chunks):
    tokenized_corpus = [
        get_chunk_text(chunk).lower().split()
        for chunk in corpus_chunks
    ]
    return BM25Okapi(tokenized_corpus)


@st.cache_resource
def load_reranker(model_name):
    return CrossEncoder(model_name)


try:
    bm25 = build_bm25_index(chunks)
    reranker = load_reranker(RERANKER_MODEL)
except Exception as error:
    st.error("Could not initialize the retrieval components.")
    st.exception(error)
    st.stop()


# ============================================================
# 8. SIDEBAR CORPUS INFORMATION
# ============================================================

with st.sidebar:
    st.divider()
    st.markdown("## 📚 Research Corpus")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Chunks", f"{len(chunks):,}")

    with col2:
        st.metric("Vectors", f"{index.ntotal:,}")

    unique_sources = sorted(
        {get_chunk_source(chunk) for chunk in chunks}
    )

    st.metric("Research Papers", len(unique_sources))

    with st.expander("View source documents"):
        for source in unique_sources:
            st.write(f"📄 {source}")

    st.divider()
    st.markdown("### Retrieval Pipeline")
    st.caption("1. Dense semantic search")
    st.caption("2. BM25 keyword search")
    st.caption("3. Reciprocal Rank Fusion")
    st.caption("4. Cross-encoder reranking")
    st.caption("5. Evidence sufficiency check")
    st.caption("6. Grounded answer and citation checks")


# ============================================================
# 9. HYBRID RETRIEVAL
# ============================================================

def retrieve_evidence(question):
    """Combine dense and keyword retrieval, then rerank candidates."""

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question],
    )

    query_vector = np.array(
        [response.data[0].embedding],
        dtype=np.float32,
    )

    faiss.normalize_L2(query_vector)

    dense_scores, dense_indices = index.search(
        query_vector,
        min(DENSE_K, len(chunks)),
    )

    dense_ranking = [
        int(idx)
        for idx in dense_indices[0]
        if idx >= 0
    ]

    bm25_scores = bm25.get_scores(question.lower().split())

    bm25_ranking = [
        int(idx)
        for idx in np.argsort(bm25_scores)[::-1][:BM25_K]
    ]

    # Reciprocal Rank Fusion combines rankings without comparing
    # raw dense and BM25 scores directly.
    rrf_scores = {}

    for ranking in (dense_ranking, bm25_ranking):
        for rank, chunk_idx in enumerate(ranking, start=1):
            rrf_scores[chunk_idx] = (
                rrf_scores.get(chunk_idx, 0.0)
                + 1.0 / (60 + rank)
            )

    candidate_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )[:20]

    if not candidate_indices:
        return []

    pairs = [
        (question, get_chunk_text(chunks[idx]))
        for idx in candidate_indices
    ]

    rerank_scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(candidate_indices, rerank_scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    evidence = []

    for chunk_idx, rerank_score in ranked_results[:RERANK_TOP_K]:
        chunk = chunks[chunk_idx]

        evidence.append(
            {
                "source": get_chunk_source(chunk),
                "page": get_chunk_page(chunk),
                "text": get_chunk_text(chunk),
                "rerank_score": float(rerank_score),
                "rrf_score": float(rrf_scores[chunk_idx]),
            }
        )

    return evidence


# ============================================================
# 10. EVIDENCE SUFFICIENCY CHECK
# ============================================================

def assess_evidence(question, evidence):
    """Decide whether the retrieved passages can answer the question."""

    if not evidence:
        return False

    evidence_text = "\n\n".join(
        f"[Source: {item['source']}, PDF page: {item['page']}]\n"
        f"{item['text'][:1200]}"
        for item in evidence
    )

    prompt = f"""
You are an evidence sufficiency evaluator.

Decide whether the supplied research excerpts contain enough relevant
information to answer the user's question.

Return exactly one word:
ANSWERABLE
or
INSUFFICIENT

Do not use outside knowledge to fill gaps.

Question:
{question}

Retrieved evidence:
{evidence_text}
"""

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Evaluate evidence sufficiency conservatively.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    decision = response.choices[0].message.content or ""

    return decision.strip().upper() == "ANSWERABLE"


# ============================================================
# 11. ANSWER GENERATION
# ============================================================

def generate_answer(question, evidence):
    """Generate an answer using only retrieved evidence."""

    evidence_text = "\n\n".join(
        f"[Source: {item['source']}, PDF page: {item['page']}]\n"
        f"{item['text']}"
        for item in evidence
    )

    prompt = f"""
Answer the research question using only the supplied evidence.

Rules:
1. Do not introduce facts that are not supported by the excerpts.
2. If the evidence does not establish a detail, say so explicitly.
3. Cite factual claims using this format:
   [Exact filename.pdf, p. X]
4. Use the exact source filename and page number supplied in the evidence.
5. Distinguish findings from different papers when relevant.
6. Do not invent citations.
7. Write a clear, concise answer with useful structure.

Research question:
{question}

Retrieved evidence:
{evidence_text}
"""

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful research assistant. "
                    "Ground every factual claim in the provided evidence."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content or ""


# ============================================================
# 12. CITATION VALIDATION
# ============================================================

def validate_citations(answer, evidence):
    """
    Check whether cited source/page pairs appear in the retrieved
    evidence. This checks citation matching, not claim-level entailment.
    """

    pattern = re.compile(
        r"\[([^,\[\]]+\.pdf)\s*,\s*p{1,2}\.?\s*(\d+)"
        r"(?:\s*[-–]\s*(\d+))?\]",
        re.IGNORECASE,
    )

    citations = pattern.findall(answer)

    available_pairs = {
        (
            os.path.basename(str(item["source"])).strip().lower(),
            str(item["page"]).strip(),
        )
        for item in evidence
    }

    results = []

    for source, start_page, end_page in citations:
        source_normalized = os.path.basename(source).strip().lower()

        pages = [start_page]

        if end_page:
            try:
                start = int(start_page)
                end = int(end_page)

                if end >= start and end - start <= 100:
                    pages = [str(page) for page in range(start, end + 1)]
            except ValueError:
                pages = [start_page]

        for page in pages:
            matched = (
                source_normalized,
                str(page),
            ) in available_pairs

            results.append(
                {
                    "Source": source.strip(),
                    "Page": page,
                    "Matched to retrieved evidence": (
                        "Yes" if matched else "No"
                    ),
                }
            )

    return results


# ============================================================
# 13. MAIN RESEARCH WORKFLOW
# ============================================================

def research_assistant(question):
    evidence = retrieve_evidence(question)

    if not evidence:
        return {
            "answerable": False,
            "answer": (
                "I could not retrieve relevant passages from the "
                "available research papers."
            ),
            "evidence": [],
            "citations": [],
        }

    answerable = assess_evidence(question, evidence)

    if answerable:
        answer = generate_answer(question, evidence)
        citations = validate_citations(answer, evidence)
    else:
        answer = (
            "I couldn't find sufficient evidence in the available "
            "research papers to answer this question reliably. "
            "Try narrowing the question or adding a relevant paper."
        )
        citations = []

    return {
        "answerable": answerable,
        "answer": answer,
        "evidence": evidence,
        "citations": citations,
    }


# ============================================================
# 14. RESEARCH QUERY INTERFACE
# ============================================================

st.markdown("### 💡 Ask a Research Question")

with st.form("research_form"):
    question = st.text_area(
        "Research Query",
        placeholder=(
            "Example: What are the main differences between "
            "distributed PCA and sparse PCA?"
        ),
        height=110,
        help="Ask a question that can be investigated using your uploaded papers.",
    )

    submitted = st.form_submit_button(
        "🔎 Synthesize Answer",
        use_container_width=True,
    )


if submitted:
    if not question.strip():
        st.warning("Please enter a research question.")
    else:
        with st.spinner(
            "Searching papers, reranking evidence, and checking sufficiency..."
        ):
            try:
                result = research_assistant(question.strip())
                st.session_state["research_result"] = result
                st.session_state["research_question"] = question.strip()

            except Exception as err:
                st.error(
                    "The research query could not be completed. "
                    "Check your API key, configuration, and model access."
                )
                st.exception(err)


# ============================================================
# 15. RESULTS DASHBOARD
# ============================================================

if "research_result" in st.session_state:
    result = st.session_state["research_result"]

    st.divider()
    st.markdown("## 📊 Research Results")

    matched_count = sum(
        1
        for item in result["citations"]
        if item["Matched to retrieved evidence"] == "Yes"
    )

    total_citations = len(result["citations"])

    status_label = (
        "Sufficient evidence"
        if result["answerable"]
        else "Insufficient evidence"
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Evidence Sufficiency</div>
                <div class="metric-value">
                    {"✅" if result["answerable"] else "⚠️"}
                    {status_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Retrieved Passages</div>
                <div class="metric-value">{len(result["evidence"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Citations Matched to Retrieved Evidence
                </div>
                <div class="metric-value">
                    {matched_count}/{total_citations}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Citation matching checks whether the cited source and page "
        "appear in retrieved evidence. It does not independently prove "
        "that each cited passage supports its associated claim."
    )

    st.divider()

    answer_column, citation_column = st.columns(
        [1.6, 1],
        gap="large",
    )

    with answer_column:
        st.markdown("### 📝 Generated Synthesis")
        st.markdown(result["answer"])

    with citation_column:
        st.markdown("### 🔗 Citation Verification")

        if result["citations"]:
            st.dataframe(
                result["citations"],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No citations were extracted from the generated answer."
            )

    st.divider()
    st.markdown("### 📚 Retrieved Evidence")

    if not result["evidence"]:
        st.info("No evidence passages were retrieved.")
    else:
        for position, item in enumerate(result["evidence"], start=1):
            source = item["source"]
            page = item["page"]

            with st.expander(
                f"{position}. {source} · Page {page}"
            ):
                st.markdown(
                    f"**Cross-encoder score:** "
                    f"`{item['rerank_score']:.4f}`"
                )
                st.markdown(
                    f"**RRF score:** `{item['rrf_score']:.4f}`"
                )
                st.markdown("**Passage**")
                st.write(item["text"])

    st.divider()
    st.caption(
        "Evidence-Based RAG Research Assistant · "
        "Answers are limited to the indexed research corpus."
    )
