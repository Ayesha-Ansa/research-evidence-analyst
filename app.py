
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


# ---------- PAGE SETUP ----------
st.set_page_config(
    page_title="Research Evidence Analyst",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Evidence Analyst")
st.caption(
    "Ask questions about research papers and receive "
    "evidence-grounded answers with PDF page references."
)

BASE_DIR = Path(__file__).parent


# ---------- LOAD CONFIGURATION ----------
with open(BASE_DIR / "config.yaml", encoding="utf-8") as file:
    config = yaml.safe_load(file)


# ---------- API CLIENT ----------
# Users enter their own OpenAI API key in the sidebar.

st.sidebar.header("API Configuration")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="Enter your OpenAI API key",
    help=(
        "Enter your own OpenAI API key. "
        "Your key is used to make API requests for this session."
    )
)

if not api_key:
    st.info(
        "Please enter your OpenAI API key in the sidebar "
        "to start using the Research Evidence Analyst."
    )
    st.stop()

try:
    client = OpenAI(api_key=api_key)
except Exception:
    st.error("Could not initialize the OpenAI client. Check your API key.")
    st.stop()


# ---------- LOAD SAVED DOCUMENT DATA ----------
try:
    with open(BASE_DIR / "chunks.json", encoding="utf-8") as file:
        chunks = json.load(file)

    index = faiss.read_index(str(BASE_DIR / "evidence.index"))

except FileNotFoundError as error:
    st.error(
        f"Required project file not found: {error.filename}. "
        "Ensure config.yaml, chunks.json, and evidence.index "
        "are present in the project directory."
    )
    st.stop()

except Exception as error:
    st.error("Could not load the saved research data.")
    st.exception(error)
    st.stop()


if len(chunks) != index.ntotal:
    st.error(
        "Saved chunks and FAISS index do not match. "
        "Please check that both files belong to the same dataset."
    )
    st.stop()


# ---------- BUILD KEYWORD SEARCH AND RERANKER ----------
def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


bm25 = BM25Okapi([
    tokenize(chunk["text"]) for chunk in chunks
])


@st.cache_resource
def load_reranker(model_name):
    return CrossEncoder(model_name)


reranker = load_reranker(config["models"]["reranker"])


# ---------- HYBRID RETRIEVAL ----------
def reciprocal_rank_fusion(
    dense_indices,
    keyword_indices,
    k=60
):
    scores = {}

    for rank, idx in enumerate(dense_indices, start=1):
        idx = int(idx)

        if idx < 0:
            continue

        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

    for rank, idx in enumerate(keyword_indices, start=1):
        idx = int(idx)

        if idx < 0:
            continue

        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


def retrieve_evidence(question):
    response = client.embeddings.create(
        model=config["models"]["embedding"],
        input=[question]
    )

    query_vector = np.array(
        [response.data[0].embedding],
        dtype="float32"
    )

    faiss.normalize_L2(query_vector)

    dense_k = min(
        config["retrieval"]["dense_top_k"],
        index.ntotal
    )

    keyword_k = min(
        config["retrieval"]["bm25_top_k"],
        len(chunks)
    )

    _, dense_indices = index.search(
        query_vector,
        dense_k
    )

    keyword_scores = bm25.get_scores(
        tokenize(question)
    )

    keyword_indices = np.argsort(
        keyword_scores
    )[::-1][:keyword_k]

    fused = reciprocal_rank_fusion(
        dense_indices[0],
        keyword_indices,
        config["retrieval"]["rrf_k"]
    )

    candidates = fused[:min(20, len(fused))]

    if not candidates:
        return []

    pairs = [
        (question, chunks[idx]["text"])
        for idx, _ in candidates
    ]

    rerank_scores = reranker.predict(pairs)

    ranked = sorted(
        [
            {
                "index": idx,
                "rrf_score": rrf_score,
                "rerank_score": float(score)
            }
            for (idx, rrf_score), score
            in zip(candidates, rerank_scores)
        ],
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    evidence = []

    for item in ranked[
        :config["retrieval"]["rerank_top_k"]
    ]:
        chunk = chunks[item["index"]]

        evidence.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
            "rrf_score": item["rrf_score"],
            "rerank_score": item["rerank_score"]
        })

    return evidence


# ---------- EVIDENCE SUFFICIENCY CHECK ----------
def assess_evidence(question, evidence):
    if not evidence:
        return False, "No evidence passages were retrieved."

    evidence_text = "\n\n".join(
        f"[Source: {item['source']}, PDF page: {item['page']}]\n"
        f"{item['text'][:1200]}"
        for item in evidence
    )

    response = client.chat.completions.create(
        model=config["models"]["generation"],
        temperature=0,
        max_tokens=10,
        messages=[
            {
                "role": "system",
                "content": (
                    config["prompts"]["abstention"]
                    + "\nReturn exactly one label: "
                    "ANSWERABLE or INSUFFICIENT."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Retrieved evidence:\n{evidence_text}"
                )
            }
        ]
    )

    decision = (
        response.choices[0].message.content or ""
    ).strip().upper()

    if decision == "ANSWERABLE":
        return True, "The model judged the evidence answerable."

    return (
        False,
        "Evidence was insufficient or the decision was unclear."
    )


# ---------- ANSWER GENERATION ----------
def generate_answer(question, evidence):
    evidence_text = "\n\n".join(
        f"[Source: {item['source']}, PDF page: {item['page']}]\n"
        f"{item['text']}"
        for item in evidence
    )

    response = client.chat.completions.create(
        model=config["models"]["generation"],
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    config["prompts"]["answer"]
                    + "\nUse citations in this format: "
                    "[Exact filename.pdf, p. X]. "
                    "Only cite sources and pages in the supplied evidence."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Retrieved evidence:\n{evidence_text}"
                )
            }
        ]
    )

    return response.choices[0].message.content or ""


# ---------- CITATION CHECK ----------
def validate_citations(answer, evidence):
    available = {
        (item["source"], item["page"])
        for item in evidence
    }

    pattern = (
        r"([^()\[\];\n]+?\.pdf)\s*,\s*"
        r"pp?\.?\s*(\d+(?:\s*[-–]\s*\d+)?"
        r"(?:\s*,\s*\d+(?:\s*[-–]\s*\d+)?)*)"
    )

    results = []

    for match in re.finditer(
        pattern,
        answer,
        re.IGNORECASE
    ):
        source = match.group(1).strip()
        page_text = match.group(2)

        for part in re.split(r"\s*,\s*", page_text):
            range_match = re.fullmatch(
                r"(\d+)\s*[-–]\s*(\d+)",
                part
            )

            if range_match:
                start, end = map(
                    int,
                    range_match.groups()
                )

                pages = range(start, end + 1)
            else:
                pages = [int(part)]

            for page in pages:
                results.append({
                    "source": source,
                    "page": page,
                    "present_in_evidence": (
                        (source, page) in available
                    )
                })

    return results


# ---------- COMPLETE RAG WORKFLOW ----------
def research_assistant(question):
    evidence = retrieve_evidence(question)

    answerable, reason = assess_evidence(
        question,
        evidence
    )

    if not answerable:
        answer = (
            "I don't have sufficient evidence in the available "
            "research papers to answer this question reliably."
        )
    else:
        answer = generate_answer(
            question,
            evidence
        )

    return {
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "assessment": reason,
        "citations": validate_citations(
            answer,
            evidence
        )
    }


# ---------- SIDEBAR: PROJECT INFORMATION ----------
st.sidebar.divider()

st.sidebar.header("About this project")

st.sidebar.write(
    "Hybrid retrieval combines semantic search, BM25 keyword "
    "search, Reciprocal Rank Fusion, and cross-encoder reranking."
)

st.sidebar.write(
    "Answers are generated from retrieved research evidence, "
    "with an evidence sufficiency check before generation."
)

st.sidebar.caption(
    f"Loaded {len(chunks):,} chunks from "
    f"{len(set(c['source'] for c in chunks))} research papers."
)


# ---------- STREAMLIT INTERFACE ----------
with st.form("research_question_form"):
    question = st.text_area(
        "Ask a question about the research papers",
        placeholder=(
            "Example: What are the main challenges of "
            "distributed PCA?"
        )
    )

    submitted = st.form_submit_button(
        "Research",
        type="primary"
    )


if submitted:
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner(
                "Retrieving evidence and preparing your answer..."
            ):
                result = research_assistant(
                    question.strip()
                )

            st.subheader("Answer")
            st.write(result["answer"])

            st.caption(
                f"Evidence assessment: {result['assessment']}"
            )

            if result["citations"]:
                st.subheader("Citation checks")

                for citation in result["citations"]:
                    status = (
                        "Found in retrieved evidence"
                        if citation["present_in_evidence"]
                        else "Not found in retrieved evidence"
                    )

                    st.write(
                        f"- {citation['source']}, "
                        f"p. {citation['page']}: {status}"
                    )

            else:
                st.caption(
                    "No citations matching the expected citation "
                    "format were detected."
                )

            st.subheader("Retrieved evidence")

            for rank, item in enumerate(
                result["evidence"],
                start=1
            ):
                with st.expander(
                    f"{rank}. {item['source']} — "
                    f"PDF page {item['page']}"
                ):
                    st.caption(
                        f"Reranker score: "
                        f"{item['rerank_score']:.4f}"
                    )

                    st.write(item["text"])

        except Exception:
            st.error(
                "The application encountered an error while "
                "processing your question. Check your API key, "
                "API access, model configuration, and deployment logs."
            )
            st.exception(Exception(
                "Processing failed. See the error details above "
                "in the application logs if available."
            ))
