        overflow-wrap: anywhere;
    }
    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 9px;
        font-weight: 650;
        min-height: 42px;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🔬 Evidence-Based RAG Research Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="subtitle">
        Explore research papers with hybrid retrieval, cross-encoder reranking,
        evidence sufficiency checks, and source-page citations.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent

try:
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
except Exception as error:
    st.error("Could not load config.yaml. Make sure it is in the same folder as app.py.")
    st.exception(error)
    st.stop()


def get_setting(*paths, default=None):
    """Read the first available value among candidate dotted YAML paths."""
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
    "models.embedding_model",
    "embedding_model",
    "embedding.model",
    default="text-embedding-3-small",
)
GENERATION_MODEL = get_setting(
    "models.generation",
    "models.generation_model",
    "generation_model",
    "llm.model",
    default="gpt-4.1-mini",
)
RERANKER_MODEL = get_setting(
    "models.reranker",
    "models.reranker_model",
    "reranker_model",
    "retrieval.reranker_model",
    default="cross-encoder/ms-marco-MiniLM-L-6-v2",
)
DENSE_K = int(get_setting("retrieval.dense_top_k", "retrieval.dense_k", "dense_k", default=10))
BM25_K = int(get_setting("retrieval.bm25_top_k", "retrieval.bm25_k", "bm25_k", default=10))
RRF_K = int(get_setting("retrieval.rrf_k", default=60))
RERANK_TOP_K = int(
    get_setting(
        "retrieval.rerank_top_k",
        "retrieval.top_k",
        "rerank_top_k",
        default=5,
    )
)

# Public-demo safety limits. Adjust only if you understand the added API cost.
MAX_PDF_FILES = 3
MAX_FILE_SIZE_MB = 10
MAX_PAGES_PER_PDF = 100
MAX_UPLOADED_CHUNKS = 500
CHUNK_SIZE = int(get_setting("chunking.chunk_size", default=1200))
CHUNK_OVERLAP = int(get_setting("chunking.overlap", default=180))
EMBEDDING_BATCH_SIZE = 64

if (
    min(DENSE_K, BM25_K, RERANK_TOP_K) < 1
    or RRF_K < 0
    or not 0 <= CHUNK_OVERLAP < CHUNK_SIZE
):
    st.error("Invalid retrieval/chunking settings. Use positive top-k values, "
             "a nonnegative RRF constant, and 0 <= overlap < chunk size.")
    st.stop()


def clear_research_result():
    """Remove the previous answer and the context that identifies it."""
    for key in ("research_result", "research_question", "research_corpus_label"):
        st.session_state.pop(key, None)



# ============================================================
# 3. DUAL-MODE API KEY: HOSTED KEY FIRST, USER KEY OPTIONAL
# ============================================================

hosted_api_key = ""
try:
    hosted_api_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
except Exception:
    # Streamlit Secrets may not exist during local development.
    hosted_api_key = ""

if not hosted_api_key:
    hosted_api_key = os.getenv("OPENAI_API_KEY", "").strip()

with st.sidebar:
    st.markdown("## ⚙️ API Configuration")

    if hosted_api_key:
        api_mode = st.radio(
            "Choose API access",
            options=[
                "Hosted demo key (recommended)",
                "Use my own API key",
            ],
            index=0,
            key="api_access_mode",
            help="The hosted key is selected by default. You can switch to your own key at any time.",
        )
    else:
        st.info(
            "A hosted demo key has not been configured. Enter your own key below, "
            "or configure OPENAI_API_KEY in Streamlit Secrets."
        )
        api_mode = "Use my own API key"

    user_api_key = ""
    if api_mode == "Use my own API key":
        user_api_key = st.text_input(
            "Your OpenAI API Key",
            type="password",
