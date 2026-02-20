import os
import re
import logging
import psycopg2
import pandas as pd
import streamlit as st
from uuid import uuid4
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import html as _html

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
DB_CFG = dict(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    port=os.getenv("DB_PORT", "5433"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
)
COLLECTION_NAME   = os.getenv("COLLECTION_NAME")
DATABASE_LOCATION = os.getenv("DATABASE_LOCATION")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL        = os.getenv("CHAT_MODEL")
BATCH_SIZE        = 500

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&family=Sora:wght@400;600;700&display=swap');

:root {
    --bg0:#07090f; --bg1:#0d1117; --bg2:#111827; --bg3:#1a2235; --bg4:#1f2d42;
    --blue:#3b82f6; --blue-d:#1d4ed8; --cyan:#22d3ee; --purple:#a78bfa;
    --green:#34d399; --t1:#f1f5f9; --t2:#94a3b8; --t3:#64748b;
    --border:rgba(255,255,255,0.07); --radius:16px; --shadow:0 8px 32px rgba(0,0,0,0.5);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,.stApp{background:var(--bg0)!important;font-family:'DM Sans',sans-serif!important;color:var(--t1)!important;}
#MainMenu,footer,header,.stDeployButton{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}

/* Sidebar */
[data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] .block-container{padding:1.5rem 1.2rem!important;}
.brand{display:flex;align-items:center;gap:10px;padding-bottom:1.2rem;border-bottom:1px solid var(--border);margin-bottom:1.4rem;}
.brand-icon{width:36px;height:36px;background:linear-gradient(135deg,var(--blue),var(--cyan));border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:17px;flex-shrink:0;}
.brand-name{font-family:'Sora',sans-serif;font-size:1.05rem;font-weight:700;background:linear-gradient(90deg,var(--blue),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sb-label{font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--t3);margin-bottom:.6rem;}
.route-card{background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:8px 10px;margin-bottom:6px;font-size:.78rem;color:var(--t2);line-height:1.5;}
.route-card strong{color:var(--cyan);display:block;margin-bottom:2px;}
.route-tag{display:inline-block;background:rgba(52,211,153,.12);color:var(--green);border-radius:4px;padding:1px 6px;font-size:.68rem;font-weight:600;margin-top:4px;}
.stButton>button{background:var(--bg3)!important;color:var(--t1)!important;border:1px solid var(--border)!important;border-radius:10px!important;font-family:'DM Sans',sans-serif!important;font-size:.85rem!important;font-weight:500!important;padding:.55rem 1rem!important;width:100%!important;transition:all .18s ease!important;}
.stButton>button:hover{background:var(--bg4)!important;border-color:var(--blue)!important;transform:translateY(-1px)!important;}

/* Chat header */
.chat-header{position:sticky;top:0;z-index:200;display:flex;align-items:center;gap:12px;padding:.9rem 1.8rem;background:rgba(13,17,23,.92);backdrop-filter:blur(16px);border-bottom:1px solid var(--border);}
.header-av{width:40px;height:40px;border-radius:50%;background:linear-gradient(135deg,var(--blue),var(--purple));display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 0 16px rgba(59,130,246,.35);}
.header-title{font-family:'Sora',sans-serif;font-size:.95rem;font-weight:600;}
.header-sub{font-size:.72rem;color:var(--green);display:flex;align-items:center;gap:5px;margin-top:1px;}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:livepulse 2s ease-in-out infinite;}
@keyframes livepulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(.8);}}

/* Messages */
.msg-row{display:flex;align-items:flex-end;gap:8px;animation:msgIn .25s ease both;max-width:100%;}
@keyframes msgIn{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
.msg-row.user{flex-direction:row-reverse;}
.msg-row.bot{flex-direction:row;}
.av{width:30px;height:30px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:13px;margin-bottom:2px;}
.av.u{background:linear-gradient(135deg,var(--blue),var(--blue-d));}
.av.b{background:var(--bg3);border:1px solid var(--border);}
.bubble{max-width:62%;padding:.7rem 1rem;font-size:.875rem;line-height:1.65;word-break:break-word;box-shadow:var(--shadow);}
.bubble.user{background:linear-gradient(135deg,var(--blue),var(--blue-d));color:#fff;border-radius:18px 18px 4px 18px;}
.bubble.bot{background:var(--bg3);color:var(--t1);border:1px solid var(--border);border-radius:18px 18px 18px 4px;}
.bubble table{width:100%;border-collapse:collapse;font-size:.8rem;margin:6px 0;background:rgba(0,0,0,.3);border-radius:8px;overflow:hidden;}
.bubble th{background:rgba(59,130,246,.18);color:var(--cyan);padding:6px 10px;text-align:left;font-weight:600;border-bottom:1px solid var(--border);font-size:.75rem;}
.bubble td{padding:5px 10px;border-bottom:1px solid rgba(255,255,255,.04);}
.bubble tr:last-child td{border-bottom:none;}
.bubble code{background:rgba(34,211,238,.1);color:var(--cyan);padding:1px 5px;border-radius:4px;font-size:.82em;}
.bubble strong{color:var(--cyan);}
.src-badge{display:inline-block;margin-top:6px;font-size:.67rem;color:var(--t3);background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:4px;padding:1px 6px;}

/* Typing dots */
.typing-row{display:flex;align-items:flex-end;gap:8px;animation:msgIn .25s ease both;padding:.3rem 1.4rem;}
.typing-bubble{background:var(--bg3);border:1px solid var(--border);border-radius:18px 18px 18px 4px;padding:.75rem 1.1rem;display:flex;align-items:center;gap:5px;}
.dot{width:7px;height:7px;border-radius:50%;animation:dotbounce 1.1s ease-in-out infinite;}
.dot:nth-child(1){background:var(--blue);animation-delay:0s;}
.dot:nth-child(2){background:var(--cyan);animation-delay:.18s;}
.dot:nth-child(3){background:var(--purple);animation-delay:.36s;}
@keyframes dotbounce{0%,80%,100%{transform:translateY(0);opacity:.5;}40%{transform:translateY(-7px);opacity:1;}}

/* Welcome */
.welcome{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center;}
.welcome-icon{width:72px;height:72px;border-radius:22px;background:linear-gradient(135deg,var(--blue),var(--purple));display:flex;align-items:center;justify-content:center;font-size:34px;margin-bottom:1.4rem;box-shadow:0 12px 40px rgba(59,130,246,.35);}
.welcome-title{font-family:'Sora',sans-serif;font-size:1.55rem;font-weight:700;margin-bottom:.5rem;}
.welcome-sub{font-size:.88rem;color:var(--t2);max-width:380px;line-height:1.7;margin-bottom:2rem;}
.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;max-width:480px;}
.chip{background:var(--bg3);border:1px solid var(--border);border-radius:20px;padding:6px 14px;font-size:.78rem;color:var(--t2);}

/* Input */
[data-testid="stChatInput"]>div{background:var(--bg2)!important;border:1px solid var(--border)!important;border-radius:14px!important;box-shadow:0 4px 24px rgba(0,0,0,.4)!important;}
[data-testid="stChatInput"]>div:focus-within{border-color:var(--blue)!important;box-shadow:0 0 0 2px rgba(59,130,246,.15)!important;}
[data-testid="stChatInput"] textarea{background:transparent!important;color:var(--t1)!important;font-family:'DM Sans',sans-serif!important;font-size:.9rem!important;}
[data-testid="stChatInput"] button{background:var(--blue)!important;border-radius:9px!important;}

::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-thumb{background:var(--bg4);border-radius:8px;}
</style>
"""

TYPING_HTML = """
<div class="typing-row">
  <div class="av b">🏦</div>
  <div class="typing-bubble">
    <div class="dot"></div><div class="dot"></div><div class="dot"></div>
  </div>
</div>
"""

WELCOME_HTML = """
<div class="welcome">
  <div class="welcome-icon">🏦</div>
  <div class="welcome-title">Bank Data Assistant</div>
  <div class="welcome-sub">
    Ask me anything about accounts, branches, balances, and irregularities.
    Powered by a smart vector knowledge base for instant, context-aware answers.
  </div>
  <div class="chips">
    <span class="chip">💰 What is the total balance?</span>
    <span class="chip">🏢 Highest balance branch?</span>
    <span class="chip">📊 Average interest rate</span>
    <span class="chip">⚠️ Irregularity summary</span>
    <span class="chip">🔍 Account 12345 info</span>
    <span class="chip">👋 How can you help?</span>
  </div>
</div>
"""

# ─────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────────────────────────
# Cached resources
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_vector_store():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DATABASE_LOCATION,
    )

@st.cache_resource
def load_llm():
    return ChatOllama(
        model=CHAT_MODEL,
        temperature=0,
        streaming=True,
        num_predict=256,  # Short = fast
        num_ctx=2048,     # Smaller window = fast
    )

# ─────────────────────────────────────────────────────────────────
# DB helper
# ─────────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(**DB_CFG, connect_timeout=5)

# ─────────────────────────────────────────────────────────────────
# INGESTION — precomputes aggregates into vector DB
# ─────────────────────────────────────────────────────────────────
def row_to_text(row):
    return (
        f"Account {row.accountno} belonging to {row.cust_name} "
        f"at branch {row.branch_name} has current balance {row.currentbalance:.2f} "
        f"with interest rate {row.intrate} percent. "
        f"Irregularity status: {row.irregularity}."
    )

def run_ingestion(vector_store):
    existing = vector_store.get()
    if existing and len(existing.get("ids", [])) > 0:
        st.warning(f"Vector store already has **{len(existing['ids']):,}** records. "
                   "Change COLLECTION_NAME in .env to re-ingest.")
        return

    status = st.empty()
    progress = st.progress(0)
    status.info("Connecting to database…")

    try:
        conn = get_conn()
        df = pd.read_sql("SELECT * FROM ccod_bal;", conn)
        conn.close()
    except Exception as e:
        status.error(f"DB error: {e}")
        return

    total_rows = len(df)
    status.info(f"Fetched {total_rows:,} rows. Building knowledge base…")
    all_docs = []

    # 1. Individual account docs
    for i, row in df.iterrows():
        all_docs.append(Document(
            page_content=row_to_text(row),
            metadata={"type": "account", "account_no": str(row.accountno),"branch_no": str(row.branchno),
                      "branch": str(row.branch_name), "source": "ccod_bal"}
        ))
    progress.progress(0.2)

    # 2. Precomputed branch aggregates
    branch_grp = df.groupby("branch_name").agg(
        total_balance=("currentbalance", "sum"),
        avg_balance=("currentbalance", "mean"),
        account_count=("accountno", "count"),
        avg_intrate=("intrate", "mean"),
        max_balance=("currentbalance", "max"),
        min_balance=("currentbalance", "min"),
    ).reset_index()

    for _, br in branch_grp.iterrows():
        all_docs.append(Document(
            page_content=(
                f"Branch {br.branch_name} aggregate: "
                f"total balance {br.total_balance:.2f}, avg balance {br.avg_balance:.2f}, "
                f"max balance {br.max_balance:.2f}, min balance {br.min_balance:.2f}, "
                f"avg interest rate {br.avg_intrate:.2f}%, total accounts {int(br.account_count)}."
            ),
            metadata={"type": "branch_aggregate", "branch": str(br.branch_name),
                      "total_balance": float(br.total_balance),
                      "avg_balance": float(br.avg_balance),
                      "account_count": int(br.account_count), "source": "precomputed"}
        ))
    progress.progress(0.35)

    # 3. Global summary
    all_docs.append(Document(
        page_content=(
            f"Global bank summary: total accounts {len(df):,}, "
            f"total balance {df.currentbalance.sum():,.2f}, "
            f"average balance {df.currentbalance.mean():,.2f}, "
            f"highest balance {df.currentbalance.max():,.2f}, "
            f"lowest balance {df.currentbalance.min():,.2f}, "
            f"average interest rate {df.intrate.mean():.2f}%."
        ),
        metadata={"type": "global_summary", "source": "precomputed"}
    ))
    progress.progress(0.4)

    # 4. Irregularity summaries per branch
    irreg_grp = df.groupby(["branch_name", "irregularity"]).size().reset_index(name="count")
    for _, ir in irreg_grp.iterrows():
        all_docs.append(Document(
            page_content=(
                f"Branch {ir.branch_name} has {int(ir['count'])} accounts "
                f"with irregularity status: {ir.irregularity}."
            ),
            metadata={"type": "irregularity", "branch": str(ir.branch_name),
                      "irregularity": str(ir.irregularity), "source": "precomputed"}
        ))
    progress.progress(0.5)

    # 5. Embed in batches
    total = len(all_docs)
    status.info(f"Embedding {total:,} documents ({total_rows:,} accounts + precomputed aggregates)…")
    for start in range(0, total, BATCH_SIZE):
        batch = all_docs[start: start + BATCH_SIZE]
        vector_store.add_documents(documents=batch, ids=[str(uuid4()) for _ in batch])
        progress.progress(0.5 + 0.5 * min(start + BATCH_SIZE, total) / total)

    progress.progress(1.0)
    status.success(f"Done! {total:,} documents embedded into vector knowledge base.")

# ─────────────────────────────────────────────────────────────────
# Query classifier
# ─────────────────────────────────────────────────────────────────
ACCOUNT_RE = re.compile(r'\b(\d{5,})\b')
AGGREGATE_KEYWORDS = {
    "highest", "lowest", "maximum", "minimum", "max", "min", "average", "avg",
    "total", "sum", "count", "how many", "most", "least", "top", "bottom",
    "all branches", "branch", "overall", "across", "entire", "irregular",
    "irregularity", "interest rate", "balance summary", "summary", "report",
}
GENERAL_RE = re.compile(
    r'\b(hello|hi|hey|good\s+morning|good\s+evening|good\s+afternoon|'
    r'how are you|what can you do|help|who are you|what is this|'
    r'thanks|thank you|bye|goodbye|assist|capabilities)\b', re.I
)

def classify_query(q: str) -> str:
    if ACCOUNT_RE.search(q):
        return "account"
    ql = q.lower()
    if any(kw in ql for kw in AGGREGATE_KEYWORDS):
        return "aggregate"
    if GENERAL_RE.search(q):
        return "general"
    return "rag"

# ─────────────────────────────────────────────────────────────────
# Handler 1 — Account lookup, vector DB only (<0.5s, no LLM)
# ─────────────────────────────────────────────────────────────────
def handle_account_query(query: str, vector_store) -> str:
    match = ACCOUNT_RE.search(query)
    if not match:
        return "I couldn't find an account number in your query."
    account_no = match.group(1)

    # Try metadata filter first (fastest)
    try:
        results = vector_store.get(where={"account_no": account_no}, limit=1)
        if results and results.get("documents"):
            return (f"**Account Information**\n\n{results['documents'][0]}\n\n"
                    f"<span class='src-badge'>⚡ Vector DB · exact match</span>")
    except Exception:
        pass

    # Fallback: similarity search
    docs = vector_store.similarity_search(f"Account number {account_no}", k=4)
    for doc in docs:
        if account_no in doc.page_content:
            return (f"**Account Information**\n\n{doc.page_content}\n\n"
                    f"<span class='src-badge'>⚡ Vector DB · similarity</span>")

    return f"No information found for account **{account_no}** in the knowledge base."

# ─────────────────────────────────────────────────────────────────
# Handler 2 — Aggregate queries, vector DB only (<1s, no LLM)
# ─────────────────────────────────────────────────────────────────
def handle_aggregate_query(query: str, vector_store) -> str:
    q = query.lower()

    # Global summary
    if any(x in q for x in ["total", "overall", "entire", "all account", "how many account", "how many customer", "global"]):
        docs = vector_store.similarity_search("global bank summary total balance accounts", k=2)
        for d in docs:
            if d.metadata.get("type") == "global_summary":
                return f"**Global Summary**\n\n{d.page_content}\n\n<span class='src-badge'>⚡ Precomputed · Vector DB</span>"

    # Top branches
    if "branch" in q and any(x in q for x in ["highest", "maximum", "most", "top", "best", "largest"]):
        docs = vector_store.similarity_search("branch total balance aggregate", k=20)
        branch_docs = [d for d in docs if d.metadata.get("type") == "branch_aggregate"]
        if branch_docs:
            top = sorted(branch_docs, key=lambda d: d.metadata.get("total_balance", 0), reverse=True)[:5]
            rows = "".join(
                f"| {d.metadata['branch']} | {d.metadata['total_balance']:,.2f} | {d.metadata['account_count']} |\n"
                for d in top
            )
            return (f"**Top Branches by Total Balance**\n\n"
                    f"| Branch | Total Balance | Accounts |\n|---|---|---|\n{rows}\n"
                    f"<span class='src-badge'>⚡ Precomputed · Vector DB</span>")

    # Bottom branches
    if "branch" in q and any(x in q for x in ["lowest", "minimum", "least", "bottom", "smallest"]):
        docs = vector_store.similarity_search("branch total balance aggregate", k=20)
        branch_docs = [d for d in docs if d.metadata.get("type") == "branch_aggregate"]
        if branch_docs:
            bottom = sorted(branch_docs, key=lambda d: d.metadata.get("total_balance", 0))[:5]
            rows = "".join(
                f"| {d.metadata['branch']} | {d.metadata['total_balance']:,.2f} | {d.metadata['account_count']} |\n"
                for d in bottom
            )
            return (f"**Bottom Branches by Total Balance**\n\n"
                    f"| Branch | Total Balance | Accounts |\n|---|---|---|\n{rows}\n"
                    f"<span class='src-badge'>⚡ Precomputed · Vector DB</span>")

    # Average / interest rate
    if any(x in q for x in ["average", "avg", "interest", "mean"]):
        docs = vector_store.similarity_search("average balance interest rate global", k=3)
        for d in docs:
            if d.metadata.get("type") == "global_summary":
                return f"**Averages**\n\n{d.page_content}\n\n<span class='src-badge'>⚡ Precomputed · Vector DB</span>"

    # Irregularity
    if any(x in q for x in ["irregular", "irregularity"]):
        docs = vector_store.similarity_search("irregularity accounts branch", k=10)
        irreg_docs = [d for d in docs if d.metadata.get("type") == "irregularity"]
        if irreg_docs:
            content = "\n".join(f"- {d.page_content}" for d in irreg_docs[:8])
            return f"**Irregularity Summary**\n\n{content}\n\n<span class='src-badge'>⚡ Precomputed · Vector DB</span>"

    # Branch general
    if "branch" in q:
        docs = vector_store.similarity_search(query, k=5)
        branch_docs = [d for d in docs if d.metadata.get("type") == "branch_aggregate"][:3]
        if branch_docs:
            content = "\n\n".join(d.page_content for d in branch_docs)
            return f"**Branch Information**\n\n{content}\n\n<span class='src-badge'>⚡ Vector DB</span>"

    # Fallback similarity
    docs = vector_store.similarity_search(query, k=3)
    if docs:
        return "\n\n".join(d.page_content for d in docs) + "\n\n<span class='src-badge'>📚 Vector DB</span>"

    return "I couldn't find matching data for your query in the knowledge base."

# ─────────────────────────────────────────────────────────────────
# Handler 3 — General conversational (LLM only, minimal context)
# ─────────────────────────────────────────────────────────────────
def stream_general(query: str, llm, recent: list):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are NYX-AI, a friendly bank data assistant. "
         "For greetings and general help, respond warmly and briefly. "
         "You can help with: account lookups, branch analytics, balances, irregularities. "
         "Keep responses under 3 sentences."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    chain = (
        {"question": RunnablePassthrough(), "chat_history": lambda _: recent}
        | prompt | llm | StrOutputParser()
    )
    return chain.stream(query)

# ─────────────────────────────────────────────────────────────────
# Handler 4 — RAG (open-ended, k=3, last 2 turns only)
# ─────────────────────────────────────────────────────────────────
def stream_rag(query: str, vector_store, llm, recent: list):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a concise bank data assistant. "
         "Use ONLY the context to answer. Be direct and brief. "
         "If context lacks info, say so in one sentence.\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    chain = (
        {
            "context": retriever | (lambda docs: "\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))),
            "question": RunnablePassthrough(),
            "chat_history": lambda _: recent,
        }
        | prompt | llm | StrOutputParser()
    )
    return chain.stream(query)

# ─────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────
def user_bubble(text: str):
    safe = _html.escape(text)
    st.markdown(
        f'<div class="msg-row user" style="padding:.2rem 1.4rem">'
        f'<div><div class="bubble user" style="padding-right: 20px; padding-left: 20px;">{safe}</div></div>'
        f'<div class=""></div></div>',
        unsafe_allow_html=True
    )

def bot_bubble_static(content: str):
    """For instant non-LLM responses."""
    col_icon, col_text, _ = st.columns([0.045, 0.62, 0.335])
    with col_icon:
        st.markdown('<div class="av b" style="margin-top:4px; margin-left: 50px;">🏦</div>', unsafe_allow_html=True)
    with col_text:
        st.markdown(
            f'<div class="bubble bot" style="max-width:100%">{content}</div>',
            unsafe_allow_html=True
        )

def bot_bubble_stream(stream_gen):
    """For streaming LLM responses. Returns full text."""
    col_icon, col_text, _ = st.columns([0.045, 0.62, 0.335])
    with col_icon:
        st.markdown('<div class="av b" style="margin-top:4px">🏦</div>', unsafe_allow_html=True)
    with col_text:
        ph = st.empty()
        full = ""
        for chunk in stream_gen:
            full += chunk
            ph.markdown(full + "▌")
        ph.markdown(full)
    return full

# ─────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NYX-AI | Bank Assistant", page_icon="🏦", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand">'
        '<div class="brand-icon">🏦</div>'
        '<div class="brand-name">NYX-AI</div>'
        '</div>', unsafe_allow_html=True
    )

    st.markdown('<div class="sb-label">Data Sync</div>', unsafe_allow_html=True)
    st.caption("Sync your database into the vector knowledge base once.")
    if st.button("📥 Ingest Data from SQL", use_container_width=True):
        vs = load_vector_store()
        run_ingestion(vs)

    st.divider()

    # st.markdown('<div class="sb-label">Query Routing</div>', unsafe_allow_html=True)
    # st.markdown("""
    # <div class="route-card"><strong>🔢 Account Number</strong>
    # Vector DB exact lookup<span class="route-tag">⚡ &lt;0.5s · No LLM</span></div>
    # <div class="route-card"><strong>📊 Aggregates / Analytics</strong>
    # Precomputed docs lookup<span class="route-tag">⚡ &lt;1s · No LLM</span></div>
    # <div class="route-card"><strong>🔍 Open-ended questions</strong>
    # Vector search k=3 + LLM<span class="route-tag">🤖 1–3s</span></div>
    # <div class="route-card"><strong>💬 Greetings / General</strong>
    # LLM only, minimal context<span class="route-tag">🤖 1–2s</span></div>
    # """, unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat header ───────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
  <div class="header-av">🏦</div>
  <div>
    <div class="header-title">Bank Data Assistant</div>
    <div class="header-sub"><span class="live-dot"></span>Online · Knowledge base powered</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── History ───────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(WELCOME_HTML, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            user_bubble(msg.content)
        else:
            bot_bubble_static(msg.content)

# ── Chat input ────────────────────────────────────────────────────

if user_question := st.chat_input("Ask about accounts, balances, branches, irregularities…"):

    user_bubble(user_question)
    st.session_state.messages.append(HumanMessage(user_question))

    # 3-dot typing animation
    typing_ph = st.empty()
    typing_ph.markdown(TYPING_HTML, unsafe_allow_html=True)

    full_response = ""

    try:
        qtype = classify_query(user_question)
        vector_store = load_vector_store()
        llm = load_llm()

        # Last 4 messages = 2 turns of context (speed + relevance)
        recent = list(st.session_state.messages[-5:-1])

        if qtype == "account":
            typing_ph.empty()
            full_response = handle_account_query(user_question, vector_store)
            bot_bubble_static(full_response)

        elif qtype == "aggregate":
            typing_ph.empty()
            full_response = handle_aggregate_query(user_question, vector_store)
            bot_bubble_static(full_response)

        elif qtype == "general":
            typing_ph.empty()
            full_response = bot_bubble_stream(stream_general(user_question, llm, recent))

        else:  # rag
            typing_ph.empty()
            full_response = bot_bubble_stream(stream_rag(user_question, vector_store, llm, recent))

    except Exception as e:
        typing_ph.empty()
        full_response = f"⚠️ Error: {str(e)}"
        bot_bubble_static(full_response)
        logging.error(f"Chat error: {e}", exc_info=True)

    if full_response:
        st.session_state.messages.append(AIMessage(full_response))

#___________________________________________________________________
#NEW POLISHED CODE
#___________________________________________________________________

# import os
# import re
# import time
# import logging
# import psycopg2
# import pandas as pd
# import streamlit as st
# from uuid import uuid4
# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# from langchain_ollama import OllamaEmbeddings, ChatOllama
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# import html as _html

# load_dotenv()

# # ─────────────────────────────────────────────────────────────────
# # PAGE CONFIG — must be the very first st.* call in the script
# # ─────────────────────────────────────────────────────────────────
# st.set_page_config(page_title="NYX-AI | Bank Assistant", page_icon="🏦", layout="wide")

# # ─────────────────────────────────────────────────────────────────
# # Config
# # ─────────────────────────────────────────────────────────────────
# DB_CFG = dict(
#     host=os.getenv("DB_HOST"),
#     database=os.getenv("DB_NAME"),
#     port=os.getenv("DB_PORT", "5433"),
#     user=os.getenv("DB_USER"),
#     password=os.getenv("DB_PASS"),
# )
# COLLECTION_NAME   = os.getenv("COLLECTION_NAME", "bank_data")
# DATABASE_LOCATION = os.getenv("DATABASE_LOCATION", "./chroma_db")
# EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
# CHAT_MODEL        = os.getenv("CHAT_MODEL", "llama3")
# BATCH_SIZE        = 500
# BOT_ICON          = "🤖"

# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# # ─────────────────────────────────────────────────────────────────
# # Metadata helpers
# # ─────────────────────────────────────────────────────────────────
# def get_balance_bucket(balance) -> str:
#     try:
#         b = float(balance)
#         if b < 0:       return "negative"
#         if b < 10000:   return "low"
#         if b < 100000:  return "medium"
#         if b < 1000000: return "high"
#         return "very_high"
#     except Exception:
#         return "unknown"

# def get_irregularity_status(val) -> str:
#     try:
#         return "irregular" if int(val) != 0 else "regular"
#     except Exception:
#         return "unknown"

# # ─────────────────────────────────────────────────────────────────
# # CSS
# # KEY SPACING FIX:
# #   .chat-body-wrap  → padding-bottom: 9rem  (pushes content above input)
# #   .last-msg-spacer → extra visible gap     (margin between last msg & input bar)
# #   [data-testid="stBottom"] → top padding  (breathing room above input box)
# # ─────────────────────────────────────────────────────────────────
# CUSTOM_CSS = """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&family=Sora:wght@400;600;700&display=swap');

# :root {
#     --bg0:#07090f; --bg1:#0d1117; --bg2:#111827; --bg3:#1a2235; --bg4:#1f2d42;
#     --blue:#3b82f6; --blue-d:#1d4ed8; --cyan:#22d3ee; --purple:#a78bfa;
#     --green:#34d399; --t1:#f1f5f9; --t2:#94a3b8; --t3:#64748b;
#     --border:rgba(255,255,255,0.07); --shadow:0 8px 32px rgba(0,0,0,0.5);
# }
# *,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
# html,body,.stApp {
#     background:var(--bg0) !important;
#     font-family:'DM Sans',sans-serif !important;
#     color:var(--t1) !important;
# }
# #MainMenu,footer,header,.stDeployButton { display:none !important; }
# .block-container { padding:0 !important; max-width:100% !important; }

# /* ── Sidebar ── */
# [data-testid="stSidebar"] {
#     background:var(--bg1) !important;
#     border-right:1px solid var(--border) !important;
# }
# [data-testid="stSidebar"] .block-container { padding:1.5rem 1.2rem !important; }
# .brand {
#     display:flex; align-items:center; gap:10px;
#     padding-bottom:1.2rem; border-bottom:1px solid var(--border); margin-bottom:1.4rem;
# }
# .brand-name {
#     font-family:'Sora',sans-serif; font-size: 48px; font-weight:700;
#     background:linear-gradient(90deg,var(--blue),var(--cyan));
#     -webkit-background-clip:text; -webkit-text-fill-color:transparent;
# }
# .brand-sub { font-size:.7rem; color:var(--t3); margin-top:1px; }
# .sb-label {
#     font-size:.7rem; font-weight:600; text-transform:uppercase;
#     letter-spacing:.1em; color:var(--t3); margin-bottom:.6rem;
# }
# .route-card {
#     background:var(--bg3); border:1px solid var(--border);
#     border-radius:10px; padding:8px 10px; margin-bottom:6px;
#     font-size:.78rem; color:var(--t2); line-height:1.5;
# }
# .route-card strong { color:var(--cyan); display:block; margin-bottom:2px; }
# .route-tag {
#     display:inline-block; background:rgba(52,211,153,.12);
#     color:var(--green); border-radius:4px; padding:1px 6px;
#     font-size:.68rem; font-weight:600; margin-top:4px;
# }
# .stButton>button {
#     background:var(--bg3) !important; color:var(--t1) !important;
#     border:1px solid var(--border) !important; border-radius:10px !important;
#     font-family:'DM Sans',sans-serif !important; font-size:.85rem !important;
#     font-weight:500 !important; padding:.55rem 1rem !important;
#     width:100% !important; transition:all .18s ease !important;
# }
# .stButton>button:hover {
#     background:var(--bg4) !important; border-color:var(--blue) !important;
#     transform:translateY(-1px) !important;
# }

# /* ── Chat header ── */
# .chat-header {
#     position:sticky; top:0; z-index:200;
#     display:flex; align-items:center; gap:12px;
#     padding:.85rem 1.8rem;
#     background:rgba(13,17,23,.96);
#     backdrop-filter:blur(20px);
#     border-bottom:1px solid var(--border);
#     min-height:64px;
# }
# .header-av {
#     width:40px; height:40px; border-radius:50%; flex-shrink:0;
#     background:linear-gradient(135deg,var(--blue),var(--purple));
#     display:flex; align-items:center; justify-content:center; font-size:20px;
#     box-shadow:0 0 18px rgba(59,130,246,.3);
# }
# .header-title { font-family:'Sora',sans-serif; font-size:.95rem; font-weight:600; }
# .header-sub {
#     font-size:.72rem; color:var(--green);
#     display:flex; align-items:center; gap:5px; margin-top:2px;
# }
# .live-dot {
#     width:6px; height:6px; border-radius:50%; background:var(--green); flex-shrink:0;
#     animation:livepulse 2s ease-in-out infinite;
# }
# @keyframes livepulse { 0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(.8);} }

# /* ─────────────────────────────────────────────
#    SPACING FIX — chat body
#    padding-bottom:9rem ensures the last message
#    is never hidden behind the sticky input bar.
# ───────────────────────────────────────────── */
# .chat-body-wrap {
#     padding:1rem 1.4rem 9rem 1.4rem;
# }

# /* Extra visible spacer rendered after last message */
# .last-msg-spacer {
#     height:1.5rem;          /* visual gap between last bubble & input area */
#     width:100%;
# }

# /* ── Message rows ── */
# .msg-row {
#     display:flex; align-items:flex-end; gap:8px;
#     animation:msgIn .25s ease both;
#     margin-bottom:.65rem;   /* gap between consecutive messages */
# }
# @keyframes msgIn { from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);} }
# .msg-row.user { flex-direction:row-reverse; }
# .msg-row.bot  { flex-direction:row; }

# .av {
#     width:30px; height:30px; border-radius:50%; flex-shrink:0;
#     display:flex; align-items:center; justify-content:center;
#     font-size:14px; margin-bottom:2px;
# }
# .av.u { background:linear-gradient(135deg,var(--blue),var(--blue-d)); }
# .av.b { background:var(--bg3); border:1px solid var(--border); margin-left:6px; }

# .bubble {
#     max-width:62%; padding:.75rem 1.05rem;
#     font-size:.875rem; line-height:1.68;
#     word-break:break-word; box-shadow:var(--shadow);
# }
# .bubble.user {
#     background:linear-gradient(135deg,var(--blue),var(--blue-d));
#     color:#fff; border-radius:18px 18px 4px 18px;
# }
# .bubble.bot {
#     background:var(--bg3); color:var(--t1);
#     border:1px solid var(--border); border-radius:18px 18px 18px 4px;
# }
# .bubble table {
#     width:100%; border-collapse:collapse; font-size:.8rem;
#     margin:8px 0; background:rgba(0,0,0,.3); border-radius:8px; overflow:hidden;
# }
# .bubble th {
#     background:rgba(59,130,246,.18); color:var(--cyan);
#     padding:6px 10px; text-align:left; font-weight:600;
#     border-bottom:1px solid var(--border); font-size:.75rem;
# }
# .bubble td { padding:5px 10px; border-bottom:1px solid rgba(255,255,255,.04); }
# .bubble tr:last-child td { border-bottom:none; }
# .bubble code {
#     background:rgba(34,211,238,.1); color:var(--cyan);
#     padding:1px 5px; border-radius:4px; font-size:.82em;
# }
# .bubble strong { color:var(--cyan); }
# .src-badge {
#     display:inline-block; margin-top:7px; font-size:.67rem; color:var(--t3);
#     background:rgba(255,255,255,.04); border:1px solid var(--border);
#     border-radius:4px; padding:1px 6px;
# }
# .rt-badge {
#     display:inline-block; font-size:.65rem; color:var(--t3);
#     background:rgba(52,211,153,.08); border:1px solid rgba(52,211,153,.2);
#     border-radius:4px; padding:1px 6px; margin-left:5px;
# }

# /* ── Typing dots ── */
# .typing-row {
#     display:flex; align-items:flex-end; gap:8px;
#     animation:msgIn .25s ease both; margin-bottom:.65rem;
# }
# .typing-bubble {
#     background:var(--bg3); border:1px solid var(--border);
#     border-radius:18px 18px 18px 4px;
#     padding:.75rem 1.1rem; display:flex; align-items:center; gap:5px;
# }
# .dot { width:7px; height:7px; border-radius:50%; animation:dotbounce 1.1s ease-in-out infinite; }
# .dot:nth-child(1) { background:var(--blue);   animation-delay:0s; }
# .dot:nth-child(2) { background:var(--cyan);   animation-delay:.18s; }
# .dot:nth-child(3) { background:var(--purple); animation-delay:.36s; }
# @keyframes dotbounce { 0%,80%,100%{transform:translateY(0);opacity:.5;} 40%{transform:translateY(-8px);opacity:1;} }

# /* ── Welcome ── */
# .welcome {
#     display:flex; flex-direction:column; align-items:center;
#     justify-content:center; padding:4rem 2rem; text-align:center;
# }
# .welcome-icon {
#     width:72px; height:72px; border-radius:22px;
#     background:linear-gradient(135deg,var(--blue),var(--purple));
#     display:flex; align-items:center; justify-content:center;
#     font-size:34px; margin-bottom:1.4rem;
#     box-shadow:0 12px 40px rgba(59,130,246,.35);
# }
# .welcome-title { font-family:'Sora',sans-serif; font-size:1.55rem; font-weight:700; margin-bottom:.5rem; }
# .welcome-sub { font-size:.88rem; color:var(--t2); max-width:380px; line-height:1.7; margin-bottom:1.5rem; }
# .chips-label { font-size:.72rem; color:var(--t3); margin-bottom:.8rem; text-transform:uppercase; letter-spacing:.08em; }
# .chip-btn>button {
#     background:var(--bg3) !important; border:1px solid var(--border) !important;
#     border-radius:20px !important; padding:6px 16px !important;
#     font-size:.78rem !important; color:var(--t2) !important;
#     font-weight:400 !important; transition:all .18s !important; width:auto !important;
# }
# .chip-btn>button:hover {
#     border-color:var(--blue) !important; color:var(--blue) !important;
#     background:rgba(59,130,246,.08) !important; transform:none !important;
# }

# /* ─────────────────────────────────────────────
#    INPUT AREA SPACING FIX
#    top padding creates visible gap between
#    the last message and the input box itself.
# ───────────────────────────────────────────── */
# [data-testid="stBottom"] {
#     background:var(--bg1) !important;
#     border-top:1px solid var(--border) !important;
#     padding:1rem 1.6rem 1.1rem 1.6rem !important;  /* top margin = gap above input */
# }
# .stChatInputContainer {
#     background:var(--bg1) !important;
#     padding:.8rem 1.4rem 1rem !important;
# }
# [data-testid="stChatInput"]>div {
#     background:var(--bg2) !important;
#     border:1px solid var(--border) !important;
#     border-radius:14px !important;
#     box-shadow:0 4px 24px rgba(0,0,0,.4) !important;
# }
# [data-testid="stChatInput"]>div:focus-within {
#     border-color:var(--blue) !important;
#     box-shadow:0 0 0 2px rgba(59,130,246,.15) !important;
# }
# [data-testid="stChatInput"] textarea {
#     background:transparent !important; color:var(--t1) !important;
#     font-family:'DM Sans',sans-serif !important; font-size:.9rem !important;
#     padding:.55rem .8rem !important;
# }
# [data-testid="stChatInput"] button {
#     background:var(--blue) !important; border-radius:9px !important; margin:4px !important;
# }

# ::-webkit-scrollbar { width:4px; }
# ::-webkit-scrollbar-thumb { background:var(--bg4); border-radius:8px; }
# </style>
# """

# TYPING_HTML = f"""
# <div class="typing-row" style="padding-left:1.4rem">
#   <div class="av b" style="margin-left:6px">{BOT_ICON}</div>
#   <div class="typing-bubble">
#     <div class="dot"></div><div class="dot"></div><div class="dot"></div>
#   </div>
# </div>
# """

# # ─────────────────────────────────────────────────────────────────
# # Session state
# # ─────────────────────────────────────────────────────────────────
# if "messages"     not in st.session_state: st.session_state.messages     = []
# if "chip_clicked" not in st.session_state: st.session_state.chip_clicked = None

# # ─────────────────────────────────────────────────────────────────
# # Cached resources — initialized once at startup
# # ─────────────────────────────────────────────────────────────────
# @st.cache_resource
# def load_vector_store():
#     embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
#     return Chroma(
#         collection_name=COLLECTION_NAME,
#         embedding_function=embeddings,
#         persist_directory=DATABASE_LOCATION,
#     )

# @st.cache_resource
# def load_llm():
#     return ChatOllama(
#         model=CHAT_MODEL,
#         temperature=0,
#         streaming=True,
#         num_predict=150,   # Lower = faster output
#         num_ctx=1024,      # Smaller context window = faster
#     )

# # Initialize at startup (cached — no repeated cost)
# vector_store = load_vector_store()
# llm = load_llm()

# # ─────────────────────────────────────────────────────────────────
# # DB helper
# # ─────────────────────────────────────────────────────────────────
# def get_conn():
#     return psycopg2.connect(
#         **DB_CFG,
#         connect_timeout=5,
#         options="-c statement_timeout=5000"
#     )

# # ─────────────────────────────────────────────────────────────────
# # INGESTION
# # BUG FIX: itertuples() returns namedtuple — use getattr(), not .get()
# # Fields: proc_date accountno acct_desc cust_name intrate odlimit
# #         drawing_power lmt_expry_dt currentbalance uncleared_bal
# #         irregularity new_irac old_irac sanction_dt arrear_cond
# #         branchno branch_name bankcd statecd
# # ─────────────────────────────────────────────────────────────────
# def compact_row_text(row) -> str:
#     return (
#         f"Account: {getattr(row, 'accountno', '')} | "
#         f"Customer: {getattr(row, 'cust_name', '')} | "
#         f"Branch: {getattr(row, 'branch_name', '')} (No: {getattr(row, 'branchno', '')}) | "
#         f"Balance: {float(getattr(row, 'currentbalance', 0)):.2f} | "
#         f"Rate: {getattr(row, 'intrate', 0)}% | "
#         f"Irregularity: {getattr(row, 'irregularity', 0)} | "
#         f"OD Limit: {getattr(row, 'odlimit', 0)} | "
#         f"Drawing Power: {getattr(row, 'drawing_power', 0)}"
#     )

# def run_ingestion(vector_store_inst):
#     existing = vector_store_inst.get()
#     if existing and len(existing.get("ids", [])) > 0:
#         st.warning(f"Vector store already has **{len(existing['ids']):,}** records. "
#                    "Change COLLECTION_NAME in .env to re-ingest.")
#         return

#     status   = st.empty()
#     progress = st.progress(0)
#     status.info("Connecting to database…")

#     try:
#         conn = get_conn()
#         df   = pd.read_sql("SELECT * FROM ccod_bal;", conn)
#         conn.close()
#     except Exception as e:
#         status.error(f"DB error: {e}")
#         return

#     df.columns = [c.lower() for c in df.columns]
#     total_rows = len(df)
#     status.info(f"Fetched {total_rows:,} rows. Building knowledge base…")
#     all_docs = []

#     # ── 1. Account docs — BUG FIX: itertuples + getattr ──
#     for row in df.itertuples(index=False):
#         bal       = float(getattr(row, "currentbalance", 0))
#         irreg_raw = getattr(row, "irregularity", 0)
#         all_docs.append(Document(
#             page_content=compact_row_text(row),
#             metadata={
#                 "doc_type":      "account",
#                 "account_no":    str(getattr(row, "accountno", "")),
#                 "branchno":      str(getattr(row, "branchno", "")),
#                 "branch":        str(getattr(row, "branch_name", "")).lower().strip(),
#                 "cust_name":     str(getattr(row, "cust_name", "")).lower().strip(),
#                 "intrate":       float(getattr(row, "intrate", 0) or 0),
#                 "status":        get_irregularity_status(irreg_raw),
#                 "irregularity":  str(irreg_raw),
#                 "balance_range": get_balance_bucket(bal),
#                 "bankcd":        str(getattr(row, "bankcd", "")),
#                 "statecd":       str(getattr(row, "statecd", "")),
#                 "source":        "ccod_bal",
#             }
#         ))
#     progress.progress(0.2)

#     # ── 2. Branch aggregate docs (precomputed) ──
#     branch_grp = df.groupby(["branch_name", "branchno"]).agg(
#         total_balance = ("currentbalance", "sum"),
#         avg_balance   = ("currentbalance", "mean"),
#         account_count = ("accountno",      "count"),
#         avg_intrate   = ("intrate",        "mean"),
#         max_balance   = ("currentbalance", "max"),
#         min_balance   = ("currentbalance", "min"),
#     ).reset_index()

#     for _, br in branch_grp.iterrows():
#         all_docs.append(Document(
#             page_content=(
#                 f"Branch: {br.branch_name} (No: {br.branchno}) | "
#                 f"Total Balance: {br.total_balance:.2f} | "
#                 f"Avg Balance: {br.avg_balance:.2f} | "
#                 f"Max: {br.max_balance:.2f} | Min: {br.min_balance:.2f} | "
#                 f"Avg Rate: {br.avg_intrate:.2f}% | Accounts: {int(br.account_count)}"
#             ),
#             metadata={
#                 "doc_type":      "branch_aggregate",
#                 "branch":        str(br.branch_name).lower().strip(),
#                 "branchno":      str(br.branchno),
#                 "total_balance": float(br.total_balance),
#                 "avg_balance":   float(br.avg_balance),
#                 "account_count": int(br.account_count),
#                 "avg_intrate":   float(br.avg_intrate),
#                 "source":        "precomputed",
#             }
#         ))
#     progress.progress(0.35)

#     # ── 3. Global summary (one doc, always limit=1 on fetch) ──
#     all_docs.append(Document(
#         page_content=(
#             f"Global Summary: Total Accounts: {len(df):,} | "
#             f"Total Balance: {df.currentbalance.sum():,.2f} | "
#             f"Avg Balance: {df.currentbalance.mean():,.2f} | "
#             f"Max Balance: {df.currentbalance.max():,.2f} | "
#             f"Min Balance: {df.currentbalance.min():,.2f} | "
#             f"Avg Interest Rate: {df.intrate.mean():.2f}%"
#         ),
#         metadata={"doc_type": "global_summary", "source": "precomputed"}
#     ))
#     progress.progress(0.4)

#     # ── 4. Irregularity per-branch docs ──
#     irreg_grp = df.groupby(["branch_name", "branchno", "irregularity"]).size().reset_index(name="count")
#     for _, ir in irreg_grp.iterrows():
#         all_docs.append(Document(
#             page_content=(
#                 f"Branch: {ir.branch_name} (No: {ir.branchno}) | "
#                 f"Irregularity: {ir.irregularity} | Count: {int(ir['count'])}"
#             ),
#             metadata={
#                 "doc_type":     "irregularity",
#                 "branch":       str(ir.branch_name).lower().strip(),
#                 "branchno":     str(ir.branchno),
#                 "irregularity": str(ir.irregularity),
#                 "source":       "precomputed",
#             }
#         ))
#     progress.progress(0.5)

#     # ── 5. Batch embed ──
#     total = len(all_docs)
#     status.info(f"Embedding {total:,} documents…")
#     for start in range(0, total, BATCH_SIZE):
#         batch = all_docs[start: start + BATCH_SIZE]
#         vector_store_inst.add_documents(documents=batch, ids=[str(uuid4()) for _ in batch])
#         progress.progress(0.5 + 0.5 * min(start + BATCH_SIZE, total) / total)

#     progress.progress(1.0)
#     status.success(f"✅ Done! {total:,} documents embedded ({total_rows:,} accounts + aggregates).")

# # ─────────────────────────────────────────────────────────────────
# # Query classifier
# # ─────────────────────────────────────────────────────────────────
# ACCOUNT_RE  = re.compile(r'\b(\d{5,})\b')
# BRANCHNO_RE = re.compile(r'\bbranch\s*(?:no|number|#)?\s*[:\-]?\s*(\d{2,6})\b', re.I)
# PURE_NUM_RE = re.compile(r'^\s*\d{3,6}\s*$')   # bare number like "301"

# AGGREGATE_KW = {
#     "highest", "lowest", "maximum", "minimum", "max", "min",
#     "average", "avg", "total", "sum", "count", "how many",
#     "most", "least", "top", "bottom", "all branches", "branch",
#     "overall", "across", "entire", "irregular", "irregularity",
#     "interest rate", "balance summary", "summary", "report",
# }
# GENERAL_RE = re.compile(
#     r'\b(hello|hi|hey|good\s*morning|good\s*evening|good\s*afternoon|'
#     r'how are you|what can you do|help|who are you|what is this|'
#     r'thanks|thank you|bye|goodbye|assist|capabilities)\b', re.I
# )

# def classify_query(q: str) -> str:
#     if ACCOUNT_RE.search(q):                          return "account"
#     if BRANCHNO_RE.search(q):                         return "branchno"
#     if PURE_NUM_RE.match(q):                          return "branchno"   # bare "301"
#     if any(kw in q.lower() for kw in AGGREGATE_KW):  return "aggregate"
#     if GENERAL_RE.search(q):                          return "general"
#     return "rag"

# # ─────────────────────────────────────────────────────────────────
# # Guard — check if vector DB has data
# # ─────────────────────────────────────────────────────────────────
# def db_has_data() -> bool:
#     try:
#         res = vector_store.get(limit=1)
#         return bool(res and res.get("ids"))
#     except Exception:
#         return False

# # ─────────────────────────────────────────────────────────────────
# # Handler 1 — Account lookup (metadata filter, <0.3s, no LLM)
# # ─────────────────────────────────────────────────────────────────
# def handle_account_query(query: str) -> str:
#     match = ACCOUNT_RE.search(query)
#     account_no = match.group(1) if match else None
#     logging.info(f"Route: account | account_no={account_no}")

#     if account_no:
#         try:
#             res = vector_store.get(where={"account_no": account_no}, limit=1)
#             if res and res.get("documents"):
#                 return (f"Account Details\n\n{res['documents'][0]}\n\n"
#                         f"<span class='src-badge'>⚡ Vector DB · exact filter</span>")
#         except Exception:
#             pass
#         docs = vector_store.similarity_search(f"Account number {account_no}", k=4)
#         for doc in docs:
#             if account_no in doc.page_content:
#                 return (f"**Account Details**\n\n{doc.page_content}\n\n"
#                         f"<span class='src-badge'>⚡ Vector DB · similarity</span>")
#     return f"No information found for account **{account_no}**."

# # ─────────────────────────────────────────────────────────────────
# # Handler 1b — Branch number lookup (metadata filter, no LLM)
# # ─────────────────────────────────────────────────────────────────
# def handle_branchno_query(query: str) -> str:
#     match   = BRANCHNO_RE.search(query)
#     pure    = PURE_NUM_RE.match(query)
#     branchno = (match.group(1) if match else query.strip()) if not match and pure else (match.group(1) if match else None)
#     logging.info(f"Route: branchno | branchno={branchno}")

#     if branchno:
#         try:
#             res = vector_store.get(where={"branchno": branchno, "doc_type": "branch_aggregate"}, limit=1)
#             if res and res.get("documents"):
#                 return (f"**Branch Details (No: {branchno})**\n\n{res['documents'][0]}\n\n"
#                         f"<span class='src-badge'>⚡ Vector DB · branchno filter</span>")
#         except Exception:
#             pass
#         docs = vector_store.similarity_search(f"branch number {branchno}", k=4)
#         for doc in docs:
#             if branchno in doc.page_content:
#                 return (f"**Branch Info**\n\n{doc.page_content}\n\n"
#                         f"<span class='src-badge'>⚡ Vector DB</span>")
#     return f"No branch found with branch number **{branchno}**."

# # ─────────────────────────────────────────────────────────────────
# # Handler 2 — Aggregate queries (precomputed docs, <1s, no LLM)
# # ─────────────────────────────────────────────────────────────────
# def handle_aggregate_query(query: str) -> str:
#     q = query.lower()
#     logging.info(f"Route: aggregate | query={query[:60]}")

#     # Global total / count
#     if any(x in q for x in ["total", "overall", "entire", "all account",
#                              "how many account", "how many customer", "global", "count"]):
#         try:
#             res = vector_store.get(where={"doc_type": "global_summary"}, limit=1)
#             if res and res.get("documents"):
#                 return (f"**Global Summary**\n\n{res['documents'][0]}\n\n"
#                         f"<span class='src-badge'>⚡ Precomputed · direct filter</span>")
#         except Exception:
#             pass

#     # Top branches
#     if "branch" in q and any(x in q for x in ["highest", "maximum", "most", "top", "best", "largest"]):
#         try:
#             res = vector_store.get(where={"doc_type": "branch_aggregate"}, limit=100)
#             docs_meta = list(zip(res.get("documents", []), res.get("metadatas", [])))
#         except Exception:
#             sim = vector_store.similarity_search("branch total balance", k=30)
#             docs_meta = [(d.page_content, d.metadata) for d in sim
#                          if d.metadata.get("doc_type") == "branch_aggregate"]
#         if docs_meta:
#             top = sorted(docs_meta, key=lambda x: x[1].get("total_balance", 0), reverse=True)[:5]
#             rows = "".join(
#                 f"| {m['branch'].title()} | {m.get('branchno','—')} | {m['total_balance']:,.2f} | {m['account_count']} |\n"
#                 for _, m in top
#             )
#             return (f"**Top 5 Branches by Total Balance**\n\n"
#                     f"| Branch | Branch No | Total Balance | Accounts |\n|---|---|---|---|\n{rows}\n"
#                     f"<span class='src-badge'>⚡ Precomputed · sorted in Python</span>")

#     # Bottom branches
#     if "branch" in q and any(x in q for x in ["lowest", "minimum", "least", "bottom", "smallest"]):
#         try:
#             res = vector_store.get(where={"doc_type": "branch_aggregate"}, limit=100)
#             docs_meta = list(zip(res.get("documents", []), res.get("metadatas", [])))
#         except Exception:
#             sim = vector_store.similarity_search("branch total balance", k=30)
#             docs_meta = [(d.page_content, d.metadata) for d in sim
#                          if d.metadata.get("doc_type") == "branch_aggregate"]
#         if docs_meta:
#             bottom = sorted(docs_meta, key=lambda x: x[1].get("total_balance", 0))[:5]
#             rows = "".join(
#                 f"| {m['branch'].title()} | {m.get('branchno','—')} | {m['total_balance']:,.2f} | {m['account_count']} |\n"
#                 for _, m in bottom
#             )
#             return (f"**Bottom 5 Branches by Total Balance**\n\n"
#                     f"| Branch | Branch No | Total Balance | Accounts |\n|---|---|---|---|\n{rows}\n"
#                     f"<span class='src-badge'>⚡ Precomputed · sorted in Python</span>")

#     # Average / interest rate
#     if any(x in q for x in ["average", "avg", "interest", "mean", "rate"]):
#         try:
#             res = vector_store.get(where={"doc_type": "global_summary"}, limit=1)
#             if res and res.get("documents"):
#                 return (f"**Averages**\n\n{res['documents'][0]}\n\n"
#                         f"<span class='src-badge'>⚡ Precomputed · direct filter</span>")
#         except Exception:
#             pass

#     # Irregularity
#     if any(x in q for x in ["irregular", "irregularity"]):
#         try:
#             res = vector_store.get(where={"doc_type": "irregularity"})
#             docs_list = res.get("documents", [])
#         except Exception:
#             sim = vector_store.similarity_search("irregularity accounts branch", k=10)
#             docs_list = [d.page_content for d in sim if d.metadata.get("doc_type") == "irregularity"]
#         if docs_list:
#             # Ensure only 1 global summary if duplicates
#             docs_list = docs_list[:10]
#             content = "\n".join(f"- {d}" for d in docs_list)
#             return (f"**Irregularity Summary**\n\n{content}\n\n"
#                     f"<span class='src-badge'>⚡ Precomputed · direct filter</span>")

#     # Generic branch
#     if "branch" in q:
#         docs = vector_store.similarity_search(query, k=5)
#         branch_docs = [d.page_content for d in docs if d.metadata.get("doc_type") == "branch_aggregate"][:3]
#         if branch_docs:
#             return f"**Branch Information**\n\n" + "\n\n".join(branch_docs) + \
#                    "\n\n<span class='src-badge'>⚡ Vector DB</span>"

#     # Fallback
#     docs = vector_store.similarity_search(query, k=3)
#     if docs:
#         return "\n\n".join(d.page_content for d in docs) + \
#                "\n\n<span class='src-badge'>📚 Vector DB · similarity</span>"
#     return "I couldn't find matching data for your query in the knowledge base."

# # ─────────────────────────────────────────────────────────────────
# # Handler 3 — General (LLM only, minimal context)
# # ─────────────────────────────────────────────────────────────────
# def stream_general(query: str, recent: list):
#     logging.info(f"Route: general | query={query[:60]}")
#     prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are NYX-AI, a friendly bank data assistant. "
#          "Respond warmly and briefly. You help with: account lookups, "
#          "branch analytics, balances, irregularities. "
#          "Answer in under 60 words. No repetition."),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{question}"),
#     ])
#     chain = (
#         {"question": RunnablePassthrough(), "chat_history": lambda _: recent}
#         | prompt | llm | StrOutputParser()
#     )
#     return chain.stream(query)

# # ─────────────────────────────────────────────────────────────────
# # Handler 4 — RAG (k=2, capped context, 1024 ctx)
# # ─────────────────────────────────────────────────────────────────
# def stream_rag(query: str, recent: list):
#     logging.info(f"Route: rag | query={query[:60]}")
#     retriever = vector_store.as_retriever(
#         search_type="similarity", search_kwargs={"k": 2}
#     )
#     prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a concise bank data assistant. "
#          "Use ONLY the context below. Answer in under 80 words. "
#          "Use bullet points if listing. No repetition.\n\nContext:\n{context}"),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{question}"),
#     ])
#     chain = (
#         {
#             # Cap total context to 1500 chars for speed
#             "context": retriever | (lambda docs: "\n".join(d.page_content for d in docs)[:1500]),
#             "question": RunnablePassthrough(),
#             "chat_history": lambda _: recent,
#         }
#         | prompt | llm | StrOutputParser()
#     )
#     return chain.stream(query)

# # ─────────────────────────────────────────────────────────────────
# # UI helpers
# # ─────────────────────────────────────────────────────────────────
# def user_bubble(text: str):
#     safe = _html.escape(text)
#     st.markdown(
#         f'<div class="msg-row user" style="padding-right:1.4rem">'
#         f'<div><div class="bubble user">{safe}</div></div>'
#         f'<div class=""></div></div>',
#         unsafe_allow_html=True
#     )

# def bot_bubble_static(content: str, elapsed: float = None):
#     badge = f"<span class='rt-badge'>⚡ {elapsed:.2f}s</span>" if elapsed else ""
#     col_icon, col_text, _ = st.columns([0.05, 0.63, 0.32])
#     with col_icon:
#         st.markdown(
#             f'<div class="" style="margin-top:4px;margin-left:6px"></div>', #chat iconnnn
#             unsafe_allow_html=True
#         )
#     with col_text:
#         st.markdown(
#             f'<div class="bubble bot" style="max-width:100%">{content}{badge}</div>',
#             unsafe_allow_html=True
#         )

# def bot_bubble_stream(stream_gen, start_time: float = None) -> str:
#     col_icon, col_text, _ = st.columns([0.05, 0.63, 0.32])
#     with col_icon:
#         st.markdown(
#             f'<div class="av b" style="margin-top:4px;margin-left:6px">{BOT_ICON}</div>',
#             unsafe_allow_html=True
#         )
#     with col_text:
#         ph   = st.empty()
#         full = ""
#         for chunk in stream_gen:
#             full += chunk
#             ph.markdown(full + "▌")
#         elapsed = time.time() - start_time if start_time else None
#         badge   = f"\n\n<span class='rt-badge'>🤖 {elapsed:.2f}s</span>" if elapsed else ""
#         ph.markdown(full + badge, unsafe_allow_html=True)
#     return full

# # ─────────────────────────────────────────────────────────────────
# # App Layout
# # ─────────────────────────────────────────────────────────────────
# st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# # ── Sidebar ──────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown(
#         '<div class="brand">'
#         '<div>'
#         '<div class="brand-name">NYX-AI</div>'
#         '<div class="brand-sub">Secure Financial Intelligence</div>'
#         '</div></div>',
#         unsafe_allow_html=True
#     )

#     st.markdown('<div class="sb-label">Data Sync</div>', unsafe_allow_html=True)
#     st.caption("Sync your database into the vector knowledge base once.")
#     if st.button("📥 Ingest Data from SQL", use_container_width=True):
#         run_ingestion(vector_store)

#     st.divider()
#     st.markdown('<div class="sb-label">Query Routing</div>', unsafe_allow_html=True)
#     st.markdown("""
#     <div class="route-card"><strong>🔢 Account / Branch No</strong>
#     Metadata filter lookup<span class="route-tag">⚡ &lt;0.3s · No LLM</span></div>
#     <div class="route-card"><strong>📊 Aggregates / Analytics</strong>
#     Precomputed docs<span class="route-tag">⚡ &lt;1s · No LLM</span></div>
#     <div class="route-card"><strong>🔍 Open-ended questions</strong>
#     Vector k=2 + LLM<span class="route-tag">🤖 1–3s</span></div>
#     <div class="route-card"><strong>💬 Greetings / General</strong>
#     LLM only<span class="route-tag">🤖 1–2s</span></div>
#     """, unsafe_allow_html=True)

#     st.divider()
#     if st.button("🗑️ Clear Chat", use_container_width=True):
#         st.session_state.messages     = []
#         st.session_state.chip_clicked = None
#         st.rerun()

# # ── Chat header ───────────────────────────────────────────────────
# st.markdown(f"""
# <div class="chat-header">
#   <div class="header-av">🤖</div>
#   <div>
#     <div class="header-title">Bank Data Assistant</div>
#     <div class="header-sub"><span class="live-dot"></span>Online</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

# # ── Welcome screen ────────────────────────────────────────────────
# CHIPS = [
#     "💰 What is the total balance?",
#     "🏢 Which branch has the highest balance?",
#     "📊 What is the average interest rate?",
#     "⚠️ Give me an irregularity summary",
#     "🔍 Show account 20300100000001 info",
#     "👋 How can you help me?",
# ]

# if not st.session_state.messages and st.session_state.chip_clicked is None:
#     st.markdown("""
#     <div class="welcome">
#       <div class="welcome-icon">🏦</div>
#       <div class="welcome-title">Bank Data Assistant</div>
#       <div class="welcome-sub">
#         Ask me anything about accounts, branches, balances, and irregularities.
#         Powered by a smart vector knowledge base for instant, context-aware answers.
#       </div>
#       <div class="chips-label">Try asking…</div>
#     </div>
#     """, unsafe_allow_html=True)

#     cols = st.columns(3)
#     for i, chip in enumerate(CHIPS):
#         with cols[i % 3]:
#             st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
#             if st.button(chip, key=f"chip_{i}", use_container_width=True):
#                 st.session_state.chip_clicked = chip
#                 st.rerun()
#             st.markdown('</div>', unsafe_allow_html=True)

# # ── Chat history ──────────────────────────────────────────────────
# else:
#     st.markdown('<div class="chat-body-wrap">', unsafe_allow_html=True)
#     for msg in st.session_state.messages:
#         if isinstance(msg, HumanMessage):
#             user_bubble(msg.content)
#         else:
#             bot_bubble_static(msg.content)
#     # Visible spacer between last message and input bar
#     st.markdown('<div class="last-msg-spacer"></div>', unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)

# # ── Chat input ────────────────────────────────────────────────────
# user_question = st.chat_input("Ask about accounts, balances, branches, irregularities…")

# if st.session_state.chip_clicked and not user_question:
#     user_question = st.session_state.chip_clicked
#     st.session_state.chip_clicked = None

# # ── Process ───────────────────────────────────────────────────────
# if user_question:

#     # Guard: warn if no data ingested yet
#     if not db_has_data():
#         st.warning("⚠️ No data found in knowledge base. Please click **Ingest Data from SQL** first.")
#         st.stop()

#     user_bubble(user_question)
#     st.session_state.messages.append(HumanMessage(user_question))

#     typing_ph = st.empty()
#     typing_ph.markdown(TYPING_HTML, unsafe_allow_html=True)

#     full_response = ""
#     t0 = time.time()

#     # BUG FIX: safe history slice regardless of message count
#     recent = st.session_state.messages[-5:] if len(st.session_state.messages) > 1 else []

#     try:
#         qtype = classify_query(user_question)

#         if qtype == "account":
#             typing_ph.empty()
#             full_response = handle_account_query(user_question)
#             bot_bubble_static(full_response, elapsed=time.time() - t0)

#         elif qtype == "branchno":
#             typing_ph.empty()
#             full_response = handle_branchno_query(user_question)
#             bot_bubble_static(full_response, elapsed=time.time() - t0)

#         elif qtype == "aggregate":
#             typing_ph.empty()
#             full_response = handle_aggregate_query(user_question)
#             bot_bubble_static(full_response, elapsed=time.time() - t0)

#         elif qtype == "general":
#             typing_ph.empty()
#             full_response = bot_bubble_stream(stream_general(user_question, recent), start_time=t0)

#         else:
#             typing_ph.empty()
#             full_response = bot_bubble_stream(stream_rag(user_question, recent), start_time=t0)

#     except Exception as e:
#         typing_ph.empty()
#         full_response = f"⚠️ Error: {str(e)}"
#         bot_bubble_static(full_response)
#         logging.error(f"Chat error: {e}", exc_info=True)

#     if full_response:
#         st.session_state.messages.append(AIMessage(full_response))