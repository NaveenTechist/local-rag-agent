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
BATCH_SIZE        = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ─────────────────────────────────────────────────────────────────
# Session state — must be first, before anything else
# ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# ─────────────────────────────────────────────────────────────────
# Cached Resources
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
    return ChatOllama(model=CHAT_MODEL, temperature=0, streaming=True)

# ─────────────────────────────────────────────────────────────────
# DB helper
# ─────────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(**DB_CFG)

# ─────────────────────────────────────────────────────────────────
# Route 1 — Exact account number → SQL lookup
# ─────────────────────────────────────────────────────────────────
def exact_sql_lookup(query: str):
    match = re.search(r'\b(\d{5,})\b', query)
    if not match:
        return None

    account_no = match.group(1)
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """SELECT accountno, cust_name, branch_name,
                      currentbalance, intrate, irregularity
               FROM ccod_bal WHERE accountno = %s LIMIT 1;""",
            (account_no,)
        )
        row = cur.fetchone()
        conn.close()

        if row:
            accountno, name, branch, balance, rate, irreg = row
            return (
                f"**Account Details**\n\n"
                f"| Field | Value |\n|---|---|\n"
                f"| Account No | {accountno} |\n"
                f"| Customer | {name} |\n"
                f"| Branch | {branch} |\n"
                f"| Balance | {balance:,.2f} |\n"
                f"| Interest Rate | {rate}% |\n"
                f"| Irregularity | {irreg} |\n\n"
                f"*Source: ccod_bal (live SQL)*"
            )
        return f"No account found with number **{account_no}**."

    except Exception as e:
        logging.error(f"SQL lookup error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────
# Route 2 — Aggregate questions → SQL aggregation
# Handles: highest, lowest, average, total, count, how many, etc.
# ─────────────────────────────────────────────────────────────────
AGGREGATE_KEYWORDS = [
    "highest", "lowest", "maximum", "minimum", "average", "avg",
    "total", "sum", "count", "how many", "most", "least",
    "top", "bottom", "rank", "all branches", "all accounts",
    "overall", "across", "entire"
]

def is_aggregate_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in AGGREGATE_KEYWORDS)

def aggregate_sql_query(query: str):
    """
    Detect the intent and run the appropriate SQL aggregation.
    Returns formatted string or None if pattern not matched.
    """
    q = query.lower()
    try:
        conn = get_conn()
        cur = conn.cursor()

        # Which branch has highest/lowest total balance?
        if "branch" in q and ("highest" in q or "maximum" in q or "most" in q or "top" in q):
            cur.execute("""
                SELECT branch_name, SUM(currentbalance) as total_balance
                FROM ccod_bal
                GROUP BY branch_name
                ORDER BY total_balance DESC
                LIMIT 5;
            """)
            rows = cur.fetchall()
            conn.close()
            if rows:
                table = "| Branch | Total Balance |\n|---|---|\n"
                table += "\n".join(f"| {r[0]} | {r[1]:,.2f} |" for r in rows)
                return f"**Top 5 Branches by Total Balance** *(SQL Aggregation)*\n\n{table}\n\n*Source: ccod_bal*"

        elif "branch" in q and ("lowest" in q or "minimum" in q or "least" in q or "bottom" in q):
            cur.execute("""
                SELECT branch_name, SUM(currentbalance) as total_balance
                FROM ccod_bal
                GROUP BY branch_name
                ORDER BY total_balance ASC
                LIMIT 5;
            """)
            rows = cur.fetchall()
            conn.close()
            if rows:
                table = "| Branch | Total Balance |\n|---|---|\n"
                table += "\n".join(f"| {r[0]} | {r[1]:,.2f} |" for r in rows)
                return f"**Bottom 5 Branches by Total Balance** *(SQL Aggregation)*\n\n{table}\n\n*Source: ccod_bal*"

        # Average balance
        elif "average" in q or "avg" in q:
            cur.execute("SELECT AVG(currentbalance) FROM ccod_bal;")
            avg = cur.fetchone()[0]
            conn.close()
            return f"**Average Balance across all accounts:** `{avg:,.2f}`\n\n*Source: ccod_bal*"

        # Total balance
        elif "total" in q and "balance" in q:
            cur.execute("SELECT SUM(currentbalance) FROM ccod_bal;")
            total = cur.fetchone()[0]
            conn.close()
            return f"**Total Balance across all accounts:** `{total:,.2f}`\n\n*Source: ccod_bal*"

        # Count of accounts
        elif "how many" in q and ("account" in q or "customer" in q):
            cur.execute("SELECT COUNT(*) FROM ccod_bal;")
            count = cur.fetchone()[0]
            conn.close()
            return f"**Total number of accounts:** `{count:,}`\n\n*Source: ccod_bal*"

        # Irregularity stats
        elif "irregular" in q:
            cur.execute("""
                SELECT irregularity, COUNT(*) as count
                FROM ccod_bal
                GROUP BY irregularity
                ORDER BY count DESC;
            """)
            rows = cur.fetchall()
            conn.close()
            if rows:
                table = "| Irregularity Status | Count |\n|---|---|\n"
                table += "\n".join(f"| {r[0]} | {r[1]:,} |" for r in rows)
                return f"**Irregularity Summary** *(SQL Aggregation)*\n\n{table}\n\n*Source: ccod_bal*"

        conn.close()
        return None

    except Exception as e:
        logging.error(f"Aggregate SQL error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────
# Route 3 — RAG chain (vector search + LLM)
# chat_history captured as plain list — never accessed via lambda
# from background thread (that was the AttributeError root cause)
# ─────────────────────────────────────────────────────────────────
def build_rag_chain(vector_store, llm, chat_history: list):
    # k=10 gives better semantic coverage for general questions
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful bank data assistant.

Use the context below to answer the question.
- If the context contains relevant data, use it and mention the source.
- If the context is partially relevant, use what you can and state any limitations.
- Only say "I don't know" if the context has absolutely no relevant information.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        if not docs:
            return "No relevant documents found in the vector store."
        return "\n\n".join(
            f"[Row {d.metadata.get('row_index', '?')}]: {d.page_content}"
            for d in docs
        )

    # Capture chat_history as a plain Python list RIGHT NOW on the main thread
    # Never reference st.session_state inside the chain — it runs in a background thread
    captured_history = list(chat_history)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: captured_history,
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever  # also return retriever for debug mode

# ─────────────────────────────────────────────────────────────────
# Row → Text
# ─────────────────────────────────────────────────────────────────
def row_to_text(row):
    return (
        f"Account {row.accountno} belonging to {row.cust_name} "
        f"at branch {row.branch_name} has current balance {row.currentbalance} "
        f"with interest rate {row.intrate} percent. "
        f"Irregularity status is {row.irregularity}."
    )

# ─────────────────────────────────────────────────────────────────
# Ingestion — batched, no splitter, duplicate-safe
# ─────────────────────────────────────────────────────────────────
def run_ingestion(vector_store):
    status = st.empty()

    existing = vector_store.get()
    if existing and len(existing.get("ids", [])) > 0:
        count = len(existing["ids"])
        st.warning(
            f"⚠️ Vector store already has **{count:,}** chunks. "
            "Change COLLECTION_NAME in .env to re-ingest fresh data."
        )
        return

    status.info("Connecting to database...")
    try:
        conn = get_conn()
        df = pd.read_sql("SELECT * FROM ccod_bal;", conn)
        conn.close()
    except Exception as e:
        status.error(f"DB connection failed: {e}")
        return

    status.info(f"Fetched {len(df):,} rows. Building documents...")

    docs = [
        Document(
            page_content=row_to_text(row),
            metadata={"source": "ccod_bal", "row_index": i + 1}
        )
        for i, row in df.iterrows()
    ]

    total = len(docs)
    progress_bar = st.progress(0)
    status.info(f"Embedding {total:,} documents in batches of {BATCH_SIZE}...")

    for start in range(0, total, BATCH_SIZE):
        batch = docs[start: start + BATCH_SIZE]
        uuids = [str(uuid4()) for _ in batch]
        vector_store.add_documents(documents=batch, ids=uuids)
        done = min(start + BATCH_SIZE, total)
        progress_bar.progress(done / total)

    status.success(f"✅ Done! {total:,} rows embedded and stored.")

# ─────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bank RAG Chatbot", page_icon="🏦", layout="wide")
st.title("🏦 Bank Data Assistant")

with st.sidebar:
    st.title("NYX-AI")
    st.caption("")
    # st.markdown(f"**LLM:** `{CHAT_MODEL}`")
    # st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}`")

    st.divider()
    st.subheader("🔄 Sync Data")
    st.caption("Run once to changes your data.")
    if st.button("📥 Ingest Data from SQL", use_container_width=True):
        vs = load_vector_store()
        run_ingestion(vs)

    # st.divider()
    # st.subheader("🔍 Query Routing")
    # st.info(
    #     "**Account number (5+ digits)?**\n→ Direct SQL (instant)\n\n"
    #     "**Aggregate words (highest, total, avg)?**\n→ SQL aggregation (instant)\n\n"
    #     "**Everything else?**\n→ Vector search + LLM"
    # )

    # st.divider()
    # st.subheader("🐛 Debug Mode")
    # st.session_state.debug_mode = st.toggle(
    #     "Show retrieved documents", value=st.session_state.debug_mode
    # )

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Display history ──
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ── Chat input ──
if user_question := st.chat_input("Ask about accounts, balances, branches, irregularities..."):

    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append(HumanMessage(user_question))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        # ── Route 1: Exact account number → SQL ──
        exact = exact_sql_lookup(user_question)
        if exact:
            full_response = exact
            placeholder.markdown(full_response)

        # ── Route 2: Aggregate question → SQL aggregation ──
        elif is_aggregate_query(user_question):
            agg_result = aggregate_sql_query(user_question)
            if agg_result:
                full_response = agg_result
                placeholder.markdown(full_response)
            else:
                # Aggregation pattern not matched → fall through to RAG
                vector_store = load_vector_store()
                llm = load_llm()
                chain, retriever = build_rag_chain(
                    vector_store, llm, st.session_state.messages
                )
                if st.session_state.debug_mode:
                    with st.expander("🔍 Retrieved Documents"):
                        docs = retriever.get_relevant_documents(user_question)
                        for d in docs:
                            st.write(f"Row {d.metadata.get('row_index')}: {d.page_content}")
                for chunk in chain.stream(user_question):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

        # ── Route 3: General question → Vector search + LLM ──
        else:
            vector_store = load_vector_store()
            llm = load_llm()
            chain, retriever = build_rag_chain(
                vector_store, llm, st.session_state.messages
            )
            if st.session_state.debug_mode:
                with st.expander("🔍 Retrieved Documents"):
                    docs = retriever.get_relevant_documents(user_question)
                    for d in docs:
                        st.write(f"Row {d.metadata.get('row_index')}: {d.page_content}")
            for chunk in chain.stream(user_question):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

    st.session_state.messages.append(AIMessage(full_response))