# LIBRARIES
import streamlit as st
import sys
import os

# let Python find rag.py inside the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from rag import rag_answer, OLLAMA_MODEL  # noqa: E402
############################################################################################################

# PAGE SETUP
# title shown on the browser tab
st.set_page_config(page_title="ImmiBrain", page_icon="🎓", layout="wide")

# title and short description shown at the top of the page
st.title("🎓 ImmiBrain")
st.caption(
    "A RAG prototype answering visa questions grounded in official "
    "visa PDFs. Answers include source citations. Always verify against "
    "official Home Affairs guidance for real decisions."
)

############################################################################################################
# SIDEBAR
with st.sidebar:
    st.header("Settings")
    # k = number of chunks sent to the language model (default 6)
    k = st.slider("Number of chunks retrieved", min_value=2, max_value=8, value=6)
    st.markdown(f"**Model:** `{OLLAMA_MODEL}` (via Ollama, local)")
    st.divider()
    st.markdown(
        "**Disclaimer:** This is a university prototype, not immigration "
        "advice. Always confirm with an official source or a registered "
        "migration agent."
    )

############################################################################################################
# CHAT HISTORY
# Streamlit re-runs this whole script on every action,
# so the history is saved in session_state to keep it during the session
if "history" not in st.session_state:
    st.session_state.history = []

# show all the earlier messages (and their sources) again
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            with st.expander("Sources"):
                for s in turn["sources"]:
                    st.markdown(f"- **{s['source']}**, page {s['page']} (relevance: {s['score']})")

############################################################################################################
# NEW QUESTION
# text box at the bottom of the page
query = st.chat_input("Ask about student visa conditions, work rights, extensions...")

if query:
    # save and show the user's question
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching visa documents..."):
            try:
                # get the answer and sources from the RAG pipeline in rag.py
                result = rag_answer(query, k=k)
                st.markdown(result["answer"])
                # show the chunks that were retrieved
                if result["sources"]:
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.markdown(f"- **{s['source']}**, page {s['page']} (relevance: {s['score']})")
                # save the answer so it stays on the page
                st.session_state.history.append(
                    {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                )
            except Exception as e:
                # shown if Ollama is not running or the database was not created
                err = f"Error: {e}\n\nMake sure Ollama is running and you've run `python data_ingestion.py` first."
                st.error(err)
                st.session_state.history.append({"role": "assistant", "content": err})