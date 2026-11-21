"""PwC India Advisory — Standard vs HyDE vs Graph RAG benchmark (LangChain + LangGraph)."""
import streamlit as st

from rag import config
from ui import tab1_chat, tab2_graph, tab3_workflow, tab4_telemetry, tab5_compare
from ui.state import init_state

st.set_page_config(page_title="PwC India · RAG benchmark", page_icon=":material/hub:", layout="wide")
init_state()
s = st.session_state

with st.sidebar:
    st.markdown("### :orange[PwC] India Advisory")
    st.caption("AI & Advanced Analytics · RAG benchmark")
    st.markdown("**Practice area**")
    st.segmented_control("Practice area", list(config.PRACTICE_AREAS), key="area", default="tax", required=True,
                         format_func=lambda k: {"tax": ":material/receipt_long: Tax", "grc": ":material/shield: GRC",
                                                "deals": ":material/handshake: Deals"}[k],
                         label_visibility="collapsed", selection_mode="single")
    st.markdown("**RAG engine**")
    st.segmented_control("RAG engine", list(config.ENGINES), key="engine", default="graph", required=True,
                         format_func=lambda k: config.ENGINES[k], label_visibility="collapsed")
    st.caption(config.PRACTICE_AREAS[s.area])
    st.divider()
    st.markdown("**System status**")
    with st.container(horizontal=True):
        st.badge("LLM ready" if config.has_api_key() else "No API key", icon=":material/key:",
                 color="green" if config.has_api_key() else "red")
        st.badge("Indexed" if s.kb else "Not indexed", icon=":material/database:", color="green" if s.kb else "gray")
    st.caption(f"Model: `{config.MODEL_NAME}` via OpenRouter")
    st.caption(f"Embeddings: `{config.EMBED_MODEL}`")

st.title("Comparative RAG benchmark")
st.caption("Standard RAG vs HyDE RAG vs Graph RAG — domain-grounded QA for Tax, Risk Advisory and Deal Services")

TABS = [":material/chat: Ingestion & chat", ":material/bubble_chart: Knowledge graph", ":material/account_tree: Workflow",
        ":material/tune: Hyperparameters & KPIs", ":material/compare_arrows: Benchmark"]
t1, t2, t3, t4, t5 = st.tabs(TABS, on_change="rerun", key="main_tabs")
for tab, module in zip((t1, t2, t3, t4, t5), (tab1_chat, tab2_graph, tab3_workflow, tab4_telemetry, tab5_compare)):
    with tab:
        if tab.open:
            module.render()
