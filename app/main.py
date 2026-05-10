"""
app/main.py
─────────────
Streamlit entry point for DE-AI Assistant.

Layout:
  ┌─────────────────────────────────────────────┐
  │  Sidebar: Pipeline Health + Catalogue Search │
  │  Main: Chat interface with tool indicators   │
  └─────────────────────────────────────────────┘

Run: streamlit run app/main.py
"""

import streamlit as st
import sys
from pathlib import Path

# Ensure project root is on the path when running via streamlit
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.agents.orchestrator import Orchestrator
from app.agents.tools.health_monitor import get_pipeline_status
from app.ui.styles import get_styles
from app.ui.components import (
    render_health_card,
    render_quality_report,
    render_source_citations,
    render_tool_indicator,
    render_lineage_graph,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=settings.app_title,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(get_styles(), unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_results" not in st.session_state:
    st.session_state.tool_results = []

orch: Orchestrator = st.session_state.orchestrator

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <h2 style="color:#818cf8; margin-bottom:4px;">🤖 DE-AI Assistant</h2>
        <p style="color:#64748b; font-size:0.85rem;">Powered by llama-3.1-8b-instant</p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Pipeline health mini-dashboard
    st.markdown("### 📊 Pipeline Health")
    try:
        runs = get_pipeline_status()
        for run in runs[:5]:  # Show top 5
            render_health_card(run)
    except Exception as e:
        st.warning(f"Health data unavailable: {e}")

    st.divider()

    # Quick search
    st.markdown("### 🔍 Quick Search")
    search_term = st.text_input("Search catalogue...", key="sidebar_search")
    if search_term:
        from app.agents.tools.catalog_explorer import search_tables
        results = search_tables(search_term)
        for t in results[:3]:
            st.markdown(
                f"**{t['name']}** — *{t.get('owner', '?')}*\n\n{t.get('description', '')[:80]}..."
            )

    st.divider()

    # Session controls
    if st.button("🔄 New Conversation", use_container_width=True):
        orch.reset()
        st.session_state.messages = []
        st.session_state.tool_results = []
        st.rerun()

# ── Main chat area ─────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#e2e8f0;'>💬 Data Engineering Assistant</h1>",
    unsafe_allow_html=True,
)

# Example prompts
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(4)
    prompts = [
        "What is the retry strategy for Bronze ingestion?",
        "Which tables contain PII data?",
        "Show me recent pipeline failures",
        "Run a quality check on the orders table",
    ]
    for col, prompt in zip(cols, prompts):
        with col:
            if st.button(prompt, key=f"prompt_{prompt[:20]}", use_container_width=True):
                st.session_state._pending_prompt = prompt

# Display message history
for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    content = msg["content"]
    with st.chat_message(role):
        st.markdown(content)

        # Show tool indicators & rich results for assistant messages
        if role == "assistant" and i < len(st.session_state.tool_results):
            tr = st.session_state.tool_results[i]
            render_tool_indicator(tr.get("tool_used"))
            render_source_citations(tr.get("sources", []))

            # Quality report rendering
            result = tr.get("tool_result")
            if tr.get("tool_used") == "run_quality_check" and isinstance(result, dict):
                render_quality_report(result)

            # Lineage rendering
            if tr.get("tool_used") == "get_lineage" and isinstance(result, dict):
                table_name = "table"
                render_lineage_graph(table_name, result)

# ── Chat input ─────────────────────────────────────────────────────────────────
# Handle example prompt clicks
pending = st.session_state.pop("_pending_prompt", None)
user_input = st.chat_input("Ask about your data pipelines...") or pending

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get orchestrator response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = orch.chat(user_input)

        response = result["response"]
        st.markdown(response)
        render_tool_indicator(result.get("tool_used"))
        render_source_citations(result.get("sources", []))

        tool_result = result.get("tool_result")
        if result.get("tool_used") == "run_quality_check" and isinstance(tool_result, dict):
            render_quality_report(tool_result)
        if result.get("tool_used") == "get_lineage" and isinstance(tool_result, dict):
            render_lineage_graph("table", tool_result)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.tool_results.append(result)
