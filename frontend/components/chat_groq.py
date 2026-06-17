"""
chat_groq.py :  Groq-powered conversational analytics for blinkBOT.
Decoupled Version: Streamlit UI only. Defers all LLM calls and code execution to FastAPI.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import streamlit as st

from components.api_client import ask_chat

_CHAT_HISTORY_PATH = Path("../data/chat_history.json")

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR TOKENS  
# ─────────────────────────────────────────────────────────────────────────────
_YELLOW      = "#F8D000"
_YELLOW_SOFT = "#FEF3C7"
_DARK        = "#1A1A1A"
_MUTED       = "#6B6B6B"
_BORDER      = "#E8E8E0"

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE — SAVE / LOAD / DELETE
# ─────────────────────────────────────────────────────────────────────────────
def save_chat(history: list) -> None:
    """Append the active session history to data/chat_history.json."""
    ts = datetime.now(timezone.utc).isoformat()
    
    # Read existing history first
    existing_records = []
    if _CHAT_HISTORY_PATH.exists():
        try:
            existing_records = json.loads(_CHAT_HISTORY_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    records: list[dict] = []
    for entry in history:
        records.append({
            "role": "user",
            "content": entry.get("question", ""),
            "code_executed": None,
            "timestamp": ts,
        })
        records.append({
            "role": "assistant",
            "content": entry.get("summary") or entry.get("error") or "",
            "code_executed": entry.get("code_executed"),
            "timestamp": ts,
        })
        
    existing_records.extend(records)
    _CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CHAT_HISTORY_PATH.write_text(json.dumps(existing_records, indent=2, ensure_ascii=False))


def load_chat() -> list[dict]:
    """
    Read data/chat_history.json. 
    In this decoupled architecture, we don't re-execute code on the frontend.
    We just load the text summary.
    """
    if not _CHAT_HISTORY_PATH.exists():
        return []
    try:
        records: list[dict] = json.loads(_CHAT_HISTORY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    entries: list[dict] = []
    i = 0
    while i < len(records) - 1:
        user_rec = records[i]
        asst_rec = records[i + 1]
        if user_rec.get("role") == "user" and asst_rec.get("role") == "assistant":
            entry: dict = {
                "question":      user_rec.get("content", ""),
                "summary":       asst_rec.get("content", ""),
                "result_df":     None,
                "fig":           None,
                "error":         None,
                "code_executed": asst_rec.get("code_executed"),
            }
            entries.append(entry)
            i += 2
        else:
            i += 1
    return entries


def _delete_entry_from_file(pair_index: int) -> None:
    if not _CHAT_HISTORY_PATH.exists():
        return
    try:
        records: list[dict] = json.loads(_CHAT_HISTORY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return
    start = pair_index * 2
    del records[start : start + 2]
    _CHAT_HISTORY_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False))


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT CHAT PAGE
# ─────────────────────────────────────────────────────────────────────────────
def show_chat(df: pd.DataFrame):
    # Note: df is passed for backwards compatibility but we no longer need it 
    # since data manipulation is on the backend.
    
    st.markdown(f"""
        <div style="padding: 28px 0 8px;">
            <div style="font-size:0.7rem; font-weight:800; letter-spacing:0.14em;
                        text-transform:uppercase; color:{_MUTED}; margin-bottom:6px;">
                BLINKIT
            </div>
            <h1 style="margin:0; font-size:2rem; font-weight:900;
                       letter-spacing:-0.03em; color:{_DARK};">
                blink<span style="color:{_YELLOW};">BOT</span>
            </h1>
            <p style="margin:6px 0 0; font-size:0.88rem; color:{_MUTED};">
                Hello there! You can ask anything about the Blinkit dataset, get instant analysis,
                charts, and plain-English summaries.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<hr style="border:none;border-top:1px solid {_BORDER};margin:16px 0 24px;">',
        unsafe_allow_html=True,
    )

    if "chat_history"    not in st.session_state:
        st.session_state.chat_history    = []
    if "show_older_chat" not in st.session_state:
        st.session_state.show_older_chat = False
    if "older_chat"      not in st.session_state:
        st.session_state.older_chat      = []

    if st.session_state.show_older_chat:
        hdr_col, close_col = st.columns([18, 2])
        with hdr_col:
            st.markdown(f"""
                <div style="font-size:0.7rem; font-weight:800; letter-spacing:0.12em;
                            text-transform:uppercase; color:{_MUTED}; margin-bottom:12px;">
                    Saved Chat History
                </div>
            """, unsafe_allow_html=True)
        with close_col:
            if st.button("Close", key="close_older_chat", use_container_width=True):
                st.session_state.show_older_chat = False
                st.rerun()

        older = st.session_state.older_chat
        if not older:
            st.info("No saved chat history found in `data/chat_history.json`.")
        else:
            for i, entry in enumerate(older):
                msg_col, del_col = st.columns([22, 1])
                with msg_col:
                    with st.chat_message("user", avatar="🟡"):
                        st.markdown(entry["question"])
                    with st.chat_message("assistant", avatar="🛵"):
                        _render_response(entry, chart_key=f"old_{i}")
                with del_col:
                    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
                    if st.button("🗑️", key=f"del_older_{i}", help="Delete this exchange"):
                        _delete_entry_from_file(i)
                        st.session_state.older_chat = load_chat()
                        st.rerun()

        st.markdown(
            f'<hr style="border:none;border-top:1px solid {_BORDER};margin:20px 0 16px;">',
            unsafe_allow_html=True,
        )
        st.markdown(f"""
            <div style="font-size:0.7rem; font-weight:800; letter-spacing:0.12em;
                        text-transform:uppercase; color:{_MUTED}; margin-bottom:12px;">
                Current Session
            </div>
        """, unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown(f"""
            <div style="background:{_YELLOW_SOFT}; border-left:3px solid {_YELLOW};
                        border-radius:0 8px 8px 0; padding:14px 18px;
                        margin-bottom:20px; font-size:0.86rem; color:#78350F;">
                <strong>Try asking:</strong><br>
                &bull; Which product category generates the highest average profit margin?<br>
                &bull; Show me the top 10 best-selling products by sold quantity.<br>
                &bull; Plot sales by city as a bar chart.<br>
                &bull; What percentage of orders were delayed, broken down by seller?
            </div>
        """, unsafe_allow_html=True)

    for i, entry in enumerate(st.session_state.chat_history):
        with st.chat_message("user", avatar="🟡"):
            st.markdown(entry["question"])
        with st.chat_message("assistant", avatar="🛵"):
            _render_response(entry, chart_key=f"cur_{i}")

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    btn_save, btn_view, btn_clear, _ = st.columns([1.6, 1.9, 1.3, 4.5])

    with btn_save:
        if st.button("Save Chat", use_container_width=True, disabled=not st.session_state.chat_history):
            try:
                save_chat(st.session_state.chat_history)
                st.session_state.chat_history = []  # Clear after saving
                if st.session_state.show_older_chat:
                    st.session_state.older_chat = load_chat()
                st.session_state._save_ok = True
            except Exception as exc:
                st.session_state._save_err = str(exc)
            st.rerun()

    if st.session_state.pop("_save_ok", False):
        st.success("Saved successfully", icon=None)
    if "_save_err" in st.session_state:
        st.error(f"Save failed: {st.session_state.pop('_save_err')}")

    with btn_view:
        label = "Hide Older Chat" if st.session_state.show_older_chat else "View Older Chat"
        if st.button(label, use_container_width=True):
            st.session_state.show_older_chat = not st.session_state.show_older_chat
            if st.session_state.show_older_chat:
                st.session_state.older_chat = load_chat()
            st.rerun()

    with btn_clear:
        if st.button("Clear", use_container_width=True, disabled=not st.session_state.chat_history):
            st.session_state.chat_history = []
            st.rerun()

    prompt = st.chat_input("Ask a question about the Blinkit dataset…")
    if prompt:
        with st.chat_message("user", avatar="🟡"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🛵"):
            status = st.empty()
            status.markdown("🛵 &nbsp;*delivering your response in < 10s...*")
            result = ask_chat(prompt)
            status.empty()
            _render_response(result, chart_key=f"cur_{len(st.session_state.chat_history)}")

        st.session_state.chat_history.append({"question": prompt, **result})
        st.rerun()


def _render_response(entry: dict, chart_key: str = ""):
    if entry.get("error"):
        st.error(f"Sorry, I couldn't complete that analysis.\n\n_{entry['error']}_")
        return

    if entry.get("summary"):
        st.markdown(entry["summary"])

    if entry.get("result_df") is not None and not entry["result_df"].empty:
        st.dataframe(entry["result_df"], width='stretch', hide_index=True)

    if entry.get("fig") is not None:
        st.plotly_chart(entry["fig"], width='stretch', key=chart_key or None)
