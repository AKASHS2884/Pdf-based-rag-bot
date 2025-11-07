#app.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import time
import os
import json

st.set_page_config(
    page_title="FAQ Assistant",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# -----------------------------
# Auth helpers (static bearer)
# -----------------------------
def get_auth_headers() -> Optional[Dict[str, str]]:
    token = st.session_state.get("auth_token", "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return None


def is_authenticated() -> bool:
    return bool(st.session_state.get("auth_token", "").strip())


def verify_token(token: str) -> Dict[str, Any]:
    try:
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(f"{BACKEND_URL}/auth/verify", headers=headers, timeout=10)
        if r.status_code == 200:
            return {"success": True, "data": r.json()}
        else:
            try:
                return {"success": False, "error": r.json().get("detail", f"HTTP {r.status_code}")}
            except:
                return {"success": False, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# -----------------------------
# Sample iPhone 17 FAQ CSV
# -----------------------------
@st.cache_data
def generate_sample_iphone17_faq_csv() -> bytes:
    data = [
        {"question": "Why should I buy iPhone 17?",
         "answer": "iPhone 17 offers the A19 chip for exceptional performance and efficiency. Larger sensors and improved processing deliver stunning photos and videos. Ceramic Shield and IP68 add durability. With USB‑C, MagSafe, and all‑day battery life, it's a reliable, modern companion for daily life."},
        {"question": "Does iPhone 17 support USB‑C?",
         "answer": "Yes, iPhone 17 and iPhone 17 Pro feature USB‑C with Power Delivery for fast charging and high‑speed data transfer. The universal connector simplifies travel and everyday use."},
        {"question": "Is MagSafe available on iPhone 17?",
         "answer": "Yes, iPhone 17 supports MagSafe accessories for snap‑on chargers, wallets, and mounts, ensuring secure attachment and optimal charging alignment."},
        {"question": "What's new in the iPhone 17 camera system?",
         "answer": "iPhone 17 features larger sensors, improved image processing, and Pro models support ProRAW/ProRes. Low‑light performance and noise reduction are enhanced for professional‑quality results."},
        {"question": "Does iPhone 17 Pro have a 120Hz display?",
         "answer": "Yes, Pro models include ProMotion up to 120Hz for smooth scrolling and responsive touch interactions."},
        {"question": "How is battery life on iPhone 17?",
         "answer": "iPhone 17 provides all‑day battery life under typical use; Pro models handle extended capture and playback with optimized power efficiency."},
        {"question": "Is iPhone 17 durable?",
         "answer": "Yes, iPhone 17 has a Ceramic Shield front cover and IP68 water resistance for everyday spills and dust."},
        {"question": "What storage options are available for iPhone 17?",
         "answer": "iPhone 17 offers storage options from 128GB to 512GB, providing ample space for apps, media, and ProRes recordings."},
        {"question": "Can iPhone 17 record ProRes and RAW video?",
         "answer": "Yes, Pro models support ProRes and RAW video capture, with USB‑C external recording for professional workflows."},
        {"question": "Does iPhone 17 support Always‑On display?",
         "answer": "Yes, Pro models have an Always‑On display with glanceable widgets and adaptive refresh to conserve battery."},
    ]
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------
# Session state
# -----------------------------
def ensure_state():
    defaults = {
        "messages": [],
        "conversation_id": f"conv_{int(time.time())}",
        "backend_status": "unknown",
        "faq_loaded": False,
        "upload_status": None,
        "backend_stats": None,
        "db_stats": None,
        "conversation_history": [],
        "auth_token": "",
        "last_prompt": None,  # Add this to track last processed prompt
        "message_count": 0  # Add this to track message count
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_message(role: str, text: str, source: Optional[str] = None, confidence: Optional[float] = None,
                chunks: Optional[List] = None, db_stored: Optional[bool] = None):
    msg = {"role": role, "text": text, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if source is not None: msg["source"] = source
    if confidence is not None:
        try:
            msg["confidence"] = float(confidence)
        except:
            msg["confidence"] = 0.0
    if chunks is not None: msg["chunks"] = chunks
    if db_stored is not None: msg["db_stored"] = db_stored
    st.session_state["messages"].append(msg)
    st.session_state["message_count"] = len(st.session_state["messages"])


# -----------------------------
# Backend API calls
# -----------------------------
def check_health() -> Dict[str, Any]:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            st.session_state["backend_status"] = "connected"
            st.session_state["faq_loaded"] = bool(data.get("faq_loaded", False))
            return {"success": True, "data": data}
        else:
            st.session_state["backend_status"] = "error"
            return {"success": False, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        st.session_state["backend_status"] = "disconnected"
        return {"success": False, "error": str(e)}


def upload_faq(file) -> Dict[str, Any]:
    try:
        headers = get_auth_headers() or {}
        files = {"file": (file.name, file.getvalue(), file.type or "text/csv")}
        r = requests.post(f"{BACKEND_URL}/upload-faq", files=files, headers=headers, timeout=60)
        if r.status_code == 200:
            result = r.json()
            st.session_state["upload_status"] = ("success", result.get("message", "Uploaded successfully"))
            st.session_state["faq_loaded"] = True
            return {"success": True, "message": result.get("message", "")}
        else:
            error_detail = r.json().get("detail", "Upload failed") if r.headers.get('content-type', '').startswith(
                'application/json') else f"HTTP {r.status_code}"
            st.session_state["upload_status"] = ("error", error_detail)
            return {"success": False, "error": error_detail}
    except Exception as e:
        st.session_state["upload_status"] = ("error", f"Connection error: {e}")
        return {"success": False, "error": str(e)}


def get_stats() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{BACKEND_URL}/faq-stats", timeout=10)
        if r.status_code == 200:
            stats = r.json()
            st.session_state["backend_stats"] = stats
            return stats
    except Exception:
        return None
    return None


def get_db_stats() -> Optional[Dict[str, Any]]:
    try:
        headers = get_auth_headers() or {}
        r = requests.get(f"{BACKEND_URL}/db-stats", headers=headers, timeout=10)
        if r.status_code == 200:
            stats = r.json()
            st.session_state["db_stats"] = stats
            return stats
    except Exception:
        return None
    return None


def get_conversation_history(conversation_id: str = None, limit: int = 50) -> Optional[List[Dict]]:
    try:
        headers = get_auth_headers() or {}
        payload = {"conversation_id": conversation_id, "limit": limit}
        r = requests.post(f"{BACKEND_URL}/conversation-history", json=payload, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            history = data.get("history", [])
            st.session_state["conversation_history"] = history
            return history
    except Exception as e:
        st.error(f"Failed to get conversation history: {e}")
        return None
    return None


def call_chat_api(user_text: str) -> Dict[str, Any]:
    payload = {"message": user_text, "conversation_id": st.session_state["conversation_id"]}
    try:
        headers = get_auth_headers() or {}
        r = requests.post(f"{BACKEND_URL}/chat", json=payload, headers=headers, timeout=60)
        if r.status_code == 200:
            return r.json()
        else:
            try:
                error_detail = r.json().get("detail", f"HTTP {r.status_code}")
            except:
                error_detail = f"HTTP {r.status_code}"
            return {"response": f"I encountered an error: {error_detail}. Please try again.", "source": "error",
                    "confidence": 0.0, "db_stored": False}
    except requests.exceptions.Timeout:
        return {"response": "The request timed out. Please try again.", "source": "error", "confidence": 0.0,
                "db_stored": False}
    except Exception as e:
        return {"response": f"Connection error: {str(e)}. Is the backend running?", "source": "error",
                "confidence": 0.0, "db_stored": False}


# -----------------------------
# UI components
# -----------------------------
def render_message(msg: Dict[str, Any]):
    with st.chat_message(msg["role"] if msg["role"] in ("user", "assistant") else "assistant"):
        st.markdown(msg["text"])
        meta = []
        if "source" in msg:
            source_emoji = {"faq": "📚", "system": "⚙️", "error": "❌"}.get(msg["source"], "")
            meta.append(f"{source_emoji} {msg['source']}")
        if "confidence" in msg and msg.get("source") == "faq":
            try:
                pct = int(round(float(msg["confidence"]) * 100))
                meta.append(f"🎯 {pct}%")
            except:
                pass
        if "db_stored" in msg:
            db_icon = "💾✅" if msg["db_stored"] else "💾❌"
            meta.append(f"{db_icon} DB")
        meta.append(f"🕐 {msg['ts']}")
        if meta: st.caption(" • ".join(meta))
        if "chunks" in msg and msg["chunks"]:
            with st.expander("🔍 Retrieved Chunks", expanded=False):
                for i, c in enumerate(msg["chunks"]):
                    st.write(f"**Chunk {i + 1}:** {c.get('question', '')}")
                    st.write(c.get("text", ""))


def render_sidebar():
    with st.sidebar:
        st.subheader("🔐 Authentication")
        token_in = st.text_input("Access Token", type="password", value=st.session_state.get("auth_token", ""))
        colA, colB = st.columns(2)
        with colA:
            if st.button("Authenticate", use_container_width=True):
                tok = token_in.strip()
                if not tok:
                    st.error("Please paste the access token (default: Cvhs@12345).")
                else:
                    res = verify_token(tok)
                    if res["success"]:
                        st.session_state["auth_token"] = tok
                        st.success("Token verified.")
                    else:
                        st.error(f"Verify failed: {res.get('error', 'Unknown error')}")
        with colB:
            if st.button("Clear Token", use_container_width=True):
                st.session_state["auth_token"] = ""
                st.success("Token cleared.")

        st.divider()
        st.subheader("🔌 Connection")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Check Health", use_container_width=True):
                result = check_health()
                if result.get("success"):
                    data = result.get("data", {})
                    db_status = data.get("database_status", "unknown")
                    st.success(f"✅ Connected (DB: {db_status})")
                else:
                    st.error(f"❌ {result.get('error')}")

        with col2:
            status = st.session_state["backend_status"]
            if status == "connected":
                st.success("🟢 Online")
            elif status == "disconnected":
                st.error("🔴 Offline")
            elif status == "error":
                st.error("⚠️ Error")
            else:
                st.info("❓ Unknown")

        st.divider()
        st.subheader("📁 FAQ Data")
        sample = generate_sample_iphone17_faq_csv()
        st.download_button("📥 Download Sample FAQ", data=sample, file_name="iphone17_faq_sample.csv", mime="text/csv",
                           use_container_width=True)
        uploaded_file = st.file_uploader("📤 Upload FAQ (CSV/JSON)", type=["csv", "json"])
        if st.button("Upload FAQ", use_container_width=True,
                     disabled=(uploaded_file is None or not is_authenticated())):
            if not is_authenticated():
                st.warning("Please authenticate first.")
            elif uploaded_file is None:
                st.warning("Select a file.")
            else:
                with st.spinner("Processing..."):
                    result = upload_faq(uploaded_file)
                    if result.get("success"):
                        st.success(result.get("message", "Uploaded"))
                    else:
                        st.error(result.get("error", "Upload failed"))

        if st.session_state["upload_status"]:
            typ, msg = st.session_state["upload_status"]
            (st.success if typ == "success" else st.error)(msg)
        st.info(f"FAQ Loaded: {'Yes' if st.session_state['faq_loaded'] else 'No'}")

        st.divider()
        st.subheader("📊 Statistics")
        col_faq, col_db = st.columns(2)

        with col_faq:
            if st.button("📈 FAQ Stats", use_container_width=True,
                         disabled=st.session_state["backend_status"] != "connected"):
                stats = get_stats()
                if stats:
                    st.json(stats)
                else:
                    st.warning("Could not fetch FAQ stats.")

        with col_db:
            if st.button("🗄️ DB Stats", use_container_width=True,
                         disabled=(st.session_state["backend_status"] != "connected" or not is_authenticated())):
                stats = get_db_stats()
                if stats:
                    st.json(stats)
                else:
                    st.warning("Could not fetch DB stats.")

        st.divider()
        st.subheader("📜 History")

        # Conversation History Section
        col_current, col_all = st.columns(2)

        with col_current:
            if st.button("📋 Current Conv", use_container_width=True, disabled=(not is_authenticated())):
                history = get_conversation_history(st.session_state["conversation_id"], 20)
                if history:
                    st.write("**Current Conversation History:**")
                    for entry in history[:5]:  # Show last 5 entries
                        st.text(f"Q: {entry.get('user_query', '')[:50]}...")
                        st.text(f"A: {entry.get('response', '')[:50]}...")
                        st.caption(f"🕐 {entry.get('created_at', '')}")
                        st.divider()
                else:
                    st.info("No history found for current conversation.")

        with col_all:
            if st.button("📚 All Convs", use_container_width=True, disabled=(not is_authenticated())):
                history = get_conversation_history(None, 50)
                if history:
                    st.write("**All Conversations (Last 50):**")
                    for entry in history[:10]:  # Show last 10 entries
                        st.text(f"Conv: {entry.get('conversation_id', 'Unknown')}")
                        st.text(f"Q: {entry.get('user_query', '')[:40]}...")
                        st.caption(f"🕐 {entry.get('created_at', '')}")
                        st.divider()
                else:
                    st.info("No conversation history found.")

        st.divider()
        st.subheader("⚙️ Controls")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["messages"].clear()
            st.session_state["conversation_id"] = f"conv_{int(time.time())}"
            add_message("assistant", "Chat cleared. Upload your iPhone 17 FAQs and ask a question.", source="system",
                        confidence=1.0, db_stored=False)


# -----------------------------
# Database History View
# -----------------------------
def render_database_tab():
    st.subheader("🗄️ Database View")

    if not is_authenticated():
        st.warning("Please authenticate to view database information.")
        return

    # Top controls row
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        if st.button("🔄 Refresh DB Stats", use_container_width=True):
            stats = get_db_stats()
            if stats and "error" not in stats:
                st.success("Database stats refreshed!")
            else:
                st.error("Failed to refresh database stats")

    with col2:
        limit = st.selectbox("Limit", [10, 25, 50, 100], index=1)

    with col3:
        conv_filter = st.text_input("🔍 Filter by Conversation ID", placeholder="Optional filter...")

    st.divider()

    # Display DB Stats
    if st.session_state.get("db_stats"):
        stats = st.session_state["db_stats"]
        if "error" not in stats:
            st.subheader("📊 Database Statistics")

            # Metrics in columns
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

            with col_stat1:
                total_convs = stats.get("total_conversations", 0)
                st.metric("Total Conversations", total_convs)

            with col_stat2:
                unique_convs = stats.get("unique_conversations", 0)
                st.metric("Unique Conversations", unique_convs)

            with col_stat3:
                faq_responses = stats.get("faq_responses", 0)
                st.metric("FAQ Responses", faq_responses)

            with col_stat4:
                avg_conf = stats.get("avg_confidence", 0)
                if avg_conf and avg_conf != 0:
                    st.metric("Avg Confidence", f"{float(avg_conf):.1%}")
                else:
                    st.metric("Avg Confidence", "N/A")

            # Additional info
            last_conv = stats.get("last_conversation")
            if last_conv:
                st.info(f"🕐 Last conversation: {last_conv}")
        else:
            st.error(f"❌ Database Error: {stats.get('error')}")
    else:
        st.info("Click 'Refresh DB Stats' to load database statistics.")

    st.divider()

    # Conversation History Section
    st.subheader("📜 Conversation History")

    col_load, col_export = st.columns([3, 1])

    with col_load:
        if st.button("📥 Load Conversation History", use_container_width=True):
            conv_id = conv_filter.strip() if conv_filter.strip() else None
            with st.spinner("Loading conversation history..."):
                history = get_conversation_history(conv_id, limit)

            if history:
                st.session_state["loaded_history"] = history
                st.success(f"✅ Loaded {len(history)} conversation entries")
            else:
                st.warning("No conversation history found.")

    with col_export:
        # Export functionality
        if st.session_state.get("loaded_history"):
            history_df = pd.DataFrame(st.session_state["loaded_history"])
            csv_data = history_df.to_csv(index=False)
            st.download_button(
                label="📤 CSV",
                data=csv_data,
                file_name=f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Display loaded history
    if st.session_state.get("loaded_history"):
        history = st.session_state["loaded_history"]
        st.write(f"**📋 Showing {len(history)} conversation entries:**")

        # Create a more compact DataFrame for display
        display_data = []
        for entry in history:
            display_data.append({
                "ID": entry.get("id"),
                "Conv ID": entry.get("conversation_id", "")[:15] + "..." if len(
                    entry.get("conversation_id", "")) > 15 else entry.get("conversation_id", ""),
                "User Query": entry.get("user_query", "")[:60] + "..." if len(
                    entry.get("user_query", "")) > 60 else entry.get("user_query", ""),
                "Response Preview": entry.get("response", "")[:60] + "..." if len(
                    entry.get("response", "")) > 60 else entry.get("response", ""),
                "Source": entry.get("source", ""),
                "Confidence": f"{float(entry.get('confidence', 0)):.1%}" if entry.get('confidence') else "N/A",
                "Created": entry.get("created_at", "")[:16] if entry.get("created_at") else ""
                # Show date and time only
            })

        # Display the data table
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Detailed view with pagination
        st.subheader("🔍 Detailed View")

        # Pagination controls
        total_entries = len(history)
        entries_per_page = 5
        total_pages = (total_entries + entries_per_page - 1) // entries_per_page

        if total_pages > 1:
            col_prev, col_page, col_next = st.columns([1, 2, 1])

            if 'detail_page' not in st.session_state:
                st.session_state['detail_page'] = 1

            with col_prev:
                if st.button("⬅️ Previous", disabled=(st.session_state['detail_page'] <= 1)):
                    st.session_state['detail_page'] -= 1
                    st.rerun()

            with col_page:
                st.write(f"Page {st.session_state['detail_page']} of {total_pages}")

            with col_next:
                if st.button("➡️ Next", disabled=(st.session_state['detail_page'] >= total_pages)):
                    st.session_state['detail_page'] += 1
                    st.rerun()
        else:
            st.session_state['detail_page'] = 1

        # Show entries for current page
        start_idx = (st.session_state.get('detail_page', 1) - 1) * entries_per_page
        end_idx = min(start_idx + entries_per_page, total_entries)

        for i in range(start_idx, end_idx):
            entry = history[i]
            with st.expander(f"Entry {i + 1} - ID: {entry.get('id')} ({entry.get('created_at', '')[:16]})",
                             expanded=False):
                st.write(f"**🆔 Database ID:** {entry.get('id')}")
                st.write(f"**💬 Conversation ID:** {entry.get('conversation_id', 'N/A')}")
                st.write(f"**❓ User Query:**")
                st.text_area("Query", entry.get('user_query', ''), height=100, disabled=True, key=f"query_{i}")
                st.write(f"**💡 Response:**")
                st.text_area("Response", entry.get('response', ''), height=150, disabled=True, key=f"response_{i}")

                # Metadata row
                col_source, col_conf, col_date = st.columns(3)
                with col_source:
                    source = entry.get('source', 'N/A')
                    source_emoji = {"faq": "📚", "system": "⚙️", "error": "❌"}.get(source, "❓")
                    st.write(f"**Source:** {source_emoji} {source}")

                with col_conf:
                    conf = entry.get('confidence', 0)
                    if conf and conf > 0:
                        st.write(f"**Confidence:** 🎯 {float(conf):.1%}")
                    else:
                        st.write(f"**Confidence:** N/A")

                with col_date:
                    st.write(f"**Created:** 🕐 {entry.get('created_at', 'N/A')}")
    else:
        st.info("👆 Click 'Load Conversation History' to view stored conversations from the database.")


# -----------------------------
# Main Chat Interface
# -----------------------------
def render_main_chat():
    # Chat input must be at the main level, not inside tabs/columns
    prompt = st.chat_input("Ask about the iPhone 17 series…",
                           disabled=(st.session_state["backend_status"] != "connected" or not is_authenticated()))

    if prompt:
        add_message("user", prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = call_chat_api(prompt)
        add_message("assistant",
                    res.get("response", "There was a problem."),
                    source=res.get("source", "system"),
                    confidence=res.get("confidence", 0.0),
                    chunks=res.get("retrieved_chunks"),
                    db_stored=res.get("db_stored", False))

    # Display messages
    if st.session_state["messages"]:
        for msg in st.session_state["messages"]:
            render_message(msg)
    else:
        st.info(
            "Paste the access token in the sidebar (default: Cvhs@12345), verify, upload the sample FAQ, then start chatting.")

    if st.session_state["backend_status"] != "connected":
        st.warning(f"Backend not connected. Expected at {BACKEND_URL}. Start the FastAPI server and try Health.")


# -----------------------------
# App main
# -----------------------------
def main():
    ensure_state()
    st.title("📱 iPhone 17 FAQ Assistant with PostgreSQL")
    st.caption(
        "Secure static token • Semantic retrieval (FAISS) • Reranking • Mistral-7B-Instruct • PostgreSQL Storage")

    # Always render sidebar
    render_sidebar()

    # Create tabs for different views
    tab1, tab2 = st.tabs(["💬 Chat", "🗄️ Database"])

    with tab1:
        st.subheader("💬 Chat Interface")

        # Display messages within the tab
        if st.session_state["messages"]:
            for msg in st.session_state["messages"]:
                render_message(msg)
        else:
            st.info(
                "Paste the access token in the sidebar (default: Cvhs@12345), verify, upload the sample FAQ, then start chatting.")

        if st.session_state["backend_status"] != "connected":
            st.warning(f"Backend not connected. Expected at {BACKEND_URL}. Start the FastAPI server and try Health.")

    with tab2:
        render_database_tab()

    # Chat input at main level (outside of tabs)
    st.divider()
    prompt = st.chat_input("Ask about the iPhone 17 series…",
                           disabled=(st.session_state["backend_status"] != "connected" or not is_authenticated()))

    # Process the prompt only if it's new and different from the last one
    if prompt and prompt.strip() and prompt != st.session_state.get("last_prompt"):
        st.session_state["last_prompt"] = prompt

        # Add user message
        add_message("user", prompt)

        # Process response in a placeholder to show real-time updates
        response_placeholder = st.empty()
        with response_placeholder.container():
            with st.spinner("Thinking..."):
                res = call_chat_api(prompt)

        # Clear the placeholder and add the actual response
        response_placeholder.empty()
        add_message("assistant",
                    res.get("response", "There was a problem."),
                    source=res.get("source", "system"),
                    confidence=res.get("confidence", 0.0),
                    chunks=res.get("retrieved_chunks"),
                    db_stored=res.get("db_stored", False))

        # Rerun to show the new messages
        st.rerun()


if __name__ == "__main__":
    main()

