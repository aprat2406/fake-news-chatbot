"""
Streamlit Chatbot UI for Fake News Detection
--------------------------------------------
- Connects to FastAPI inference API (http://localhost:8000/predict)
- Displays model prediction (Fake/True) and confidence
- Shows retrieved evidence (if available)
- Maintains chat history across messages
"""

import streamlit as st
import requests
import json
from datetime import datetime

# -------------------
# Configuration
# -------------------
API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Fake News Detection Chatbot", page_icon="🧠", layout="wide")

# -------------------
# Session state
# -------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# -------------------
# Header
# -------------------
st.title("🧠 Fake News Detection Chatbot")
st.caption("Check whether a news claim or statement is real or fake using our fine-tuned BERT model.")

st.markdown("---")

# -------------------
# Input Section
# -------------------
user_input = st.text_area("📰 Enter a news headline, statement, or paragraph:", height=120, key="input_text")

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    use_rag = st.checkbox("Use RAG (Evidence Search)", value=False)
with col2:
    top_k = st.number_input("Top Evidence", min_value=1, max_value=5, value=3, step=1)

# -------------------
# Predict Button
# -------------------
if st.button("🔍 Check Veracity"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                payload = {"text": user_input, "use_rag": use_rag, "top_k": top_k}
                response = requests.post(API_URL, json=payload, timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    label = result.get("label", "unknown").upper()
                    conf = round(result.get("confidence", 0.0) * 100, 2)
                    evidence = result.get("evidence", [])

                    # Save in chat history
                    st.session_state["history"].append({
                        "user": user_input,
                        "label": label,
                        "confidence": conf,
                        "evidence": evidence,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })

                    st.success(f"✅ Prediction: **{label}**  (Confidence: {conf}%)")

                    # Evidence Section
                    if use_rag and evidence:
                        st.markdown("#### 🔎 Supporting Evidence")
                        for idx, ev in enumerate(evidence, start=1):
                            st.markdown(
                                f"**{idx}.** {ev.get('text','')}\n\n*Source:* {ev.get('source','Unknown')} *(Score: {round(ev.get('score',0),3)})*"
                            )
                    elif use_rag:
                        st.info("No evidence found for this query.")
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error contacting API: {e}")

# -------------------
# Conversation History
# -------------------
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("🗂️ Chat History")
    for chat in reversed(st.session_state["history"]):
        st.markdown(f"**🧍 User:** {chat['user']}")
        st.markdown(f"**🤖 Chatbot:** {chat['label']}  (Confidence: {chat['confidence']}%)")
        if chat["evidence"]:
            with st.expander("See Evidence"):
                for idx, ev in enumerate(chat["evidence"], start=1):
                    st.markdown(f"{idx}. {ev.get('text','')}  \n*Source:* {ev.get('source','Unknown')}")

