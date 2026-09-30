import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
HISTORY_FILE = "chat_history.json"

@st.cache_data
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(all_sessions):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_sessions[-20:], f, indent=2, ensure_ascii=False)  # Last 20

@st.cache_resource
def load_vector_store():
    if not os.path.exists("faq.txt"):
        st.error("❌ Create **faq.txt**!")
        st.stop()
    loader = TextLoader("faq.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(texts, embeddings)

st.set_page_config(page_title="📱 FAQ Chatbot", page_icon="📱", layout="wide", initial_sidebar_state="expanded")

# Sidebar: Button-based History (Always Visible!)
with st.sidebar:
    st.title("📚 Chat History")
    
    # Debug: Show file status
    if os.path.exists(HISTORY_FILE):
        size = os.path.getsize(HISTORY_FILE)
        st.metric("💾 History File", f"{size/1000:.1f} KB")
    else:
        st.warning("📝 **No history yet** - Chat first!")
    
    all_sessions = load_history()
    st.caption(f"**{len(all_sessions)} sessions** saved")
    
    if all_sessions:
        st.subheader("📋 Recent Chats (Click to Load)")
        for session in all_sessions[-5:]:  # Last 5
            btn_label = f"**{session['title'][:30]}...** ({len(session['messages'])} msgs)"
            if st.button(btn_label, key=session["session_id"], use_container_width=True):
                st.session_state.messages = session["messages"]
                st.success(f"✅ Loaded **{session['title']}**")
                st.rerun()
        
        if st.button("🗑️ Clear All", type="secondary"):
            os.remove(HISTORY_FILE)
            st.rerun()
    else:
        st.info("💬 **Chat 3+ messages** → Auto-saves here!")

# Main Chat
st.title("📱 Persistent FAQ Chatbot")
col1, col2 = st.columns([3,1])
with col2:
    if st.button("🔄 Refresh History", use_container_width=True):
        st.rerun()

# Init/Load messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 History works! Ask 'hours' to test & save."}]

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("💭 Ask FAQ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍..."):
            db = load_vector_store()
            docs = db.similarity_search(prompt, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            st.markdown(f"**📋 Matches:**\n\n{context}")

        st.session_state.messages.append({"role": "assistant", "content": context})

    # Auto-save if 3+ messages
    if len(st.session_state.messages) >= 3:
        all_sessions = load_history()
        session_id = f"chat_{len(all_sessions)+1}_{datetime.now().strftime('%H%M')}"
        session_data = {
            "session_id": session_id,
            "title": f"FAQ Chat - {prompt[:50]}",
            "date": datetime.now().strftime("%m/%d %H:%M"),
            "messages": st.session_state.messages.copy()
        }
        all_sessions.append(session_data)
        save_history(all_sessions)
        st.sidebar.success("💾 Auto-saved!")
        st.rerun()

# Footer debug
with st.expander("🔧 Debug Info"):
    st.json({"sessions": len(load_history()), "file_exists": os.path.exists(HISTORY_FILE)})