import streamlit as st
import os
from supporting_func import create_vector_store, create_rag_chain

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg,#0f172a,#020617);
    color:white;
}

.stButton button {
    background: linear-gradient(90deg,#06b6d4,#3b82f6);
    border:none;
    border-radius:10px;
    color:white;
    font-weight:bold;
    transition:0.3s;
}

.stButton button:hover {
    transform:scale(1.05);
}

.block-container {
    padding-top:2rem;
}

.css-1kyxreq {
    justify-content:center;
}

</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center;'>📄 RAG PDF Assistant</h1>
<p style='text-align:center;color:gray;'>
Upload a PDF and ask questions from it
</p>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:

    st.markdown("## ⚙️ Control Panel")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=['pdf']
    )

    process_button = st.button("Process Document")

    st.divider()

    st.markdown("### System status")

    if "rag_chain" in st.session_state:
        st.success("Model Ready")
    else:
        st.warning("No document loaded")

# ---------- SESSION STATE ----------
if "vector_store" not in st.session_state:
    st.session_state.vector_store=None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain=None

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

# ---------- PROCESS FILE ----------
if process_button and uploaded_file:

    with st.spinner("Processing PDF..."):

        temp_dir="temp_docs"

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        file_path=os.path.join(
            temp_dir,
            uploaded_file.name
        )

        with open(file_path,"wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.vector_store=create_vector_store(file_path)

        st.session_state.rag_chain=create_rag_chain(
            st.session_state.vector_store
        )

    st.success("Document ready")

# ---------- CHAT UI ----------
if st.session_state.rag_chain:

    st.markdown("## 💬 Ask questions")

    question=st.chat_input("Ask something about the PDF")

    if question:

        st.session_state.chat_history.append(
            ("user",question)
        )

        with st.spinner("Thinking..."):

            try:

                response=st.session_state.rag_chain.invoke(
                    {"input":question}
                )

                answer=response["answer"]

                st.session_state.chat_history.append(
                    ("assistant",answer)
                )

            except Exception as e:

                st.error(e)

# ---------- DISPLAY CHAT ----------
for role,msg in st.session_state.chat_history:

    if role=="user":

        with st.chat_message("user"):
            st.write(msg)

    else:

        with st.chat_message("assistant"):
            st.write(msg)