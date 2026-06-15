import asyncio
import nest_asyncio
import os
import tempfile
import shutil
import streamlit as st
from dotenv import load_dotenv
from model.enum.llm_enum import LLMProviders
from repository.vector_db_repository import VectorDBRepository
from service.llm_chat_service import LLMChatService
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.runtime.uploaded_file_manager import UploadedFile
from transformers.utils import logging

nest_asyncio.apply()

# load environment variables from .env file
load_dotenv()

logging.set_verbosity_error()

# Streamlit page setup
st.set_page_config(
    page_title="Study Pal", 
    page_icon="📚",
    layout="wide"
)

if "vector_db_repo" not in st.session_state:
    st.session_state.vector_db_repo = VectorDBRepository()

if "selected_provider" not in st.session_state:
    st.session_state.selected_provider = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "active_subject" not in st.session_state:
    st.session_state.active_subject = None

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_session_id() -> str:
    ctx = get_script_run_ctx()
    return ctx.session_id

def ingest_files(
    uploaded_files: list[UploadedFile],
    session_id: str,
    subject: str
):
    tmp_dir = tempfile.mkdtemp()
    files_path = []

    for uploaded_file in uploaded_files:
        tmp_path = os.path.join(tmp_dir, uploaded_file.name)  # preserves original name
        with open(tmp_path, "wb") as tmp_file:
            tmp_file.write(uploaded_file.read())
        files_path.append(tmp_path)

    try:
        st.session_state.vector_db_repo.ingest_document(
            session_id=session_id, 
            subject=subject, 
            docs_path=files_path
        )
    except Exception as e:
        print(f"Error ingesting files: {e}")
    finally:
        shutil.rmtree(tmp_dir)  # delete temp dir and all files in it after ingestion


def get_known_subjects() -> list[str]:
    """
    Return subjects already ingested in this session.
    Adjust to however VectorDBRepository exposes its namespaces/collections.
    """
    session_id = get_session_id()
    try:
        return st.session_state.vector_db_repo.list_subjects(session_id=session_id)
    except Exception as e:
        print(f"Error getting known subjects: {e}")
        # fallback subjects list
        return list(st.session_state.get("ingested_subjects", set()))
    
# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 Study Pal")

#def choose_model():

    # Select provider
    st.subheader("AI Model")

    if st.session_state.selected_provider is None or st.session_state.selected_provider == "":
        provider = LLMProviders(
            st.selectbox(
                "Select AI Provider", 
                [provider.value for provider in LLMProviders]
            )
        )
        
        if st.session_state.selected_model is None or st.session_state.selected_model == "":
            # Select model based on provider
            model = st.selectbox(
                "Select AI Model", 
                provider.get_available_models()
            )

            # Confirm button
            if st.button("Confirm Selection"):
                st.session_state.selected_provider = provider
                st.session_state.selected_model = model
                st.rerun()
    else:
        # Show current selection with option to change
        st.caption(
            f"✓ {st.session_state.selected_provider} — {st.session_state.selected_model}"
        )
        if st.button("Change Selection"):
            st.session_state.selected_provider = None
            st.session_state.selected_model = None
            st.rerun()

    st.divider()

    # --- File upload ---
    with st.form(key="my_form", clear_on_submit=True):
        st.subheader("Upload documents")
        raw_subject = st.text_input("Subject name", placeholder="e.g. Linear Algebra")
        subject = raw_subject.replace(" ", "_")

        uploaded_files = st.file_uploader(
            "Upload files (PDF, PPT, DOC)", 
            type=["pdf", "ppt", "pptx", "doc", "docx"], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        submit_button = st.form_submit_button("Ingest Documents", use_container_width=True)

        if len(uploaded_files) > 0 and len(subject) > 0:
            if submit_button:
                with st.spinner("Uploading and indexing document..."):
                    session_id = get_session_id()
                    ingest_files(
                        uploaded_files=uploaded_files, 
                        session_id=session_id, 
                        subject=subject
                    )
                    # Track subject locally as a fallback
                    subjects = st.session_state.setdefault("ingested_subjects", set())
                    subjects.add(subject)
                st.success("Document uploaded and indexed successfully!")
                st.rerun()
        
        st.divider()

    # --- Subject navigation ---
    st.subheader("Subjects")
    known_subjects = get_known_subjects()

    if not known_subjects:
        st.caption("No subjects yet. Upload documents above.")
    else:
        for subj in sorted(known_subjects):
            is_active = st.session_state.active_subject == subj
            if st.button(
                f"{'▶ ' if is_active else ''}{subj}",
                key=f"subj_{subj}",
                use_container_width=True,
            ):
                st.session_state.active_subject = subj
                st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────

active_subject = st.session_state.active_subject

if st.session_state.selected_provider is None:
    st.info("👈 Select an AI model in the sidebar to get started.")
    st.stop()

if not active_subject:
    st.info("👈 Upload documents and select a subject from the sidebar.")
    st.stop()

st.title("📚 Study Pal - RAG Powered AI Study Assistant")
st.subheader(f"📖 {active_subject.replace('_', ' ')}")

llm = st.session_state.selected_provider.get_llm_instance(st.session_state.selected_model)

if "llm_chat_service" not in st.session_state:
    st.session_state.llm_chat_service = LLMChatService(
        retriever=st.session_state.vector_db_repo.config.get_vector_store().as_retriever(),
        llm_provider=st.session_state.selected_provider,
        model_name=st.session_state.selected_model,
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(st.session_state.llm_chat_service.initialize())

session_id = get_session_id()
history = st.session_state.llm_chat_service.get_session_history(session_id=session_id)
# Render existing messages
if history:
    for message in history:
        role = None
        if message.type == "human":
            role = "user"
        elif message.type == "ai":
            role = "assistant"
        else:
            continue

        if role and message.content:
            with st.chat_message(role):
                st.markdown(message.content)

user_input = st.chat_input(f"Ask AI about {active_subject.replace('_', ' ')}…")

if user_input:
    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(
            st.session_state.llm_chat_service.generate_response(
                session_id=session_id,
                query=user_input
            )
        )
        st.markdown(response["output"])
    
        with st.expander("📚 Sources"):
            for doc in response["source_documents"]:
                st.markdown(f"- {doc}")