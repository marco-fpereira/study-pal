import asyncio
import nest_asyncio
import os
import tempfile
import shutil
import streamlit as st
from dotenv import load_dotenv
from model.enum.llm_enum import LLMProviders
from model.enum.user_session_type_enum import UserSessionTypeEnum
from repository.vector_db_repository import VectorDBRepository
from service.llm_chat_service import LLMChatService
import streamlit.components.v1 as components
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

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "choose_button_disabled" not in st.session_state:
    st.session_state.choose_button_disabled = False

if "generate_questions_button_disabled" not in st.session_state:
    st.session_state.generate_questions_button_disabled = False

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_session_id() -> str:
    ctx = get_script_run_ctx()
    return ctx.session_id


def clean_input_alphanumeric():
    # Retrieve current text from session state
    raw_text = st.session_state["user_session_input"]
    
    # Remove any character that is NOT a letter or number
    import re
    cleaned_text = re.sub(r'[^a-zA-Z0-9-_ ]', '', raw_text)
    
    # Overwrite the session state with the alphanumeric string
    st.session_state["user_session_input"] = cleaned_text


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


def get_known_subjects(session_id: str) -> list[str]:
    """
    Return subjects already ingested in this session.
    Adjust to however VectorDBRepository exposes its namespaces/collections.
    """
    try:
        return st.session_state.vector_db_repo.list_subjects(session_id=session_id)
    except Exception as e:
        print(f"Error getting known subjects: {e}")
        # fallback subjects list
        return list(st.session_state.get("ingested_subjects", set()))


def disable_select_user_button():
    st.session_state.choose_button_disabled = True


def disable_generate_questions_button():
    st.session_state.generate_questions_button_disabled = True

def enable_generate_questions_button():
    st.session_state.generate_questions_button_disabled = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.html(
    """
    <style>
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.2rem; /* Default is usually 1rem. Set lower to decrease space */
        }
    </style>
    """
)

with st.sidebar:
    st.title("📚 Study Pal")

    user_session_type = st.radio(
        label="How do you want to enter your user?", 
        options=[
            UserSessionTypeEnum.CHOOSE.value,           
            UserSessionTypeEnum.RANDOM.value, 
        ],
        horizontal=True,
        index=0,
        disabled = st.session_state.session_id is not None
    )

    if st.session_state.session_id is None:
        match (user_session_type):
            case UserSessionTypeEnum.CHOOSE.value:
                _session_id = st.text_input(
                    label="Inform your user for current session: ", placeholder="characters and numbers only",
                    key="user_session_input",
                    on_change=clean_input_alphanumeric
                )
                if _session_id and _session_id.strip():
                    if st.button(
                        "Confirm", 
                        on_click=disable_select_user_button, 
                        disabled=st.session_state.choose_button_disabled
                    ):
                        with st.spinner("Loading your user..."):
                            _session_id = _session_id.replace(" ", "_")
                            st.session_state.session_id = _session_id
                            st.rerun()
            case UserSessionTypeEnum.RANDOM.value:
                if st.button(
                    "Confirm", 
                    on_click=disable_select_user_button,
                    disabled=st.session_state.choose_button_disabled
                ):
                    with st.spinner("Loading your user..."):
                        st.session_state.session_id = get_session_id()
                        st.rerun()
    else:
        st.subheader(f"User {st.session_state.session_id}")

        st.divider()

        # --- Select provider --- 
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
                        ingest_files(
                            uploaded_files=uploaded_files, 
                            session_id=st.session_state.session_id, 
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
        known_subjects = get_known_subjects(st.session_state.session_id)

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
        
        st.divider()

        #  --- Enable question mode ---
        toggle_exam_mode_enabled = st.toggle(label="Enable exam mode", width="stretch")
        if "toggle_exam_mode_enabled" not in st.session_state or st.session_state.toggle_exam_mode_enabled != toggle_exam_mode_enabled:
            st.session_state.toggle_exam_mode_enabled = toggle_exam_mode_enabled

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

if "llm_chat_service" not in st.session_state or st.session_state.llm_chat_service.model_name != st.session_state.selected_model:
    st.session_state.llm_chat_service = LLMChatService(
        vector_db_repository=st.session_state.vector_db_repo,
        llm_provider=st.session_state.selected_provider,
        model_name=st.session_state.selected_model,
    )

history = st.session_state.llm_chat_service.get_session_history(session_id=st.session_state.session_id)

if st.session_state.toggle_exam_mode_enabled:

    if "exam_context" not in st.session_state:
        st.session_state.exam_context = st.session_state.llm_chat_service.generate_exam_context_for_responses(
            session_id=st.session_state.session_id
        )

    def toggle_select_all():
        if st.session_state.select_all:
            st.session_state.topic_multiselect = st.session_state.exam_context
        else:
            st.session_state.topic_multiselect = []

    st.checkbox("Select All", key="select_all", on_change=toggle_select_all)
    topics = st.multiselect(
        label="Choose topics to generate questions about", 
        key="topic_multiselect",
        options=st.session_state.exam_context,
        accept_new_options=True,
    )

    with st.form(key="exam_form", clear_on_submit=True):
        if st.form_submit_button(
            "Generate Questions", 
            on_click=disable_generate_questions_button,
            disabled=st.session_state.generate_questions_button_disabled
        ):
            with st.spinner("Generating questions..."):
                topics_str = "\n- ".join(topics)
                query = f"Generate questions about the below topics and return them as an HTML to test user's knowledge.\n\nTopics:\n{topics_str}"
                response = asyncio.run(
                    st.session_state.llm_chat_service.generate_response(
                        session_id=st.session_state.session_id,
                        query=query,
                        enable_exam_mode=True
                    )
                )
            enable_generate_questions_button()

            components.html(
                response["output"][0]["text"],
                height=1200,
                scrolling=True
            )

else:
    # Render existing messages
    if history:
        for message in history["messages"]:
            role = message["role"]
            content = message["content"]
            exam_mode = message.get("exam_mode", False)

            with st.chat_message(role):
                if not exam_mode:
                    st.markdown(content)

    user_input = st.chat_input(f"Ask AI about {active_subject.replace('_', ' ')}…")

    if user_input:
        with st.chat_message('user'):
            st.markdown(user_input)

        with st.chat_message('assistant'):
            response = asyncio.run(
                st.session_state.llm_chat_service.generate_response(
                    session_id=st.session_state.session_id,
                    query=user_input,
                    enable_exam_mode=False
                )
            )

            st.markdown(response["output"])    
            if response["source_documents"]:
                with st.expander("📚 Sources"):
                    for doc in response["source_documents"]:
                        st.markdown(f"- {doc}")
