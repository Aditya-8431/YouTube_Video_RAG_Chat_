import streamlit as st

from app import (
    indexing_chain,
    create_retriever,
    build_rag_chain,
    extract_video_id
)

st.set_page_config(
    page_title="VideoChat RAG",
    page_icon="V",
    layout="wide"
)

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "active_chat" not in st.session_state:
    st.session_state.active_chat = None

st.markdown("""
<style>
    .stApp {
        background: #000000;
        color: #f5f5f5;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 30px;
        padding-bottom: 40px;
    }

    section[data-testid="stSidebar"] {
        background: #050505;
        border-right: 1px solid #1a1a1a;
    }

    section[data-testid="stSidebar"] > div {
        background: #050505;
    }

    h1, h2, h3, h4 {
        color: #ffffff;
    }

    .app-title {
        font-size: 38px;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 4px;
    }

    .app-subtitle {
        color: #8a8a8a;
        font-size: 15px;
        margin-bottom: 28px;
    }

    .section-box {
        background: #080808;
        border: 1px solid #1c1c1c;
        border-radius: 14px;
        padding: 20px;
    }

    .info-box {
        background: #080808;
        border: 1px solid #1c1c1c;
        border-radius: 12px;
        padding: 16px;
        transition: 0.2s ease;
    }

    .info-box:hover {
        border-color: #2563eb;
        box-shadow: 0 0 0 1px #2563eb;
    }

    .info-title {
        color: #ffffff;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .info-text {
        color: #8a8a8a;
        font-size: 13px;
        line-height: 1.5;
    }

    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 2px;
    }

    .sidebar-subtitle {
        color: #707070;
        font-size: 12px;
        margin-bottom: 18px;
    }

    .sidebar-label {
        color: #666666;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .video-status {
        color: #3b82f6;
        font-size: 13px;
        margin-top: 10px;
    }

    .video-id {
        color: #707070;
        font-size: 12px;
        margin-top: 4px;
    }

    div.stButton > button {
        background: #0a0a0a;
        color: #dddddd;
        border: 1px solid #242424;
        border-radius: 9px;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: #0b1220;
        color: #60a5fa;
        border-color: #2563eb;
    }

    div.stButton > button:focus {
        box-shadow: none;
    }

    div[data-baseweb="input"] > div {
        background: #050505;
        border: 1px solid #252525;
        border-radius: 9px;
    }

    div[data-baseweb="input"] > div:focus-within {
        border-color: #2563eb;
        box-shadow: 0 0 0 1px #2563eb;
    }

    textarea {
        background: #050505 !important;
        color: #ffffff !important;
    }

    textarea:focus {
        border-color: #2563eb !important;
    }

    .stChatMessage {
        background: #050505;
        border: 1px solid #171717;
        border-radius: 12px;
    }

    .stChatMessage:hover {
        border-color: #202020;
    }

    .welcome {
        text-align: center;
        padding: 80px 20px;
        color: #777777;
    }

    .welcome h3 {
        color: #dddddd;
        margin-bottom: 8px;
    }

    .welcome p {
        color: #707070;
        font-size: 14px;
    }

    .status-line {
        display: inline-block;
        color: #60a5fa;
        font-size: 12px;
        border: 1px solid #1d4ed8;
        background: #07111f;
        border-radius: 20px;
        padding: 5px 10px;
        margin-bottom: 12px;
    }

    hr {
        border-color: #171717;
    }

    iframe {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


def create_conversation(video_id, video_url, rag_chain):
    chat_id = str(len(st.session_state.conversations) + 1)

    conversation = {
        "id": chat_id,
        "title": "New conversation",
        "video_id": video_id,
        "video_url": video_url,
        "rag_chain": rag_chain,
        "messages": []
    }

    st.session_state.conversations.insert(0, conversation)
    st.session_state.active_chat = chat_id


def get_active_conversation():
    for conversation in st.session_state.conversations:
        if conversation["id"] == st.session_state.active_chat:
            return conversation
    return None


def clear_conversation():
    conversation = get_active_conversation()

    if conversation:
        conversation["messages"] = []


def delete_conversation(chat_id):
    st.session_state.conversations = [
        conversation
        for conversation in st.session_state.conversations
        if conversation["id"] != chat_id
    ]

    if st.session_state.active_chat == chat_id:
        st.session_state.active_chat = None


with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">VideoChat RAG</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">YouTube video conversations</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "New Chat",
        use_container_width=True
    ):
        st.session_state.active_chat = None
        st.rerun()

    st.divider()

    st.markdown(
        '<div class="sidebar-label">CONVERSATIONS</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.conversations:
        st.caption("No conversations yet.")

    else:
        for conversation in st.session_state.conversations:
            title = conversation["title"]

            if len(title) > 30:
                title = title[:30] + "..."

            if st.button(
                title,
                key=f"conversation_{conversation['id']}",
                use_container_width=True
            ):
                st.session_state.active_chat = conversation["id"]
                st.rerun()

    active_conversation = get_active_conversation()

    if active_conversation:
        st.divider()

        st.markdown(
            '<div class="sidebar-label">CURRENT VIDEO</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="video-status">Video loaded</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="video-id">{active_conversation["video_id"]}</div>',
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "Clear Conversation",
            use_container_width=True
        ):
            clear_conversation()
            st.rerun()

        if st.button(
            "Delete Conversation",
            use_container_width=True
        ):
            delete_conversation(
                active_conversation["id"]
            )
            st.rerun()


st.markdown(
    '<div class="app-title">VideoChat RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">Ask questions about a YouTube video using retrieval-augmented generation.</div>',
    unsafe_allow_html=True
)


active_conversation = get_active_conversation()


if active_conversation is None:

    st.markdown(
        '<div class="section-box">',
        unsafe_allow_html=True
    )

    st.subheader("Load a YouTube video")

    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="Paste your YouTube video URL"
    )

    load_button = st.button(
        "Load Video",
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-box">
                <div class="info-title">Transcript</div>
                <div class="info-text">
                    The transcript is retrieved and divided
                    into smaller chunks.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-box">
                <div class="info-title">Retrieval</div>
                <div class="info-text">
                    FAISS searches the transcript for
                    relevant information.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="info-box">
                <div class="info-title">Generation</div>
                <div class="info-text">
                    Gemini generates an answer using
                    the retrieved context.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if load_button:

        if not youtube_url.strip():
            st.error("Please enter a YouTube URL.")

        else:

            try:
                video_id = extract_video_id(
                    youtube_url.strip()
                )

                with st.status(
                    "Preparing video...",
                    expanded=True
                ):
                    st.write("Fetching transcript...")

                    data = indexing_chain.invoke(
                        youtube_url.strip()
                    )

                    st.write("Creating retriever...")

                    vector_store = data["vector_store"]

                    retriever = create_retriever(
                        vector_store
                    )

                    st.write("Creating RAG chain...")

                    rag_chain = build_rag_chain(
                        retriever
                    )

                create_conversation(
                    video_id,
                    youtube_url.strip(),
                    rag_chain
                )

                st.success(
                    "Video loaded successfully."
                )

                st.rerun()

            except Exception as e:
                st.error(str(e))


else:

    left_column, right_column = st.columns(
        [1, 1.4],
        gap="large"
    )

    with left_column:

        st.markdown(
            '<div class="status-line">Video loaded</div>',
            unsafe_allow_html=True
        )

        st.subheader("Video")

        st.video(
            active_conversation["video_url"]
        )

        st.caption(
            f"Video ID: {active_conversation['video_id']}"
        )

    with right_column:

        st.subheader("Chat")

        if not active_conversation["messages"]:

            st.markdown(
                """
                <div class="welcome">
                    <h3>Start a conversation</h3>
                    <p>
                        Ask questions about the content of this video.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            suggestion1, suggestion2 = st.columns(2)

            with suggestion1:
                if st.button(
                    "Summarize the video",
                    use_container_width=True
                ):
                    selected_question = "Summarize the video."
                else:
                    selected_question = None

            with suggestion2:
                if st.button(
                    "What are the main topics?",
                    use_container_width=True
                ):
                    selected_question = "What are the main topics discussed?"
                else:
                    if "selected_question" not in locals():
                        selected_question = None

        else:
            selected_question = None

        question = st.chat_input(
            "Ask something about this video..."
        )

        if question:
            selected_question = question

        if selected_question:

            active_conversation["messages"].append(
                {
                    "role": "user",
                    "content": selected_question
                }
            )

            if len(active_conversation["messages"]) == 1:
                title = selected_question.strip()

                if len(title) > 30:
                    title = title[:30] + "..."

                active_conversation["title"] = title

            with st.chat_message("user"):
                st.markdown(selected_question)

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching the video..."
                ):

                    try:

                        answer = (
                            active_conversation["rag_chain"]
                            .invoke(selected_question)
                        )

                        st.markdown(answer)

                        active_conversation[
                            "messages"
                        ].append(
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        )

                    except Exception as e:

                        st.error(str(e))

            st.rerun()

        for message in active_conversation["messages"]:

            with st.chat_message(
                message["role"]
            ):
                st.markdown(
                    message["content"]
                )