import os
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    RequestBlocked,
    IpBlocked,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from langchain_core.output_parsers import StrOutputParser


# 1. ENVIRONMENT

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "\n❌ GEMINI_API_KEY not found.\n"
        f"Expected .env file at:\n{ENV_FILE}\n\n"
        "Your .env should contain:\n"
        "GEMINI_API_KEY=your_api_key_here"
    )


# 2. CONFIGURATION

# FAISS indexes will be stored here.
VECTOR_DB_DIR = BASE_DIR / "vectorstores"
VECTOR_DB_DIR.mkdir(exist_ok=True)

# Gemini free-tier friendly embedding settings.
EMBED_BATCH_SIZE = 10
EMBED_BATCH_DELAY = 8
EMBED_RETRY_DELAY = 60

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4


# 3. GEMINI MODELS

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=GEMINI_API_KEY,
)


# 4. YOUTUBE VIDEO ID

def extract_video_id(url: str) -> str:
    """
    Extract YouTube video ID from common URL formats.
    """

    parsed_url = urlparse(url)

    hostname = parsed_url.hostname

    if hostname:
        hostname = hostname.lower()

    # Standard YouTube URLs
    if hostname in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
    }:

        query_params = parse_qs(
            parsed_url.query
        )

        video_ids = query_params.get("v")

        if video_ids:
            return video_ids[0]

    # Short YouTube URLs
    if hostname == "youtu.be":

        video_id = parsed_url.path.strip("/")

        if video_id:
            return video_id

    raise ValueError(
        "Invalid YouTube URL.\n"
        "Example:\n"
        "https://www.youtube.com/watch?v=VIDEO_ID"
    )


# 5. GET TRANSCRIPT

def get_transcript(video_id: str) -> str:
    """
    Fetch English transcript and convert it to one string.
    """

    api = YouTubeTranscriptApi()

    try:

        transcript_data = api.fetch(
            video_id,
            languages=["en"],
        )

        transcript = " ".join(
            snippet.text
            for snippet in transcript_data
        )

        if not transcript.strip():

            raise RuntimeError(
                "Transcript is empty."
            )

        return transcript.strip()

    except TranscriptsDisabled:

        raise RuntimeError(
            "This video has captions disabled."
        )

    except NoTranscriptFound:

        raise RuntimeError(
            "No English transcript was found."
        )

    except VideoUnavailable:

        raise RuntimeError(
            "This YouTube video is unavailable."
        )

    except (RequestBlocked, IpBlocked):

        raise RuntimeError(
            "YouTube blocked the transcript request. "
            "This is a YouTube IP/network restriction."
        )


# 6. TEXT SPLITTING

def split_transcript(transcript: str):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.create_documents(
        [transcript]
    )


# 7. RATE-LIMIT SAFE GEMINI EMBEDDING

def generate_embeddings(documents):
    """
    Generate Gemini embeddings in controlled batches.

    This prevents the application from sending all
    transcript chunks at once.
    """

    total = len(documents)

    all_vectors = []

    for start in range(
        0,
        total,
        EMBED_BATCH_SIZE
    ):

        end = min(
            start + EMBED_BATCH_SIZE,
            total
        )

        batch = documents[start:end]

        print(
            f"   Embedding chunks "
            f"{start + 1}-{end}/{total}"
        )

        while True:

            try:

                vectors = embeddings.embed_documents(
                    [
                        doc.page_content
                        for doc in batch
                    ]
                )

                all_vectors.extend(
                    vectors
                )

                break

            except Exception as e:

                error_message = str(e)

                if (
                    "429" in error_message
                    or "RESOURCE_EXHAUSTED"
                    in error_message
                ):

                    print(
                        "   ⚠️ Gemini embedding quota "
                        "temporarily reached."
                    )

                    print(
                        f"   Waiting "
                        f"{EMBED_RETRY_DELAY} seconds..."
                    )

                    time.sleep(
                        EMBED_RETRY_DELAY
                    )

                else:

                    raise

        # Avoid hitting the free-tier limit.
        if end < total:

            time.sleep(
                EMBED_BATCH_DELAY
            )

    return all_vectors



# 8. FAISS CACHE

def get_vector_db_path(video_id: str) -> Path:

    return VECTOR_DB_DIR / video_id


def index_exists(video_id: str) -> bool:

    db_path = get_vector_db_path(
        video_id
    )

    return (
        (db_path / "index.faiss").exists()
        and
        (db_path / "index.pkl").exists()
    )


def create_vector_store(
    documents,
    video_id: str,
):

    vectors = generate_embeddings(
        documents
    )

    text_embeddings = [
        (
            documents[i].page_content,
            vectors[i]
        )
        for i in range(
            len(documents)
        )
    ]

    metadatas = [
        documents[i].metadata
        for i in range(
            len(documents)
        )
    ]

    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings,
        metadatas=metadatas,
    )

    db_path = get_vector_db_path(
        video_id
    )

    db_path.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(
        str(db_path)
    )

    print(
        "✅ FAISS index saved."
    )

    return vector_store


def load_vector_store(video_id: str):

    db_path = get_vector_db_path(
        video_id
    )

    return FAISS.load_local(
        str(db_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# 9. INDEXING CHAIN FUNCTIONS

def prepare_video(url: str):

    video_id = extract_video_id(
        url
    )

    return {
        "url": url,
        "video_id": video_id,
    }


def fetch_video_transcript(data):

    video_id = data["video_id"]

    print(
        f"\n🎥 Video ID: {video_id}"
    )

    # IMPORTANT:
    # Don't embed the same video again.

    if index_exists(video_id):

        print(
            "✅ Existing FAISS index found."
        )

        return {
            **data,
            "cached": True,
        }

    print(
        "\n📥 Fetching transcript..."
    )

    transcript = get_transcript(
        video_id
    )

    print(
        f"✅ Transcript loaded: "
        f"{len(transcript):,} characters"
    )

    return {
        **data,
        "transcript": transcript,
        "cached": False,
    }


def split_video_transcript(data):

    # If cached, no need to split again.
    if data["cached"]:

        return data

    print(
        "\n✂️ Splitting transcript..."
    )

    documents = split_transcript(
        data["transcript"]
    )

    print(
        f"✅ Created "
        f"{len(documents)} chunks."
    )

    return {
        **data,
        "documents": documents,
    }


def build_video_index(data):

    video_id = data["video_id"]

    # Load existing index

    if data["cached"]:

        print(
            "\n📦 Loading existing FAISS index..."
        )

        vector_store = load_vector_store(
            video_id
        )

        print(
            "✅ Existing vector store loaded."
        )

        return {
            **data,
            "vector_store": vector_store,
        }

    # Create new index

    documents = data["documents"]

    print(
        "\n🧠 Creating Gemini embeddings..."
    )

    vector_store = create_vector_store(
        documents,
        video_id,
    )

    print(
        "✅ Vector store created."
    )

    return {
        **data,
        "vector_store": vector_store,
    }


# 10. COMPLETE INDEXING CHAIN

indexing_chain = (
    RunnableLambda(prepare_video)
    | RunnableLambda(fetch_video_transcript)
    | RunnableLambda(split_video_transcript)
    | RunnableLambda(build_video_index)
)


# 11. RETRIEVER

def create_retriever(vector_store):

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": TOP_K
        },
    )


def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# 12. RAG PROMPT

prompt = PromptTemplate(
    template="""
You are a helpful assistant answering questions
about a YouTube video.

Answer ONLY using the provided transcript context.

Do NOT use outside knowledge.

If the answer cannot be found in the transcript,
say exactly:

"I don't know based on this video."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=[
        "context",
        "question",
    ],
)


# 13. CLEAN GEMINI OUTPUT

def clean_response(content):
    """
    Gemini may return content as a string or
    a list of content blocks.

    Convert everything to readable text.
    """

    if isinstance(
        content,
        str
    ):

        return content.strip()

    if isinstance(
        content,
        list
    ):

        text_parts = []

        for block in content:

            if isinstance(
                block,
                dict
            ):

                text = (
                    block.get("text")
                    or block.get("_text")
                    or block.get("content")
                )

                if text:

                    text_parts.append(
                        str(text)
                    )

            elif isinstance(
                block,
                str
            ):

                text_parts.append(
                    block
                )

        return "\n".join(
            text_parts
        ).strip()

    return str(content).strip()


# 14. COMPLETE RAG CHAIN

def build_rag_chain(retriever):

    rag_chain = (
        RunnableParallel(
            {
                "context": (
                    retriever
                    | RunnableLambda(format_docs)
                ),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(clean_response)
    )

    return rag_chain


# 15. MAIN APPLICATION

def main():

    print("\n")
    print("=" * 70)
    print("                 🎥 VIDEO RAG CHAT")
    print("=" * 70)

    # Get YouTube URL

    url = input(
        "\nPaste YouTube video URL:\n> "
    ).strip()

    if not url:

        print(
            "\n❌ URL cannot be empty."
        )

        return

    try:

        # INDEXING CHAIN

        print(
            "\n🚀 Running indexing chain..."
        )

        indexed_data = indexing_chain.invoke(
            url
        )

        vector_store = indexed_data[
            "vector_store"
        ]

        # RETRIEVER

        retriever = create_retriever(
            vector_store
        )

        print(
            "\n✅ Retriever ready."
        )

        # RAG CHAIN

        rag_chain = build_rag_chain(
            retriever
        )

        print(
            "\n✅ RAG chain created."
        )

        # READY

        print("\n")
        print("=" * 70)
        print("                 ✅ VIDEO READY")
        print("=" * 70)

        print(
            "\nAsk anything about this video."
        )

        print(
            "Type 'exit' to close."
        )

        # CHAT LOOP

        while True:

            question = input(
                "\nYou:\n> "
            ).strip()

            if not question:

                continue

            if question.lower() == "exit":

                print(
                    "\n👋 Goodbye!"
                )

                break

            try:

                print(
                    "\n🔎 Retrieving relevant context..."
                )

                answer = rag_chain.invoke(
                    question
                )

                print(
                    "\n🤖 Assistant:\n"
                )

                print(
                    answer
                )

            except Exception as e:

                print(
                    "\n❌ Error while answering:"
                )

                print(
                    str(e)
                )

    except Exception as e:

        print(
            "\n❌ Application Error:\n"
        )

        print(
            str(e)
        )


# 16. RUN

if __name__ == "__main__":
    main()