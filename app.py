import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai

# ==========================
# CONFIG
# ==========================

GEMINI_API_KEY = "AQ.Ab8RN6Kkt-obbkOEVRTt-sGyeNpslzVhKfhixiQy8m45wacONA"

client = genai.Client(api_key=GEMINI_API_KEY)

# ==========================
# PDF LOADING
# ==========================

@st.cache_resource
def load_rag_system():

    reader = PdfReader("data/notes.pdf")

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # Chunking
    chunks = []

    chunk_size = 500

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    # Embedding Model
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = model.encode(chunks)

    # FAISS Index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(
            embeddings,
            dtype=np.float32
        )
    )

    return model, index, chunks


embedding_model, index, chunks = load_rag_system()

# ==========================
# RAG FUNCTION
# ==========================

def ask_rag(question):

    query_embedding = embedding_model.encode([question])

    distances, indices = index.search(
        np.array(query_embedding, dtype=np.float32),
        k=3
    )

    retrieved_chunks = []

    for idx in indices[0]:
        retrieved_chunks.append(chunks[idx])

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a helpful AI assistant.

Answer only from the provided context.

If the answer is not available in the context,
say:
"I could not find that information in the document."

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# ==========================
# STREAMLIT UI
# ==========================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 PDF RAG Chatbot")
st.write("Ask questions from your PDF.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_prompt = st.chat_input(
    "Ask a question..."
)

if user_prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = ask_rag(user_prompt)

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )