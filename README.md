# PDF RAG Chatbot using Gemini API

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions from a PDF document using Google's Gemini API. Instead of relying solely on the language model's knowledge, the chatbot retrieves relevant information from the uploaded PDF and uses it as context to generate accurate responses.

The project demonstrates the complete RAG pipeline, including document loading, text chunking, embedding generation, vector search using FAISS, and response generation using Gemini.

---

## Features

* Extracts text from PDF documents
* Splits documents into manageable chunks
* Generates semantic embeddings using Sentence Transformers
* Stores embeddings in a FAISS vector database
* Retrieves the most relevant chunks based on user queries
* Uses Gemini API to generate context-aware answers
* Command-line chat interface
* Beginner-friendly implementation without LangChain

---

## Tech Stack

* Python
* Gemini API
* Sentence Transformers
* FAISS
* PyPDF
* NumPy

---

## Project Structure

```text
chatbot/
│
├── data/
│   └── notes.pdf
│
├── rag_chatbot.py
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd chatbot
```

Install dependencies:

```bash
pip install google-genai
pip install sentence-transformers
pip install faiss-cpu
pip install pypdf
pip install numpy
```

---

## Getting a Gemini API Key

1. Visit Google AI Studio.
2. Create an API key.
3. Copy the API key.
4. Replace the placeholder value in the code:

```python
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

---

## How It Works

### Step 1: PDF Loading

The chatbot reads text from the PDF using PyPDF.

### Step 2: Text Chunking

The extracted text is divided into smaller chunks to improve retrieval accuracy.

### Step 3: Embedding Generation

Sentence Transformers converts each chunk into a numerical vector representation.

### Step 4: Vector Storage

All embeddings are stored in a FAISS index for efficient similarity search.

### Step 5: Retrieval

When a user asks a question:

* The query is converted into an embedding.
* FAISS finds the most relevant chunks.
* Retrieved chunks are combined into a context.

### Step 6: Response Generation

The context and question are sent to Gemini, which generates the final answer.

---

## Running the Chatbot

Place your PDF file inside the data folder:

```text
data/notes.pdf
```

Run the chatbot:

```bash
python rag_chatbot.py
```

Example:

```text
You: What is machine learning?

Bot: Machine learning is a branch of artificial intelligence that enables systems to learn from data.
```

Exit the chatbot:

```text
exit
```

---

## Example Workflow

```text
User Question
      │
      ▼
Query Embedding
      │
      ▼
FAISS Similarity Search
      │
      ▼
Relevant Chunks Retrieved
      │
      ▼
Gemini API
      │
      ▼
Generated Response
```

---

## Future Improvements

* Streamlit Web Interface
* Multiple PDF Support
* ChromaDB Integration
* Conversation Memory
* Hybrid Search
* Metadata Filtering
* LangChain Integration
* Agentic RAG Workflows

---

## Learning Outcomes

Through this project, you will understand:

* Retrieval-Augmented Generation (RAG)
* Text Chunking Strategies
* Embeddings and Semantic Search
* Vector Databases
* FAISS Similarity Search
* Prompt Engineering
* Gemini API Integration

---

## Author

Suminder Singh

B.Tech Computer Science Engineering

Data Analytics | Machine Learning | Generative AI Enthusiast
