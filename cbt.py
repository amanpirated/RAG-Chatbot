#---------------Reading pdf-------------------
from pypdf import PdfReader
reader=PdfReader("C:\\Users\\Admin\\OneDrive\\Desktop\\chatbot\\data\\notes.pdf")
text=""

for page in reader.pages:
    text+=page.extract_text()
print(" PDF loaded successfully")

#-----------------CHUNKING---------------------
def chunkText(text,chunk_size=500):
    chunks=[]
    for i in range(0,len(text),chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks
chunks=chunkText(text)

#-----------------Embedding--------------------
from sentence_transformers import SentenceTransformer

model=SentenceTransformer("all-MiniLM-l6-v2")
embeddings=model.encode(chunks)

#--------------Storing Embeddings----------------
import faiss
import numpy as np

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

#-----------Retrieving Relevant Chunks-------------
query = "What is supervised learning?"
query_embedding = model.encode([query])

distances, indices = index.search(
    np.array(query_embedding),
    k=3
)
retrieved_chunks = [
    chunks[i]
    for i in indices[0]
]

#-----------------Sending Comtext to Gemini----------------
from google import genai
client=genai.Client(api_key="AQ.Ab8RN6JvuxPiNxYeZR96Ws0Fbmw9u6e6YN7NJ48xjOQuvtDmyg")

context='/n'.join(retrieved_chunks)
prompt=f""" 
Answer only from the context. 
Context: {context} 
Question: {query}
"""

#-----------------Generating Response------------------
response=client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)
print(response.text)

#----------------Chat loop-----------------------
while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        k=3
    )

    context = "\n".join(
        chunks[i]
        for i in indices[0]
    )

    prompt = f"""
    Context:
    {context}

    Question:
    {query}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("Bot:", response.text)