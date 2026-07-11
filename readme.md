# Tredence Website Q&A Chatbot (RAG + Visualization)

This project is an end-to-end Retrieval-Augmented Generation (RAG) pipeline that powers a Q&A chatbot over the **Tredence** homepage. It is implemented entirely into Jupyter notebooks and uses:

- `requests` + `BeautifulSoup` for web scraping  
- `OpenAI` embeddings and chat completions  
- `Chroma` as a local vector store  
- `Gradio` for a minimal chat UI  
- `UMAP` + `matplotlib` for visualizing the embedding space and retrieved chunks  

The core notebooks are:

- `RAG_WEBSITE_CONTENT.ipynb` – scraping, RAG pipeline, and chatbot  
- `ADVANCED_RAG_VISUALIZATION_PIPELINE.ipynb` – exploratory visualization patterns adapted to this project  

---

## 1. Project Overview

The goal is to answer questions about the Tredence website using **only** content scraped from the homepage.

High-level steps:

1. Scrape the Tredence homepage (`https://www.tredence.com/`) and extract textual content.  
2. Chunk the page into overlapping segments suitable for embeddings.  
3. Embed chunks using OpenAI embeddings and store them in a Chroma collection.  
4. For a user question:
   - Embed the question  
   - Retrieve the top-k most similar chunks from Chroma  
   - Feed those chunks as context into an OpenAI chat model  
   - Return an answer grounded in the retrieved content  
5. Visualize the embedding space using UMAP to inspect how queries and retrieved chunks are positioned relative to the corpus.  

---

## 2. Tech-Stack

- **Language:** Python 3.10+  
- **Scraping:** `requests`, `beautifulsoup4`  
- **RAG / Vector Store:**  
  - `openai` (embeddings + chat completions)  
  - `chromadb` (PersistentClient / local DuckDB + Parquet)  
- **UI:** `gradio` (ChatInterface)  
- **Visualization:** `umap-learn`, `numpy`, `matplotlib`, `pandas`  

---

## 3. Setup

### 3.1. Clone and Environment

```bash
git clone <your-repo-url>
cd <your-repo-dir>

python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, a minimal set is:

```text
requests
beautifulsoup4
pandas
chromadb
openai
gradio
umap-learn
matplotlib
numpy
```

### 3.2. OpenAI API Key

Set your OpenAI API key as an environment variable or directly in the notebook:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

or inside the notebook (for local dev only):

```python
import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
```

The notebooks assume:

- Embeddings: `text-embedding-3-small`  
- Chat model: `gpt-4o-mini`  

You can change these in the cells where `client.embeddings.create` and `client.chat.completions.create` are called.

---

## 4. RAG Pipeline (Notebook: `RAG_WEBSITE_CONTENT.ipynb`)

### 4.1. Scraping and Corpus Creation

1. Use `requests` to fetch the Tredence homepage.
2. Use `BeautifulSoup` to:
   - Try `soup.find("main")` for main content.
   - Fallback to concatenating all `<p>` tags if `<main>` is not present.
3. Store the extracted text in a DataFrame:

```python
docs_df = pd.DataFrame([{"url": url, "text": extracted_text}])
```

### 4.2. Chunking

A simple character-based chunker with overlap is used:

- `CHUNK_SIZE` ~ 1000 characters  
- `CHUNK_OVERLAP` ~ 200 characters  

This produces a Python list `chunks`, each representing a text segment from the homepage.

### 4.3. Embeddings + Chroma Vector Store

1. Initialize a Chroma `PersistentClient`:

```python
from chromadb import PersistentClient

chroma_client = PersistentClient(path="./tredence_rag_db")
collection = chroma_client.get_or_create_collection(name="tredence_website")
```

2. Create embeddings for all `chunks` with OpenAI:

```python
chunk_embeddings = embed_texts(chunks)  # wraps client.embeddings.create(...)
```

3. Add chunks to the collection with metadata:

```python
collection.add(
    ids=[f"tredence_chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=chunk_embeddings,
    metadatas=[{"url": docs_df.loc[0, "url"], "chunk_index": i} for i in range(len(chunks))]
)
```

---

## 5. Question Answering

The main RAG logic is implemented via three functions:

- `retrieve_relevant_chunks(question, top_k)`  
- `build_context(retrieved_chunks)`  
- `answer_question(question)`

**Flow:**

1. Embed the question with OpenAI.
2. `collection.query` with `query_embeddings` to get top-k similar chunks.
3. Build a combined context string from the retrieved chunks.
4. Call `client.chat.completions.create` with a system prompt that enforces grounding in the context.
5. Return the model’s answer.

Example:

```python
test_question = "What services does Tredence provide?"
print(answer_question(test_question))
```

---

## 6. Gradio Chat UI

A minimal Gradio UI wraps `answer_question`:

```python
import gradio as gr

def rag_chatbot(message, history):
    if not message or message.strip() == "":
        return "Please ask a question about the Tredence website."
    return answer_question(message)

demo = gr.ChatInterface(
    fn=rag_chatbot,
    title="Tredence Website Q&A",
    description="Ask questions about the Tredence homepage (RAG over scraped content)."
)

demo.launch()
```

This gives a browser-based chat interface where each user message is answered using the RAG pipeline.

---

## 7. Embedding Space Visualization


1. Retrieves all embeddings, metadata, and documents from Chroma.  
2. Fits a UMAP model on the full embedding matrix.  
3. For a given query:
   - Embeds the query  
   - Queries Chroma for the top-k chunks  
   - Projects:
     - All corpus embeddings  
     - Retrieved chunk embeddings  
     - The query embedding  
   - Plots them in 2D with `matplotlib`:
     - Grey points: entire corpus (colored by chunk bucket)  
     - Lime outlines: retrieved chunks  
     - Red X: query  

This helps debug and interpret retrieval by showing where the query and retrieved chunks sit in the embedding space.

---

## 8. Extending the Project

Potential next steps:

- **Add more pages**: Extend scraping to multiple Tredence URLs and index them in the same collection.  
- **Richer metadata**: Tag chunks by section (e.g., services, industries) and use metadata filters in Chroma.  
- **Logging & analytics**: Log questions and retrieved chunk IDs for later analysis and clustering.  
- **Evaluation**: Build a small labeled set of Q&A pairs and measure retrieval and answer quality.

---

## 9. Running the Notebooks

1. Start Jupyter or VS Code (with Python extension) in the project directory.  
2. Open `RAG_WEBSITE_CONTENT.ipynb`.  
3. Run cells in order:
   - Scraping + corpus creation  
   - Chunking  
   - Embeddings + Chroma  
   - RAG Q&A functions  
   - Gradio UI  
---
