

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run app.py
```

The app will be accessible at `http://localhost:8501`.

## Usage Instructions

### Step 1: Upload Data
Upload a CSV, JSON, PDF, or DOCX file. You can specify which columns to index for vector-based search.

### Step 2: Embedding and Storage
The data is stored in **Chroma**, and embeddings are generated using models like `all-MiniLM-L6-v2` (for English) or `keepitreal/vietnamese-sbert` (for Vietnamese).

### Step 3: Choose LLM
Select from:
- **Gemini API** (requires a [Gemini API key](https://aistudio.google.com/app/apikey))

### Step 4: Configure Chunking
Select a chunking method to organize the content:
- **Recursive Token Chunking**: Divide text based on token count.
- **Semantic Chunking**: Group text semantically.

### Step 5: Interact with the Chatbot
Start chatting with the bot, which will enhance responses using the retrieved content.

