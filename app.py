import streamlit as st
import pandas as pd
import json
import uuid  # For generating unique IDs
import os
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from IPython.display import Markdown
from chunking import ProtonxSemanticChunker
from utils import process_batch, divide_dataframe, clean_collection_name
from search import vector_search, hyde_search
from llms.onlinellms import OnlineLLMs
import time
import pdfplumber  # PDF extraction
import io
from docx import Document  # DOCX extraction
from components import notify
from constant import  VI, USER, ASSISTANT, VIETNAMESE, ONLINE_LLM,  GEMINI, DB
from collection_management import list_collection
from dotenv import load_dotenv

load_dotenv()

def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]

llm_options = {
    "Online": "Online"
}

# Tiêu đề chính
st.markdown(
    """
    <h1 style='display: flex; align-items: center;'>
        <img src="https://tuyensinh.uit.edu.vn/sites/default/files/uploads/images/uit_footer.png" width="50" style='margin-right: 10px'>
        UIT Admissions Chatbot 🎓
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown("Welcome to the UIT Admissions Chatbot!❓❓❓ Discover all the information you need about admissions, 📚programs, 💸scholarships, 🌟Student Life at UIT and more with us.")

if "language" not in st.session_state:
    st.session_state.language = VIETNAMESE  
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "llm_model" not in st.session_state:
    st.session_state.llm_model = None

if "client" not in st.session_state:
    st.session_state.client = chromadb.PersistentClient("db")

if "collection" not in st.session_state:
    st.session_state.collection = None

if "search_option" not in st.session_state:
    st.session_state.search_option = "Vector Search"

if "open_dialog" not in st.session_state:
    st.session_state.open_dialog = None

if "source_data" not in st.session_state:
    st.session_state.source_data = "UPLOAD"

if "chunks_df" not in st.session_state:
    st.session_state.chunks_df = pd.DataFrame()

if "random_collection_name" not in st.session_state:
    st.session_state.random_collection_name = None

# --- End of initialization

# Sidebar settings
st.sidebar.header("Settings")

st.session_state.number_docs_retrieval = st.sidebar.number_input(
    "Number of documnents retrieval", 
    min_value=1, 
    max_value=50,
    value=15,
    step=1,
    help="Set the number of document which will be retrieved."
)

language_choice = "Vietnamese"
if language_choice == VIETNAMESE:
    if st.session_state.get("language") != VI:
        st.session_state.language = VI
        # Only load the model if it hasn't been loaded before
        if st.session_state.get("embedding_model_name") != 'keepitreal/vietnamese-sbert':
            st.session_state.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
            st.session_state.embedding_model_name = 'keepitreal/vietnamese-sbert'
        st.success("Using Vietnamese embedding model: keepitreal/vietnamese-sbert")

# Step 1: File Upload (CSV, JSON, PDF, or DOCX) and Column Detection

header_i = 1
st.header(f"{header_i}. Setup data source")
st.subheader(f"{header_i}.1. Upload data (Upload CSV, JSON, PDF, or DOCX files)", divider=True)
uploaded_files = st.file_uploader(
    "", 
    accept_multiple_files=True
)

# Initialize a variable for tracking the success of saving the data
st.session_state.data_saved_success = False

if uploaded_files is not None:
    all_data = []
    
    for uploaded_file in uploaded_files:
        print(uploaded_file.type)
        # Determine file type and read accordingly
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            all_data.append(df)

        elif uploaded_file.name.endswith(".json"):
            json_data = json.load(uploaded_file)
            df = pd.json_normalize(json_data)  # Normalize JSON to a flat DataFrame format
            all_data.append(df)

        elif uploaded_file.name.endswith(".pdf"):
            # Extract text from PDF
            pdf_text = []
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                for page in pdf.pages:
                    pdf_text.append(page.extract_text())

            # Convert PDF text into a DataFrame (assuming one column for simplicity)
            df = pd.DataFrame({"content": pdf_text})
            all_data.append(df)

        elif uploaded_file.name.endswith(".docx") or uploaded_file.name.endswith(".doc"):
            # Extract text from DOCX
            doc = Document(io.BytesIO(uploaded_file.read()))
            docx_text = [para.text for para in doc.paragraphs if para.text]

            # Convert DOCX text into a DataFrame (assuming one column for simplicity)
            df = pd.DataFrame({"content": docx_text})
            all_data.append(df)

# Concatenate all data into a single DataFrame
if all_data:
    df = pd.concat(all_data, ignore_index=True)
    st.dataframe(df)

    doc_ids = [str(uuid.uuid4()) for _ in range(len(df))]

    if "doc_ids" not in st.session_state:
        st.session_state.doc_ids = doc_ids

    # Add or replace the '_id' column in the DataFrame+
    df['doc_id'] = st.session_state.doc_ids

    st.subheader("Chunking")

    # **Ensure `df` is not empty before calling selectbox**
    if not df.empty:
        # Display selectbox to choose the column for vector search
        index_column = st.selectbox("Choose the column to index (for vector search):", df.columns)
        st.write(f"Selected column for indexing: {index_column}")
    else:
        st.warning("The DataFrame is empty, please upload valid data.")
            
    chunk_options = ["SemanticChunker"]

    # Step 4: Chunking options
    if not st.session_state.get("chunkOption"):
        st.session_state.chunkOption = "SemanticChunker" 

    currentChunkerIdx = 0
    
    st.radio(
        "",
        chunk_options,
        captions=[
            "Chunking with semantic comparison between chunks",
        ],
        key="chunkOption",
        index=currentChunkerIdx
    )
    # Lấy lựa chọn phương pháp chia nhỏ
    chunkOption = st.session_state.get("chunkOption")
    
    if chunkOption == "SemanticChunker":
        embedding_option = "TF-IDF"
    chunk_records = []

    # Iterate over rows in the original DataFrame
    for index, row in df.iterrows():
        chunker = None
        selected_column_value = row[index_column]
        chunks = []
        if not (type(selected_column_value) == str and len(selected_column_value) > 0):
            continue
        
        if chunkOption == "SemanticChunker":
            if embedding_option == "TF-IDF":
                chunker = ProtonxSemanticChunker(
                    embedding_type="tfidf",
                )
            chunks = chunker.split_text(selected_column_value)
        
        # For each chunk, add a dictionary with the chunk and original_id to the list
        for chunk in chunks:
            chunk_record = {**row.to_dict(), 'chunk': chunk}
            
            # Rearrange the dictionary to ensure 'chunk' and '_id' come first
            chunk_record = {
                'chunk': chunk_record['chunk'],
                **{k: v for k, v in chunk_record.items() if k not in ['chunk', '_id']}
            }
            chunk_records.append(chunk_record)

    # Convert the list of dictionaries to a DataFrame
    st.session_state.chunks_df = pd.DataFrame(chunk_records)

if "chunks_df" in st.session_state and len(st.session_state.chunks_df) > 0:
    # Display the result
    st.write("Number of chunks:", len(st.session_state.chunks_df))
    st.dataframe(st.session_state.chunks_df)

# Button to save data
if st.button("Save Data"):
    try:
        # Check if the collection exists, if not, create a new one
        if st.session_state.collection is None:
            if uploaded_files:
                first_file_name = os.path.splitext(uploaded_files[0].name)[0]  # Get file name without extension
                collection_name = f"rag_collection_{clean_collection_name(first_file_name)}"
            else:
                # If no file name is available, generate a random collection name
                collection_name = f"rag_collection_{uuid.uuid4().hex[:8]}"
        
            st.session_state.random_collection_name = collection_name
            st.session_state.collection = st.session_state.client.get_or_create_collection(
                name=st.session_state.random_collection_name,
                metadata={"description": "A collection for RAG system"},
            )

        # Define the batch size
        batch_size = 256

        # Split the DataFrame into smaller batches
        df_batches = divide_dataframe(st.session_state.chunks_df, batch_size)

        # Check if the dataframe has data, otherwise show a warning and skip the processing
        if not df_batches:
            st.warning("No data available to process.")
        else:
            num_batches = len(df_batches)

            # Initialize progress bar
            progress_text = "Saving data to Chroma. Please wait..."
            my_bar = st.progress(0, text=progress_text)

            # Process each batch
            for i, batch_df in enumerate(df_batches):
                if batch_df.empty:
                    continue  # Skip empty batches (just in case)
                
                process_batch(batch_df, st.session_state.embedding_model, st.session_state.collection)

                # Update progress dynamically for each batch
                progress_percentage = int(((i + 1) / num_batches) * 100)
                my_bar.progress(progress_percentage, text=f"Processing batch {i + 1}/{num_batches}")

                time.sleep(0.1)  # Optional sleep to simulate processing time

            # Empty the progress bar once completed
            my_bar.empty()

            st.success("Data saved to Chroma vector store successfully!")
            st.markdown("Collection name: `{}`".format(st.session_state.random_collection_name))
            st.session_state.data_saved_success = True

    except Exception as e:
        st.error(f"Error saving data to Chroma: {str(e)}")

# Set up the interface
st.subheader(f"{header_i}.2. Or load from saved collection", divider=True)
if st.button("Load from saved collection"):
    st.session_state.open_dialog = "LIST_COLLECTION"
    def load_func(collection_name):
        st.session_state.collection = st.session_state.client.get_collection(
            name=collection_name
        )
        st.session_state.random_collection_name = collection_name
        st.session_state.data_saved_success = True
        st.session_state.source_data = DB
        data = st.session_state.collection.get(
            include=[
                "documents", 
                "metadatas"
            ],
        )
        metadatas = data["metadatas"]
        column_names = []
        if len(metadatas) > 0 and len(metadatas[0].keys()) > 0:
            column_names.extend(metadatas[0].keys())
            column_names = list(set(column_names))

        st.session_state.chunks_df = pd.DataFrame(metadatas, columns=column_names)

    def delete_func(collection_name):
        st.session_state.client.delete_collection(name=collection_name)
    
    list_collection(st.session_state, load_func, delete_func)
        

header_i += 1
header_text = "{}. Setup data ✅".format(header_i) if st.session_state.data_saved_success else "{}. Setup data".format(header_i)
st.header(header_text)

if st.session_state.data_saved_success:
    st.markdown("✅ **Data Saved Successfully!**")

# Step 3: Define which columns LLMs should answer from
if "random_collection_name" in st.session_state and st.session_state.random_collection_name is not None and st.session_state.chunks_df is not None:
    st.session_state.columns_to_answer = st.multiselect(
        "Select one or more columns LLMs should answer from (multiple selections allowed):", 
        st.session_state.chunks_df.columns
    )

# Step 2: Setup LLMs (Gemini Only)
header_i += 1
header_text_llm = "{}. Setup LLMs".format(header_i)
st.header(header_text_llm)

# Example user selection (remove local options)
st.session_state.llm_type = ONLINE_LLM
st.session_state.llm_name = GEMINI

# Input API key for Gemini
st.markdown("Obtain the API key from the [Google AI Studio](https://aistudio.google.com/app/apikey).")
st.text_input(
    "Enter your API Key:", 
    type="password", 
    key="llm_api_key",
    value=os.getenv("GEMINI_API_KEY")
)

if st.session_state.get('llm_api_key'):
    st.success("API Key saved successfully!")
    st.session_state.llm_model = OnlineLLMs(
        name=GEMINI,
        api_key=st.session_state.get('llm_api_key'),
        model_version="gemini-1.5-pro"
    )
    st.markdown("✅ **API Key Saved Successfully!**")

st.sidebar.markdown(f"1. LLM model: **{st.session_state.llm_name if 'llm_name' in st.session_state else 'Not selected'}**")
st.sidebar.markdown(f"2. Language: **{st.session_state.language}**")
st.sidebar.markdown(f"3. Embedding Model: **{st.session_state.embedding_model.__class__.__name__ if st.session_state.embedding_model else 'None'}**")
st.sidebar.markdown(f"4. Number of Documents Retrieval: **{st.session_state.number_docs_retrieval}**")
if st.session_state.get('chunkOption'):
    st.sidebar.markdown(f". Chunking Option: **{st.session_state.chunkOption}**")

header_i += 1
header_text_llm = "{}. Set up search algorithms".format(header_i)
st.header(header_text_llm)

st.radio(
    "Please select one of the options below.",
    [
        # "Keywords Search", 
        "Vector Search", 
        "Hyde Search"],
    captions = [
        # "Search using traditional keyword matching",
        "Search using vector similarity",
        "Search using the HYDE algorithm"
    ],
    key="search_option",
    index=0,
)

# Step 4: Interactive Chatbot
header_i += 1
st.header("{}. Interactive Chatbot".format(header_i))

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# URL of the Flask API

# Display the chat history using chat UI
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("How can I assist you today?"):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": USER, "content": prompt})
    # Display user message in chat message container
    with st.chat_message(USER):
        st.markdown(prompt)
    # Display assistant response in chat message container
    # Prepare the payload for the request

    with st.chat_message(ASSISTANT):
        if st.session_state.collection is not None:
            # Combine retrieved data to enhance the prompt based on selected columns
            metadatas, retrieved_data = [], ""
            if st.session_state.columns_to_answer:
                if st.session_state.search_option == "Vector Search":
                    metadatas, retrieved_data = vector_search(
                        st.session_state.embedding_model, 
                        prompt, 
                        st.session_state.collection, 
                        st.session_state.columns_to_answer,
                        st.session_state.number_docs_retrieval
                    )
                    #retrieved_data,
                    enhanced_prompt = """
                    The user prompt is: "{}". 
                    You are a chatbot designed to answer questions related to admissions at UIT (University of Information Technology). 
                    If the user greets, respond only with a friendly greeting and introduce yourself as UIT Chatbot. 
                    Otherwise, use the provided retrieved data below to answer the user's question in a friendly and helpful manner. 
                    Your responses must be accurate, detailed, and based on the retrieved data: \n{}""".format(prompt, retrieved_data)

                elif st.session_state.search_option == "Hyde Search":
              
                    if st.session_state.llm_type == ONLINE_LLM:
                        model = st.session_state.llm_model
                    else:
                        model = st.session_state.local_llms

                    metadatas, retrieved_data = hyde_search(
                        model,
                        st.session_state.embedding_model,
                        prompt,
                        st.session_state.collection,
                        st.session_state.columns_to_answer,
                        st.session_state.number_docs_retrieval,
                        num_samples=1
                    )

                    enhanced_prompt = """
                    The user prompt is: "{}". 
                    You are a chatbot designed to answer questions related to admissions at UIT (University of Information Technology). 
                    If the user greets, respond only with a friendly greeting and introduce yourself as UIT Chatbot. 
                    Otherwise, use the provided retrieved data below to answer the user's question in a friendly and helpful manner. 
                    Your responses must be accurate, detailed, and based on the retrieved data: \n{}""".format(prompt, retrieved_data)

# Đã có prompt, retrieved_data -> metadata{answer, question, chunk }
                
                if metadatas:
                    flattened_metadatas = [item for sublist in metadatas for item in sublist]  # Flatten the list of lists
                    
                    # Convert the flattened list of dictionaries to a DataFrame
                    metadata_df = pd.DataFrame(flattened_metadatas)
                    
                    # Display the DataFrame in the sidebar
                 
                    st.sidebar.subheader("Retrieval data")
                    st.sidebar.dataframe(metadata_df)
                    st.sidebar.subheader("Full prompt for LLM")
                    st.sidebar.markdown(enhanced_prompt)
                else:
                    st.sidebar.write("No metadata to display.")

                if st.session_state.llm_type == ONLINE_LLM:
                    # Generate content using the selected LLM model
                    if "llm_model" in st.session_state and st.session_state.llm_model is not None:
                        response = st.session_state.llm_model.generate_content(enhanced_prompt)

                    # Display the extracted content in the Streamlit app
                    st.markdown(response)

                    # Update chat history
                st.session_state.chat_history.append({"role": ASSISTANT, "content": response})
            else:
                st.warning("Please select a model to run.")
        else:
            st.warning("Please select columns for the chatbot to answer from.")




    