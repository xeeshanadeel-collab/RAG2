import streamlit as st
import random
import re
import requests
import json
from time import sleep
from pymongo import MongoClient 

# --- RAG Dependency Note ---
# NOTE: To run this code, you must install the PDF reading library:
# pip install pypdf
try:
    from pypdf import PdfReader
except ImportError:
    st.error("Please install the 'pypdf' library: pip install pypdf")
    PdfReader = None 


# --- Configuration (using placeholders for API key and URL) ---
try:
    API_KEY = st.secrets["API_KEY"]
except KeyError:
    st.error("Gemini API key not found. Please set the 'GEMINI_API_KEY' secret.")
    st.stop()
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
MAX_CHUNK_SIZE = 1500  # Max characters per document chunk

# --- MongoDB Configuration ---
MONGODB_URI = st.secrets["MONGODB_URI"] # REPLACE WITH YOUR URI
DB_NAME = "RAG_Chat_DB"
COLLECTION_NAME = "chat_history"
SESSION_ID = "rag_user_session_101" 

# --- MongoDB Memory Helpers ---

@st.cache_resource(show_spinner="Connecting to MongoDB...")
def get_database(uri, db_name):
    """Initializes and caches the MongoDB client connection."""
    if "<user>" in uri or "mongodb+srv" not in uri:
        st.error("Please update the `MONGODB_URI` placeholder with your actual connection string.")
        return None
    try:
        client = MongoClient(uri)
        # Ping the server to check the connection
        client.admin.command('ping')
        db = client[db_name]
        return db
    except Exception as e:
        st.error(f"Failed to connect to MongoDB. Check URI and network connection: {e}")
        return None

def load_chat_history(db, session_id):
    """Loads chat history for a session from MongoDB."""
    if db is None: return []
    try:
        collection = db[COLLECTION_NAME]
        doc = collection.find_one({"_id": session_id})
        # History is stored as a list of dicts: [{"role": "user/model", "text": "..."}]
        return doc.get("history", []) if doc else []
    except Exception as e:
        st.error(f"Error loading chat history from MongoDB: {e}")
        return []

def save_chat_history(db, session_id, history):
    """Saves the current chat history to MongoDB."""
    if db is None: return
    try:
        collection = db[COLLECTION_NAME]
        collection.update_one(
            {"_id": session_id},
            {"$set": {"history": history}},
            upsert=True
        )
    except Exception as e:
        st.error(f"Error saving chat history to MongoDB: {e}")

# --- Helper Functions (Existing, unchanged) ---

def text_splitter(text: str) -> list[str]:
    """Splits text into manageable chunks for RAG based on paragraph breaks."""
    
    chunks = []
    paragraphs = re.split(r'\n{2,}', text)
    
    current_chunk = ""
    for paragraph in paragraphs:
        if len(paragraph.strip()) == 0:
            continue
            
        if len(current_chunk) + len(paragraph) + 2 < MAX_CHUNK_SIZE:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            current_chunk = paragraph + "\n\n"
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

@st.cache_data
def load_and_split_pdf(uploaded_file):
    """Loads a PDF, extracts text, and splits it into chunks."""
    if PdfReader is None:
        return []

    st.info(f"Loading and processing PDF: {uploaded_file.name}")
    try:
        pdf_reader = PdfReader(uploaded_file)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n\n"
        
        st.success(f"Extracted {len(full_text):,} characters from {len(pdf_reader.pages)} pages.")
        
        st.info(f"Splitting document into chunks (max {MAX_CHUNK_SIZE} chars)...")
        chunks = text_splitter(full_text)
        st.success(f"Document split into {len(chunks)} chunks.")
        return chunks
        
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []


def retrieve_context(query: str, chunks: list[str], k: int = 3) -> list[str]:
    """
    SIMULATED RETRIEVAL:
    Finds the top 'k' chunks based on keyword matching.
    """
    query_words = set(query.lower().split())
    scores = []
    
    for i, chunk in enumerate(chunks):
        chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))
        match_count = len(query_words.intersection(chunk_words))
        scores.append((match_count, i))
        
    scores.sort(key=lambda x: x[0], reverse=True)
    
    top_indices = [index for score, index in scores if score > 0][:k]
    
    if not top_indices:
        st.warning("No keyword matches found. Retrieving a single random chunk for grounding.")
        return [random.choice(chunks)] if chunks else []
    
    context_chunks = [chunks[i] for i in top_indices]
    
    return context_chunks

def generate_rag_response(query: str, context: list[str], history: list) -> str:
    """
    Constructs the prompt and calls the Gemini API, incorporating history and PDF context.
    Returns the generated text.
    """
    
    if not context:
        return "Cannot answer: The PDF document is loaded, but no relevant context was retrieved for your question."
            
    context_text = "\n\n---\n\n".join(context)
    # Only include the text part of the history, not the role labels, in the model context
    chat_history_formatted = "\n".join([f"{item['role'].capitalize()}: {item['text']}" for item in history if item['text']])

    system_instruction = (
        "You are an expert Q&A assistant. "
        "Your goal is to answer the user's question ONLY by referring to the provided CONTEXT. "
        "Use the CHAT HISTORY to maintain context and continuity in the conversation. "
        "If the answer cannot be found in the context, clearly state, 'The provided documents do not contain enough information to answer this question.' "
        "Do not use external knowledge or the internet."
    )
    user_query_for_model = (
        f"--- CONTEXT ---\n{context_text}\n\n"
        f"--- CHAT HISTORY ---\n{chat_history_formatted}\n\n"
        f"---"
        f"Question: {query}"
    )
        
    # 2. CONSTRUCT THE API PAYLOAD
    payload = {
        "contents": [{"parts": [{"text": user_query_for_model}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
    }
    # No tools needed for strict RAG
    
    # 3. CALL THE API (with retry logic)
    st.info("Sending request to Gemini API with PDF RAG grounding...")
    
    generated_text = "Failed to get a response from the Gemini API after multiple retries."

    for attempt in range(3):
        try:
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status() 
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            generated_text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'API Response Error: Could not extract text.')
            
            return generated_text
            
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e}. Status code: {response.status_code}. Response: {response.text}")
            break
        except requests.exceptions.RequestException as e:
            st.warning(f"Attempt {attempt + 1}: Request failed: {e}. Retrying...")
            sleep(2 ** attempt) 
            
    return generated_text


# --- Streamlit UI ---

st.set_page_config(
    page_title="Z-BOT'S RAG Application",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Gemini PDF RAG Chatbot with Memory")
st.caption(f"Dedicated Retrieval-Augmented Generation from PDF (Session ID: `{SESSION_ID}`).")

# --- Initialize Session State and MongoDB Connection ---

# Initialize MongoDB connection
if 'db' not in st.session_state:
    st.session_state.db = get_database(MONGODB_URI, DB_NAME)

# Load chat history from MongoDB on initial run
if 'messages' not in st.session_state:
    st.session_state.messages = load_chat_history(st.session_state.db, SESSION_ID)
    if not st.session_state.messages:
         st.session_state.messages = [] 

# Initialize session state for document chunks
if 'doc_chunks' not in st.session_state:
    st.session_state.doc_chunks = []
if 'pdf_file' not in st.session_state:
    st.session_state.pdf_file = None


# --- Sidebar for Document Ingestion ---
with st.sidebar:
    st.header("1. Document Ingestion (PDF RAG)")
    
    uploaded_file = st.file_uploader(
        "Upload your PDF document:", 
        type="pdf"
    )

    if uploaded_file and uploaded_file != st.session_state.pdf_file:
        st.session_state.pdf_file = uploaded_file
        
        # Clear existing chat history when a new document is loaded
        st.session_state.messages = []
        save_chat_history(st.session_state.db, SESSION_ID, st.session_state.messages)
        st.info("Chat history cleared due to new PDF upload.")
        
        with st.spinner("Z-Bot is processing..."):
            st.session_state.doc_chunks = load_and_split_pdf(uploaded_file)
            
    st.markdown("---")
    st.metric("Loaded File", st.session_state.pdf_file.name if st.session_state.pdf_file else "None")
    st.metric("Total Chunks Loaded", len(st.session_state.doc_chunks))

    # Removed Grounding Options section

# --- Main Interface for Q&A ---
st.header("2. Chat Interface")

if not st.session_state.doc_chunks:
    st.warning("Please upload a PDF document in the sidebar to begin chatting.")
    
# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        
# Check if input should be enabled (only if PDF chunks are loaded)
if st.session_state.doc_chunks:
    if (prompt := st.chat_input("Ask a question about the PDF...")):
        
        # 1. Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 2. Append user message to local history for RAG prompt construction
        current_history = st.session_state.messages 
        current_history.append({"role": "user", "text": prompt})

        # 3. Invoke RAG pipeline
        final_answer = ""
        retrieved_context = []
        
        with st.spinner("Searching and generating response using PDF RAG grounding..."):
            
            # 3a. Retrieval Step for PDF
            retrieved_context = retrieve_context(prompt, st.session_state.doc_chunks, k=3)
            
            # 3b. Generation Step (Pass current history and context to Gemini)
            final_answer = generate_rag_response(prompt, retrieved_context, current_history)

        # 4. Display AI response and save
        if final_answer:
            with st.chat_message("model"):
                st.markdown(final_answer)
            
            # 5. Update session history with model response and save to MongoDB
            current_history.append({"role": "model", "text": final_answer})
            st.session_state.messages = current_history
            save_chat_history(st.session_state.db, SESSION_ID, st.session_state.messages)

            # 6. Display Context Used
            with st.expander("🔍 Information Sources (PDF Chunks Used)"):
                if retrieved_context:
                    st.markdown(f"**Context used from PDF ({len(retrieved_context)} chunks):**")
                    for i, chunk in enumerate(retrieved_context):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.code(chunk, language='text')
                else:
                    st.warning("No specific PDF chunks were used for grounding.")
else:
    st.info("The chat input is disabled. Please upload a PDF to enable RAG querying.")


# --- Footer and Disclaimer ---
st.markdown("---")
st.markdown("""
<style>
.stButton>button {
    background-color: #3B82F6; 
    color: white;
    border-radius: 8px;
    padding: 10px;
}
.stMetric {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
}
.stChatMessage {
    padding: 1rem;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
st.caption(f"Architecture: User Query -> PDF Retrieval (RAG) -> MongoDB Memory -> Gemini API Prompt -> Final Answer (Strictly based on PDF)")