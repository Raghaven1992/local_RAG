import os
import shutil
import warnings
import time

# Suppress irrelevant warnings for a cleaner terminal UI
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --- DYNAMIC CONFIGURATION ---
# Detects the script location to ensure it runs on any machine (Portable)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

def run_rag_system():
    # 1. INITIALIZATION
    print("--- 🤖 Initializing AI Models (Gemma 3 & Nomic) ---")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    llm = OllamaLLM(
        model="gemma3:4b", 
        base_url=OLLAMA_BASE_URL, 
        temperature=0.1  # Low temp for factual accuracy
    )

    # Safety Check: Create data folder if missing
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📁 Created 'data' folder. Place your PDFs in: {DATA_PATH}")
        return

    # 2. SELECTION MENU
    print("\n[RAG MODE SELECTION]")
    print("(1) Ingest: Wipe old data and process new PDFs")
    print("(2) Chat:  Ask questions using existing database")
    choice = input("Enter selection [1/2]: ")

    if choice == "1":
        # WIPE OLD DATA
        if os.path.exists(CHROMA_PATH):
            print(f"🧹 Clearing existing vector database...")
            shutil.rmtree(CHROMA_PATH)

        # LOAD AND CHUNK
        print(f"📂 Loading documents from: {DATA_PATH}")
        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        
        if not docs:
            print("❌ No PDF files found! Please add documents to the /data folder.")
            return

        # UPDATE: Increase chunk size and overlap to keep table rows together
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,    # Increased from 700
            chunk_overlap=300,  # Increased from 100
            separators=["\n\n", "\n", ".", " "] # Helps keep table rows intact
            )
        chunks = text_splitter.split_documents(docs)
        print(f"✂️  Created {len(chunks)} text chunks.")

        # CREATE VECTOR STORE
        print("🔢 Indexing vectors to ChromaDB (This may take a minute)...")
        db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
        print("✅ Indexing complete! Your knowledge base is ready.")
    
    else:
        # LOAD EXISTING DATABASE
        if not os.path.exists(CHROMA_PATH):
            print("❌ Database not found. Please run Option 1 first.")
            return
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        print("✅ Successfully loaded the existing knowledge base.")

    # 3. THE INTERACTIVE CHAT LOOP
    print("\n--- 🧠 AI Assistant Ready! (Type 'quit' to exit) ---")
    
    template_string = """
    ### [SYSTEM INSTRUCTION]
    You are a technical 3GPP expert. Answer the question using the provided context. 
    If the context contains related technical tables or specifications but not a direct word-for-word answer, 
    try to infer the answer based on the technical parameters (like 5QI mappings). 
    Only say you don't know if the context is completely unrelated.
    
    ### [CONTEXT]
    {context}
    
    ### [USER QUESTION]
    {question}
    
    ### [RESPONSE]
    """
    prompt_template = ChatPromptTemplate.from_template(template_string)

    while True:
        query_text = input("\nYour Question: ")
        if query_text.lower() == "quit":
            print("Shutting down...")
            break

        # --- LATENCY TRACKING START ---
        start_total = time.perf_counter()

        # STAGE 1: RETRIEVAL
        start_retrieval = time.perf_counter()
        # Retrieving Top 40 chunks for better context
        results = db.similarity_search(query_text, k=15)
        retrieval_time = time.perf_counter() - start_retrieval

        # STAGE 2: GENERATION
        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        formatted_prompt = prompt_template.format(context=context_text, question=query_text)

        print("AI is thinking...")
        start_gen = time.perf_counter()
        response = llm.invoke(formatted_prompt)
        generation_time = time.perf_counter() - start_gen

        total_latency = time.perf_counter() - start_total

        # --- OUTPUT ---
        print(f"\nResponse: {response}")
        print(f"\nSources: {list(set([doc.metadata.get('source') for doc in results]))}")
        
        print("-" * 30)
        print(f"⏱️  LATENCY REPORT:")
        print(f"  └─ Search:     {retrieval_time:.3f}s")
        print(f"  └─ Generation: {generation_time:.3f}s")
        print(f"  └─ TOTAL:      {total_latency:.3f}s")
        print("-" * 30)

if __name__ == "__main__":
    run_rag_system()