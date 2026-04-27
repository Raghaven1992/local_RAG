import os
import shutil
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
DATA_PATH = "D:/New_App/data"
CHROMA_PATH = "D:/New_App/chroma_db"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

def run_rag_system():
    # 1. SETUP MODELS
    print("--- Initializing Models ---")
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    llm = OllamaLLM(model="gemma3:4b", base_url=OLLAMA_BASE_URL, temperature=0.1)

    # 2. ASK USER: FRESH START OR CONTINUE?
    choice = input("Do you want to (1) Re-ingest documents or (2) Start chatting directly? [1/2]: ")

    if choice == "1":
        # WIPE OLD DATA
        if os.path.exists(CHROMA_PATH):
            print(f"Cleaning up old database...")
            shutil.rmtree(CHROMA_PATH)

        # LOAD AND CHUNK
        print(f"Loading PDFs from {DATA_PATH}...")
        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks.")

        # CREATE VECTOR STORE
        print("Indexing to ChromaDB (This may take a moment)...")
        db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
        print("✅ Indexing Complete!")
    else:
        # LOAD EXISTING DB
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        print("✅ Loaded existing database.")

    # 3. THE CHAT LOOP
    print("\n--- 🤖 Local RAG Assistant Ready! (Type 'quit' to exit) ---")
    
    template_string = """
    ### [SYSTEM INSTRUCTION]
    Use ONLY the provided context to answer the question. 
    If the answer isn't in the context, say you don't know.
    
    ### [CONTEXT]
    {context}
    
    ### [USER QUESTION]
    {question}
    
    ### [RESPONSE]
    """
    prompt_template = ChatPromptTemplate.from_template(template_string)

    while True:
        query = input("\nYour Question: ")
        if query.lower() == "quit": break

        # RETRIEVAL (Using simple similarity to avoid score issues)
        results = db.similarity_search(query, k=40)
        
        # DEBUG PREVIEW
        print(f"--- Found {len(results)} relevant snippets ---")
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        full_prompt = prompt_template.format(context=context_text, question=query)

        print("Generating Answer...")
        response = llm.invoke(full_prompt)
        
        sources = {doc.metadata.get("source") for doc in results}
        print(f"\nResponse: {response}")
        print(f"Sources: {sources}")

if __name__ == "__main__":
    run_rag_system()