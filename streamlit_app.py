"""Streamlit UI for Agentic RAG System"""

import streamlit as st
from dotenv import load_dotenv  # 1. Import this
import os

# 2. LOAD THIS FIRST! Before any other local imports.
# This ensures that when Config is imported, the keys are already in memory.
load_dotenv() 

from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

# 3. Now it is safe to import Config
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="🤖 SQL RAG Agentic Seach",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS
st.markdown("""
    <style>
    /* Clean up the main container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }
    /* Style the source document expanders */
    .streamlit-expanderHeader {
        font-size: 0.9em;
        color: #555;
    }
    /* Hide the default Streamlit footer */
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        # Initialize components
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()
        
        # Use default URLs
        urls = Config.DEFAULT_URLS
        
        # Process documents
        documents = doc_processor.process_urls(urls)
        
        # Create vector store
        vector_store.create_vectorstore(documents)
        
        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def main():
    """Main application with Modern Chat UI and Geometric Particle Sidebar"""
    init_session_state()
    
    # --- SIDEBAR: Geometric Particle Constellation & Settings ---
    with st.sidebar:
        st.title("⚙️ System Control")
        
        starry_html = """
        <style>
        .space-container {
            width: 100%;
            height: 200px;
            background: linear-gradient(to bottom, #020111 10%, #20124d 100%);
            border-radius: 8px;
            position: relative;
            overflow: hidden;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        }

        .star {
            position: absolute;
            background-color: white;
            border-radius: 50%;
            animation: twinkle infinite alternate;
        }

        @keyframes twinkle {
            0% { opacity: 0.2; transform: scale(0.8); }
            100% { opacity: 1; transform: scale(1.2); }
        }

        /* Generate multiple stars with different sizes, positions, and animation speeds */
        .star:nth-child(1) { width: 3px; height: 3px; top: 10%; left: 20%; animation-duration: 2s; }
        .star:nth-child(2) { width: 2px; height: 2px; top: 30%; left: 80%; animation-duration: 3s; }
        .star:nth-child(3) { width: 4px; height: 4px; top: 60%; left: 40%; animation-duration: 1.5s; }
        .star:nth-child(4) { width: 2px; height: 2px; top: 80%; left: 10%; animation-duration: 4s; }
        .star:nth-child(5) { width: 3px; height: 3px; top: 40%; left: 60%; animation-duration: 2.5s; }
        .star:nth-child(6) { width: 2px; height: 2px; top: 15%; left: 90%; animation-duration: 3.5s; }
        .star:nth-child(7) { width: 5px; height: 5px; top: 70%; left: 85%; animation-duration: 2.2s; opacity: 0.8; }
        .star:nth-child(8) { width: 3px; height: 3px; top: 85%; left: 50%; animation-duration: 1.8s; }
        .star:nth-child(9) { width: 2px; height: 2px; top: 25%; left: 10%; animation-duration: 3.1s; }
        .star:nth-child(10) { width: 4px; height: 4px; top: 50%; left: 25%; animation-duration: 2.7s; }
        .star:nth-child(11) { width: 2px; height: 2px; top: 5%; left: 50%; animation-duration: 4.5s; }
        .star:nth-child(12) { width: 3px; height: 3px; top: 90%; left: 30%; animation-duration: 1.9s; }
        
        /* Subtle glowing orb in the background */
        .nebula {
            position: absolute;
            width: 150px;
            height: 150px;
            background: radial-gradient(circle, rgba(138,43,226,0.2) 0%, rgba(0,0,0,0) 70%);
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            border-radius: 50%;
            animation: pulse-nebula 6s infinite alternate;
        }

        @keyframes pulse-nebula {
            0% { opacity: 0.5; transform: translate(-50%, -50%) scale(1); }
            100% { opacity: 1; transform: translate(-50%, -50%) scale(1.2); }
        }
        </style>

        <div class="space-container">
            <div class="nebula"></div>
            <!-- The Stars -->
            <div class="star"></div><div class="star"></div><div class="star"></div>
            <div class="star"></div><div class="star"></div><div class="star"></div>
            <div class="star"></div><div class="star"></div><div class="star"></div>
            <div class="star"></div><div class="star"></div><div class="star"></div>
        </div>
        """
        st.markdown(starry_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Keep existing loading logic and Clear button
        if not st.session_state.initialized:
            with st.spinner("Initializing Vector Store & Agent..."):
                rag_system, num_chunks = initialize_rag()
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.session_state.initialized = True
                    st.success(f"✅ System Online\n\n📚 {num_chunks} chunks loaded")
        else:
            st.success("✅ System Online and Ready")
            
        st.markdown("---")
        
        # Clear Chat History Button (Kept)
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.history = []
            st.rerun() # Refresh page instantly
            
        st.markdown("---")
        st.caption("Powered by OpenRouter & LangGraph")

    # --- MAIN SCREEN: Chat Interface (Kept exactly the same) ---
    st.title("🤖 Agentic RAG Assistant")
    st.caption("Ask me anything about the loaded documents or general knowledge!")

    # Display previous chat history
    for msg in st.session_state.history:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(msg['question'])
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write(msg['answer'])
            if msg.get('retrieved_docs'):
                with st.expander("📄 View Source Documents"):
                    for i, doc in enumerate(msg['retrieved_docs'], 1):
                        st.markdown(f"**Source {i}:**")
                        st.info(doc.page_content[:300] + "...")
            st.caption(f"⏱️ Response time: {msg['time']:.2f}s")

    # Chat Input box (Pinned to the bottom)
    if question := st.chat_input("Type your question here..."):
        
        if not st.session_state.rag_system:
            st.error("System is still loading. Please wait.")
            return

        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing documents..."):
                start_time = time.time()
                
                result = st.session_state.rag_system.run(question)
                elapsed_time = time.time() - start_time
                
                st.write(result['answer'])
                
                if result.get('retrieved_docs'):
                    with st.expander("📄 View Source Documents"):
                        for i, doc in enumerate(result['retrieved_docs'], 1):
                            st.markdown(f"**Source {i}:**")
                            st.info(doc.page_content[:300] + "...")
                
                st.caption(f"⏱️ Response time: {elapsed_time:.2f}s")
                
                st.session_state.history.append({
                    'question': question,
                    'answer': result['answer'],
                    'retrieved_docs': result.get('retrieved_docs', []),
                    'time': elapsed_time
                })
if __name__ == "__main__":
    main()