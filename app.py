import streamlit as st
import os
import shutil
from datetime import datetime
import tempfile

# Document processing
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

# Utils
import pandas as pd

# ---------- Configuration ----------
st.set_page_config(
    page_title="Document Q&A System - RAG (Free)",
    page_icon="📚",
    layout="wide"
)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("📚 RAG Document Q&A")
    st.markdown("**✨ 100% FREE - No OpenAI needed!**")
    st.markdown("---")
    
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        st.success("✅ Groq API Key detected!")
    else:
        st.error("❌ GROQ_API_KEY not found in secrets!")
        st.info("""
            **How to setup:**
            1. Open Streamlit Cloud dashboard
            2. Go to Settings → Secrets
            3. Add: `GROQ_API_KEY = "gsk_xxxxx"`
        """)
        st.stop()
    
    st.markdown("---")
    st.markdown("### 🆓 Free Technologies")
    st.info(
        """
        - **LLM**: Llama 3 70B (via Groq)
        - **Embeddings**: HuggingFace (free)
        - **Vector DB**: FAISS
        """
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Features")
    st.markdown(
        """
        - ✅ Upload PDF/TXT
        - ✅ Semantic Search
        - ✅ Source Tracking
        - ✅ Chat History
        - ✅ **100% Free!**
        """
    )

# ---------- Title ----------
st.title("📄 Document Q&A System with RAG (Free)")
st.markdown("""
    Upload your documents (PDF, TXT) and ask anything about their content.
    The AI will answer **only based on your uploaded documents**.
    
    🔥 **Powered by Groq Llama 3 (Free) + HuggingFace Embeddings + FAISS**
""")

# ---------- Session State Initialization ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# ---------- Embeddings Initialization Function ----------
@st.cache_resource
def get_embeddings():
    """Load HuggingFace embeddings (free)."""
    with st.spinner("🔄 Loading embeddings model (one time only)..."):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return embeddings

# ---------- Document Processing Function with FAISS ----------
def process_documents(uploaded_files, groq_api_key):
    """Process uploaded documents and create FAISS vector store."""
    if not groq_api_key:
        st.error("❌ Please enter your Groq API Key first!")
        return False
    
    if not uploaded_files:
        st.error("❌ Please upload files first!")
        return False
    
    with st.spinner("📖 Processing documents..."):
        all_documents = []
        progress_bar = st.progress(0)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_path)
                else:
                    loader = TextLoader(tmp_path, encoding='utf-8')
                
                documents = loader.load()
                
                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["uploaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                all_documents.extend(documents)
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")
                continue
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        if not all_documents:
            st.error("No documents were successfully processed!")
            return False
        
        # Split documents into chunks
        with st.spinner("✂️ Splitting documents into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(all_documents)
            st.info(f"✅ {len(chunks)} chunks created from {len(uploaded_files)} file(s)")
        
        # Create FAISS vector store
        with st.spinner("🗂️ Creating vector database with FAISS..."):
            embeddings = get_embeddings()
            
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )
            
            st.session_state.vectorstore = vectorstore
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )
            
            # Custom prompt for grounding - WILL FOLLOW USER'S LANGUAGE
            prompt_template = """
You are an assistant that answers questions based ONLY on the provided DOCUMENTS.

IMPORTANT INSTRUCTIONS:
1. Use ONLY information from the context below to answer the question.
2. If the answer is not in the context, say: "Sorry, I couldn't find that information in the uploaded documents."
3. DO NOT use your general knowledge.
4. CRITICAL: Answer in the SAME LANGUAGE as the user's question. 
   - If user asks in English → answer in English
   - If user asks in Indonesian → answer in Indonesian
   - If user asks in other languages → answer in that language

Context:
{context}

User's Question: {question}

Your Answer (in the same language as the user's question, based only on documents):
"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Initialize Groq Llama 3
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key=groq_api_key
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            st.session_state.qa_chain = qa_chain
            st.session_state.documents_processed = True
            
            return True

# ---------- Display Chat History ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            with st.expander("📚 View document sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.caption(f"{i}. {source}")

# ---------- Upload Section ----------
st.markdown("---")
st.subheader("📤 1. Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="Upload documents you want to ask questions about"
)

if uploaded_files:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"{len(uploaded_files)} file(s) ready to process")
    with col2:
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
            if process_documents(uploaded_files, groq_api_key):
                st.success("✅ Documents processed successfully! Ask me anything.")
                st.rerun()

# ---------- Chat Section ----------
if st.session_state.documents_processed:
    st.markdown("---")
    st.subheader("💬 2. Ask About Your Documents")
    
    if question := st.chat_input("Example: 'What is the main topic of this document?' or 'Summarize the key points...'"):
        
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching for answers in your documents..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": question})
                    answer = result["result"]
                    sources = list(set([doc.metadata.get("source", "Unknown") 
                                       for doc in result["source_documents"]]))
                    
                    st.write(answer)
                    
                    if sources:
                        with st.expander("📚 Document sources"):
                            for source in sources:
                                st.caption(f"📄 {source}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

else:
    if st.secrets.get("GROQ_API_KEY"):
        st.info("👈 Upload documents above to get started")
    else:
        st.warning("⚠️ **Please add your Groq API Key in Secrets first!**")

# ---------- Footer ----------
st.markdown("---")
st.caption(
    "🔍 Built with **LangChain + FAISS + Groq Llama 3 + HuggingFace** | "
    "100% Free - No OpenAI needed | Answers grounded only in uploaded documents"
)