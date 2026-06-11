import streamlit as st
import os
import shutil
from datetime import datetime
import tempfile

# Document processing - VERSI YANG DIPERBAIKI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Evaluation
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# Utils
import pandas as pd

# ---------- Konfigurasi ----------
st.set_page_config(
    page_title="Document Q&A System - RAG (Gratis)",
    page_icon="📚",
    layout="wide"
)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("📚 RAG Document Q&A")
    st.markdown("**✨ 100% GRATIS - No OpenAI needed!**")
    st.markdown("---")
    
    # Ambil API key dari secrets (AMAN)
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        st.success("✅ Groq API Key terdeteksi!")
    else:
        st.error("❌ GROQ_API_KEY tidak ditemukan di secrets!")
        st.info("""
            **Cara setup:**
            1. Buka dashboard Streamlit Cloud
            2. Settings → Secrets
            3. Tambahkan: `GROQ_API_KEY = "gsk_xxxxx"`
        """)
        st.stop()
    
    st.markdown("---")
    st.markdown("### 🆓 Teknologi Gratis")
    st.info(
        """
        - **LLM**: Llama 3 70B (via Groq)
        - **Embeddings**: HuggingFace (free)
        - **Vector DB**: ChromaDB
        - **Evaluation**: RAGAS
        """
    )
    
    st.markdown("---")
    st.markdown("### 📊 Evaluasi Metrics")
    st.info(
        "**Faithfulness**: Apakah jawaban berdasarkan dokumen?\n"
        "**Answer Relevancy**: Seberapa relevan jawaban?"
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Fitur")
    st.markdown(
        """
        - ✅ Upload PDF/TXT
        - ✅ Semantic Search
        - ✅ Source Tracking
        - ✅ RAGAS Evaluation
        - ✅ Chat History
        - ✅ **100% Gratis!**
        """
    )

# ---------- Title ----------
st.title("📄 Document Q&A System with RAG (Gratis)")
st.markdown("""
    Upload dokumen Anda (PDF, TXT) lalu tanyakan apapun tentang isinya.
    AI akan menjawab **hanya berdasarkan dokumen yang Anda upload**.
    
    🔥 **Powered by Groq Llama 3 (Gratis) + HuggingFace Embeddings**
""")

# ---------- Inisialisasi Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None

# ---------- Fungsi Inisialisasi Embeddings (Gratis) ----------
@st.cache_resource
def get_embeddings():
    """Load HuggingFace embeddings (gratis, lokal)."""
    with st.spinner("🔄 Loading embeddings model (sekali saja)..."):
        # Gunakan model kecil & cepat
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return embeddings

# ---------- Fungsi Proses Dokumen ----------
def process_documents(uploaded_files, groq_api_key):
    """Process uploaded documents and create vector store."""
    if not groq_api_key:
        st.error("❌ Masukkan Groq API Key terlebih dahulu!")
        return False
    
    if not uploaded_files:
        st.error("❌ Upload file terlebih dahulu!")
        return False
    
    with st.spinner("📖 Memproses dokumen..."):
        all_documents = []
        
        # Progress bar
        progress_bar = st.progress(0)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load document based on type
            try:
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_path)
                else:  # txt file
                    loader = TextLoader(tmp_path, encoding='utf-8')
                
                documents = loader.load()
                
                # Add metadata
                for doc in documents:
                    doc.metadata["source"] = uploaded_file.name
                    doc.metadata["uploaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                all_documents.extend(documents)
                
                # Cleanup temp file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")
                continue
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        if not all_documents:
            st.error("Tidak ada dokumen yang berhasil diproses!")
            return False
        
        # Split documents into chunks
        with st.spinner("✂️ Memecah dokumen menjadi chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(all_documents)
            st.info(f"✅ {len(chunks)} chunks dibuat dari {len(uploaded_files)} file")
        
        # Create vector store with free embeddings
        with st.spinner("🗂️ Membuat vector database (pakai HuggingFace)..."):
            embeddings = get_embeddings()
            
            # Clear existing Chroma DB if exists
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
            
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            vectorstore.persist()
            
            st.session_state.vectorstore = vectorstore
            
            # Create QA Chain with Groq (Llama 3 - GRATIS)
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 4}  # Retrieve top 4 relevant chunks
            )
            
            # Custom prompt untuk grounding
            prompt_template = """
            Anda adalah asisten yang membantu menjawab pertanyaan berdasarkan DOKUMEN yang diberikan.
            
            INSTRUKSI PENTING:
            1. Gunakan ONLY informasi dari konteks berikut untuk menjawab pertanyaan.
            2. Jika jawaban tidak ada dalam konteks, katakan: "Maaf, informasi tersebut tidak ditemukan dalam dokumen yang diupload."
            3. JANGAN menggunakan pengetahuan umum Anda.
            4. JANGAN membuat informasi.
            5. Jawab dalam Bahasa Indonesia jika pertanyaan dalam Bahasa Indonesia.
            
            Konteks:
            {context}
            
            Pertanyaan: {question}
            
            Jawaban (berdasarkan dokumen saja):
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Initialize Groq Llama 3 (GRATIS)
            llm = ChatGroq(
                model="llama3-70b-8192",  # Llama 3 70B - powerful & gratis
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

# ---------- Fungsi Evaluasi RAGAS ----------
def evaluate_rag(qa_chain, questions, ground_truths=None):
    """Evaluate RAG system using RAGAS metrics."""
    with st.spinner("📊 Mengevaluasi performa RAG system..."):
        answers = []
        contexts = []
        
        progress_bar = st.progress(0)
        for idx, question in enumerate(questions):
            result = qa_chain.invoke({"query": question})
            answers.append(result["result"])
            contexts.append([doc.page_content for doc in result["source_documents"]])
            progress_bar.progress((idx + 1) / len(questions))
        
        # Create dataset for RAGAS
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        dataset = Dataset.from_dict(data)
        
        # Evaluate
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
        )
        
        return result, answers, contexts

# ---------- Tampilkan Riwayat Chat ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            with st.expander("📚 Lihat sumber dokumen"):
                for i, source in enumerate(message["sources"], 1):
                    st.caption(f"{i}. {source}")

# ---------- Upload Section ----------
st.markdown("---")
st.subheader("📤 1. Upload Dokumen")

uploaded_files = st.file_uploader(
    "Upload PDF atau TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="Upload dokumen yang ingin Anda tanyakan"
)

if uploaded_files:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"{len(uploaded_files)} file siap diproses")
    with col2:
        if st.button("🚀 Proses Dokumen", type="primary", use_container_width=True):
            groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
            if process_documents(uploaded_files, groq_api_key):
                st.success("✅ Dokumen berhasil diproses! Silakan tanyakan sesuatu.")
                st.rerun()

# ---------- Chat Section ----------
if st.session_state.documents_processed:
    st.markdown("---")
    st.subheader("💬 2. Tanyakan Tentang Dokumen")
    
    # User input
    if question := st.chat_input("Contoh: 'Apa topik utama dokumen ini?' atau 'Ringkaskan kebijakan yang disebutkan...'"):
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🔍 Mencari jawaban di dokumen..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": question})
                    answer = result["result"]
                    sources = list(set([doc.metadata.get("source", "Unknown") 
                                       for doc in result["source_documents"]]))
                    
                    # Display answer
                    st.write(answer)
                    
                    # Display sources
                    if sources:
                        with st.expander("📚 Sumber dokumen"):
                            for source in sources:
                                st.caption(f"📄 {source}")
                    
                    # Save to session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # ---------- Evaluation Section ----------
    st.markdown("---")
    st.subheader("📊 3. Evaluasi Performa RAG")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Jalankan Evaluasi RAGAS", use_container_width=True):
            if len(st.session_state.messages) >= 3:
                # Take recent questions from chat history
                questions = [msg["content"] for msg in st.session_state.messages 
                           if msg["role"] == "user"][-3:]
                
                result, answers, contexts = evaluate_rag(
                    st.session_state.qa_chain, 
                    questions
                )
                
                st.session_state.evaluation_results = result
                
                # Display results
                st.success("✅ Evaluasi selesai!")
                
                # Metrics
                col_metric1, col_metric2 = st.columns(2)
                with col_metric1:
                    st.metric(
                        "Faithfulness", 
                        f"{result['faithfulness']:.2%}",
                        help="Seberapa akurat jawaban berdasarkan dokumen"
                    )
                with col_metric2:
                    st.metric(
                        "Answer Relevancy", 
                        f"{result['answer_relevancy']:.2%}",
                        help="Seberapa relevan jawaban dengan pertanyaan"
                    )
                
                # Display detailed results in expander
                with st.expander("📋 Detail Hasil Evaluasi"):
                    for i, q in enumerate(questions):
                        st.write(f"**Q{i+1}:** {q}")
                        st.write(f"**Answer:** {answers[i][:200]}...")
                        st.write("---")
            else:
                st.warning("Minimal 3 pertanyaan di chat untuk evaluasi meaningful")
    
    with col2:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Display evaluation metrics if available
    if st.session_state.evaluation_results:
        st.markdown("---")
        st.subheader("📈 Ringkasan Performa")
        
        metrics_df = pd.DataFrame({
            'Metric': ['Faithfulness', 'Answer Relevancy'],
            'Score': [
                st.session_state.evaluation_results['faithfulness'],
                st.session_state.evaluation_results['answer_relevancy']
            ]
        })
        metrics_df['Score'] = metrics_df['Score'].apply(lambda x: f"{x:.2%}")
        
        st.table(metrics_df)

else:
    if st.secrets.get("GROQ_API_KEY"):
        st.info("👈 Upload dokumen di atas untuk memulai")
    else:
        st.warning("⚠️ **Masukkan Groq API Key di Secrets terlebih dahulu!**")

# ---------- Footer ----------
st.markdown("---")
st.caption(
    "🔍 Built with **LangChain + ChromaDB + Groq Llama 3 + HuggingFace** | "
    "100% Gratis - No OpenAI needed | Answers grounded only in uploaded documents"
)