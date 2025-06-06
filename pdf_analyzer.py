import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama
import pickle
import os
from typing import List, Dict, Tuple
import streamlit as st
import tempfile

class PDFAnalyzer:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the PDF analyzer with embedding model and vector store
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.metadata = []
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using pdfplumber
        """
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
                        text += "\n"
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
        
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
        """
        Split text into overlapping chunks for better retrieval
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Find page number in chunk
            page_num = 1
            for word in chunk_words:
                if "--- Page" in word:
                    try:
                        page_num = int(word.split()[2])
                    except:
                        pass
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'page': page_num,
                'start_word': i,
                'end_word': min(i + chunk_size, len(words))
            })
        
        return chunks
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        Create embeddings for text chunks
        """
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_vector_store(self, embeddings: np.ndarray):
        """
        Build FAISS vector store for similarity search
        """
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def process_pdf(self, pdf_path: str, chunk_size: int = 500, overlap: int = 100):
        """
        Complete pipeline to process PDF and build vector store
        """
        st.info("Extracting text from PDF...")
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            st.error("No text could be extracted from the PDF")
            return False
        
        st.info("Chunking text...")
        self.chunks = self.chunk_text(text, chunk_size, overlap)
        
        st.info("Creating embeddings...")
        embeddings = self.create_embeddings(self.chunks)
        
        st.info("Building vector store...")
        self.build_vector_store(embeddings)
        
        st.success(f"Successfully processed PDF! Created {len(self.chunks)} chunks.")
        return True
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks using vector similarity
        """
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def get_ollama_response(self, query: str, context: str, model_name: str = "llama3.2") -> str:
        """
        Get response from local Ollama model
        """
        prompt = f"""
Based ONLY on the following context from the PDF document, answer the question. 
Do not use any external knowledge or information not present in the context.
If the answer cannot be found in the context, say "المعلومة غير متوفرة في النص المقدم" (Information not available in the provided text).

Context:
{context}

Question: {query}

Answer (in Arabic if the document is in Arabic, otherwise in the same language as the question):
"""
        
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    def answer_question(self, question: str, top_k: int = 5, model_name: str = "llama3.2") -> Dict:
        """
        Answer question using RAG pipeline
        """
        if self.index is None:
            return {
                'answer': 'PDF not processed yet. Please upload and process a PDF first.',
                'sources': [],
                'context': ''
            }
        
        # Search for relevant chunks
        relevant_chunks = self.search_similar_chunks(question, top_k)
        
        if not relevant_chunks:
            return {
                'answer': 'No relevant information found in the document.',
                'sources': [],
                'context': ''
            }
        
        # Combine context from relevant chunks
        context = "\n\n".join([
            f"[Page {chunk['page']}]: {chunk['text']}" 
            for chunk in relevant_chunks
        ])
        
        # Get answer from Ollama
        answer = self.get_ollama_response(question, context, model_name)
        
        return {
            'answer': answer,
            'sources': relevant_chunks,
            'context': context
        }
    
    def save_index(self, filepath: str):
        """
        Save the vector index and chunks to disk
        """
        if self.index is not None:
            faiss.write_index(self.index, f"{filepath}.faiss")
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'metadata': self.metadata
                }, f)
    
    def load_index(self, filepath: str):
        """
        Load the vector index and chunks from disk
        """
        try:
            self.index = faiss.read_index(f"{filepath}.faiss")
            with open(f"{filepath}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']
            return True
        except Exception as e:
            st.error(f"Error loading index: {str(e)}")
            return False


def main():
    st.set_page_config(
        page_title="PDF Analyzer with Local Ollama",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 PDF Analyzer with Local Ollama")
    st.markdown("Upload a PDF and ask questions about its content using your local Ollama installation.")
    
    # Initialize the analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = PDFAnalyzer()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Ollama model selection
        model_name = st.text_input(
            "Ollama Model Name", 
            value="llama3.2",
            help="Make sure this model is installed in your local Ollama"
        )
        
        # Chunk size configuration
        chunk_size = st.slider("Chunk Size", 200, 1000, 500)
        overlap = st.slider("Chunk Overlap", 50, 200, 100)
        top_k = st.slider("Number of relevant chunks to retrieve", 3, 10, 5)
        
        # Test Ollama connection
        if st.button("Test Ollama Connection"):
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=[{'role': 'user', 'content': 'Hello'}]
                )
                st.success("✅ Ollama connection successful!")
            except Exception as e:
                st.error(f"❌ Ollama connection failed: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("📁 PDF Selection")
        
        # Check for existing PDFs in workspace
        workspace_pdfs = [f for f in os.listdir(".") if f.endswith(".pdf")]
        
        if workspace_pdfs:
            st.subheader("📄 Available PDFs in Workspace")
            selected_pdf = st.selectbox(
                "Select a PDF from workspace:",
                workspace_pdfs,
                help="PDFs found in your current workspace"
            )
            
            if selected_pdf:
                pdf_path = os.path.abspath(selected_pdf)
                st.success(f"Selected: {selected_pdf}")
                
                if st.button("Process Selected PDF", type="primary"):
                    with st.spinner("Processing PDF..."):
                        success = st.session_state.analyzer.process_pdf(
                            pdf_path, 
                            chunk_size=chunk_size, 
                            overlap=overlap
                        )
                    
                    if success:
                        st.session_state.pdf_processed = True
                        st.session_state.current_pdf = selected_pdf
        
        st.subheader("📤 Or Upload New PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a different PDF document to analyze"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            st.success(f"File uploaded: {uploaded_file.name}")
            
            if st.button("Process Uploaded PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    success = st.session_state.analyzer.process_pdf(
                        tmp_path, 
                        chunk_size=chunk_size, 
                        overlap=overlap
                    )
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
                if success:
                    st.session_state.pdf_processed = True
                    st.session_state.current_pdf = uploaded_file.name
    with col2:
        st.header("❓ Ask Questions")
        
        if hasattr(st.session_state, 'pdf_processed') and st.session_state.pdf_processed:
            # Show current PDF being analyzed
            if hasattr(st.session_state, 'current_pdf'):
                st.info(f"📄 Currently analyzing: **{st.session_state.current_pdf}**")
            
            # Question input
            question = st.text_area(
                "Enter your question about the PDF:",
                height=100,
                placeholder="ما هو موضوع هذا النظام؟ (What is the subject of this system?)"
            )
            
            if st.button("Get Answer", type="primary") and question:
                with st.spinner("Searching and generating answer..."):
                    result = st.session_state.analyzer.answer_question(
                        question, 
                        top_k=top_k, 
                        model_name=model_name
                    )
                
                # Display answer
                st.subheader("📝 Answer")
                st.write(result['answer'])
                
                # Display sources
                if result['sources']:
                    st.subheader("📖 Sources")
                    for i, source in enumerate(result['sources']):
                        with st.expander(f"Source {i+1} (Page {source['page']}) - Similarity: {source['similarity_score']:.3f}"):
                            st.text(source['text'][:500] + "..." if len(source['text']) > 500 else source['text'])
        else:
            st.info("Please upload and process a PDF first to start asking questions.")
    
    # Chat history section
    st.header("💬 Chat History")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.expander(f"Q{i+1}: {q[:50]}..."):
            st.write(f"**Question:** {q}")
            st.write(f"**Answer:** {a}")
    
    # Add current Q&A to history
    if 'question' in locals() and 'result' in locals() and question:
        if (question, result['answer']) not in st.session_state.chat_history:
            st.session_state.chat_history.append((question, result['answer']))


if __name__ == "__main__":
    main()
