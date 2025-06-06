#!/usr/bin/env python3
"""
Simple CLI PDF Analyzer - Just ask questions and get answers
"""

import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama
import os
from typing import List, Dict

class SimplePDFAnalyzer:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize the PDF analyzer with embedding model and vector store
        """
        print("🚀 Initializing PDF Analyzer...")
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        
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
            print(f"❌ Error extracting text from PDF: {str(e)}")
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
        print("📊 Creating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_vector_store(self, embeddings: np.ndarray):
        """
        Build FAISS vector store for similarity search
        """
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def process_pdf(self, pdf_path: str, chunk_size: int = 500, overlap: int = 100):
        """
        Complete pipeline to process PDF and build vector store
        """
        print(f"📄 Processing PDF: {pdf_path}")
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            print("❌ No text could be extracted from the PDF")
            return False
        
        print("✂️ Chunking text...")
        self.chunks = self.chunk_text(text, chunk_size, overlap)
        
        embeddings = self.create_embeddings(self.chunks)
        
        print("🏗️ Building vector store...")
        self.build_vector_store(embeddings)
        
        print(f"✅ Successfully processed PDF! Created {len(self.chunks)} chunks.")
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
    
    def answer_question(self, question: str, top_k: int = 5, model_name: str = "llama3.2") -> str:
        """
        Answer question using RAG pipeline
        """
        if self.index is None:
            return 'PDF not processed yet.'
        
        print("🔍 Searching for relevant content...")
        relevant_chunks = self.search_similar_chunks(question, top_k)
        
        if not relevant_chunks:
            return 'No relevant information found in the document.'
        
        # Combine context from relevant chunks
        context = "\n\n".join([
            f"[Page {chunk['page']}]: {chunk['text']}" 
            for chunk in relevant_chunks
        ])
        
        print("🤖 Generating answer...")
        answer = self.get_ollama_response(question, context, model_name)
        
        return answer


def main():
    print("=" * 60)
    print("📄 Simple PDF Analyzer with Local Ollama")
    print("=" * 60)
    
    # Find PDF file automatically
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    
    if not pdf_files:
        print("❌ No PDF files found in current directory!")
        return
    
    # Use the first PDF found (or the Arabic one if available)
    pdf_file = None
    for f in pdf_files:
        if "النظام" in f or "الأساسي" in f:
            pdf_file = f
            break
    
    if not pdf_file:
        pdf_file = pdf_files[0]
    
    print(f"📂 Found PDF: {pdf_file}")
    
    # Initialize analyzer
    analyzer = SimplePDFAnalyzer()
    
    # Process PDF
    success = analyzer.process_pdf(pdf_file)
    
    if not success:
        print("❌ Failed to process PDF. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("✅ PDF processed! You can now ask questions.")
    print("💡 Example questions:")
    print("   - ما هو موضوع هذا النظام؟")
    print("   - اذكر أهم المواد في النظام")
    print("   - ما هي أحكام الحكم؟")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    
    # Interactive Q&A loop
    while True:
        try:
            print()
            question = input("❓ Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q', 'خروج']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            answer = analyzer.answer_question(question)
            
            print("\n📝 Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
