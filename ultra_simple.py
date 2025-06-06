#!python
"""
Ultra Simple PDF Q&A - Just ask and get answers
"""

import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ollama
import os

def load_pdf():
    """Find and load the Arabic PDF"""
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    
    if not pdf_files:
        print("❌ No PDF files found!")
        return None
    
    # Prefer Arabic PDF
    arabic_pdf = None
    for f in pdf_files:
        if "النظام" in f or "الأساسي" in f:
            arabic_pdf = f
            break
    
    pdf_file = arabic_pdf if arabic_pdf else pdf_files[0]
    print(f"📄 Loading: {pdf_file}")
    
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
    
    return text

def setup_rag(text):
    """Setup RAG system"""
    print("🔧 Setting up RAG system...")
    
    # Chunk text
    words = text.split()
    chunks = []
    chunk_size = 500
    overlap = 100
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        chunks.append(chunk_text)
    
    # Create embeddings
    print("📊 Creating embeddings...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Build vector store
    print("🏗️ Building vector store...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype('float32'))
    
    print(f"✅ Ready! Created {len(chunks)} chunks.")
    return model, index, chunks

def ask_question(question, model, index, chunks, top_k=5):
    """Get answer for question"""
    # Search similar chunks
    query_embedding = model.encode([question])
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding.astype('float32'), top_k)
    
    # Get relevant chunks
    relevant_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
    context = "\n\n".join(relevant_chunks)
    
    # Get answer from Ollama
    prompt = f"""
Based ONLY on the following context, answer the question in Arabic.
If the answer is not in the context, say "Information not Available".

Context: {context}

Question: {question}

Answer:"""
    
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("🚀 PDF Q&A System")
    print("=" * 50)
    
    # Load PDF
    text = load_pdf()
    if not text:
        return
    
    # Setup RAG
    model, index, chunks = setup_rag(text)
    
    print("\n💬 Ask questions (type 'quit' to exit):")
    print("=" * 50)
    
    while True:
        try:
            question = input("\n❓ ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question:
                print("🤖 Thinking...")
                answer = ask_question(question, model, index, chunks)
                print(f"\n📝 {answer}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
