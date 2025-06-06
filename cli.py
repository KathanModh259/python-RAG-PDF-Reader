#!/usr/bin/env python3
"""
Command-line interface for PDF Analyzer
"""

import argparse
import os
from pdf_analyzer import PDFAnalyzer

def main():
    parser = argparse.ArgumentParser(description='PDF Analyzer with Local Ollama')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--model', default='llama3.2', help='Ollama model name')
    parser.add_argument('--chunk-size', type=int, default=500, help='Text chunk size')
    parser.add_argument('--overlap', type=int, default=100, help='Chunk overlap')
    parser.add_argument('--top-k', type=int, default=5, help='Number of chunks to retrieve')
    
    args = parser.parse_args()
    
    # Check if PDF file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file '{args.pdf_path}' not found.")
        return
    
    # Initialize analyzer
    print("Initializing PDF Analyzer...")
    analyzer = PDFAnalyzer()
    
    # Process PDF
    print(f"Processing PDF: {args.pdf_path}")
    success = analyzer.process_pdf(args.pdf_path, args.chunk_size, args.overlap)
    
    if not success:
        print("Failed to process PDF. Exiting.")
        return
    
    print(f"\nPDF processed successfully! Created {len(analyzer.chunks)} chunks.")
    print("\nYou can now ask questions about the PDF content.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Interactive Q&A loop
    while True:
        try:
            question = input("Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("Searching and generating answer...")
            result = analyzer.answer_question(question, args.top_k, args.model)
            
            print(f"\nAnswer: {result['answer']}\n")
            
            # Optionally show sources
            show_sources = input("Show sources? (y/n): ").strip().lower()
            if show_sources in ['y', 'yes']:
                print("\nSources:")
                for i, source in enumerate(result['sources']):
                    print(f"\n--- Source {i+1} (Page {source['page']}) ---")
                    print(f"Similarity: {source['similarity_score']:.3f}")
                    print(source['text'][:300] + "..." if len(source['text']) > 300 else source['text'])
            
            print("\n" + "="*50 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
