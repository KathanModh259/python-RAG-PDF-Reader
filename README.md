# PDF Analyzer with Local Ollama

A powerful tool to analyze PDF documents and answer questions using Retrieval-Augmented Generation (RAG) with your local Ollama installation. This tool extracts content only from the provided PDF and doesn't use any external knowledge sources.

## Features

- **PDF Text Extraction**: Uses pdfplumber for robust text extraction, including Arabic text
- **Semantic Search**: Creates embeddings and uses FAISS for fast similarity search
- **Local AI**: Uses your local Ollama installation for answering questions
- **Multi-language Support**: Works with Arabic, English, and other languages
- **Two Interfaces**: Both web UI (Streamlit) and command-line interface
- **Source Citation**: Shows which parts of the PDF were used to generate answers

## Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Install and setup Ollama**:
   - Download Ollama from [https://ollama.ai](https://ollama.ai)
   - Install and start the Ollama service
   - Pull a model (e.g., `ollama pull llama3.2`)

3. **Verify Ollama is running**:
```bash
ollama list
```

## Usage

### Web Interface (Recommended)

1. **Start the Streamlit app**:
```bash
streamlit run pdf_analyzer.py
```

2. **Open your browser** to `http://localhost:8501`

3. **Upload your PDF** and click "Process PDF"

4. **Ask questions** about the PDF content

### Command Line Interface

1. **Run the CLI**:
```bash
python cli.py "path/to/your/pdf/file.pdf"
```

2. **Ask questions** interactively

### Advanced Options

```bash
python cli.py "document.pdf" --model llama3.2 --chunk-size 500 --overlap 100 --top-k 5
```

## Configuration

Edit `config.py` to customize:
- Ollama model name
- Chunk size and overlap
- Embedding model
- Number of relevant chunks to retrieve

## How It Works

1. **Text Extraction**: Extracts text from PDF using pdfplumber
2. **Text Chunking**: Splits text into overlapping chunks for better context
3. **Embedding Creation**: Creates vector embeddings using SentenceTransformers
4. **Vector Storage**: Stores embeddings in FAISS index for fast similarity search
5. **Question Processing**: 
   - Converts question to embedding
   - Finds most similar text chunks
   - Sends relevant context to Ollama
   - Returns AI-generated answer based only on PDF content

## Supported Models

Any Ollama model can be used. Popular choices:
- `llama3.2` (recommended for general use)
- `mistral`
- `codellama` (for code-related documents)
- `qwen2.5` (good for multilingual content)

Make sure to pull the model first: `ollama pull model-name`

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check if model is available: `ollama list`
- Verify the model name in configuration

### PDF Processing Issues
- Ensure the PDF contains extractable text (not just images)
- Try with a different PDF to isolate the issue
- Check file permissions

### Memory Issues
- Reduce chunk size in configuration
- Use a smaller embedding model
- Process smaller PDFs

## Example Questions

For an Arabic legal document:
- "ما هو موضوع هذا النظام؟" (What is the subject of this system?)
- "ما هي المواد المتعلقة بالحكم؟" (What are the articles related to governance?)

For English documents:
- "What is the main topic of this document?"
- "Summarize the key points"
- "What are the requirements mentioned?"

## File Structure

```
pdf_analyzer/
├── pdf_analyzer.py      # Main Streamlit application
├── cli.py              # Command-line interface
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── cache/             # Cached indexes (created automatically)
```

## License

This project is open source and available under the MIT License.
