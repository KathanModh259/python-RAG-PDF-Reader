# Setup Guide for PDF Analyzer

## Prerequisites Setup

### 1. Install Ollama

1. **Download Ollama**:
   - Go to [https://ollama.ai](https://ollama.ai)
   - Download the Windows installer
   - Run the installer and follow the instructions

2. **Verify Installation**:
   ```cmd
   ollama --version
   ```

3. **Start Ollama Service**:
   ```cmd
   ollama serve
   ```

4. **Pull a Model** (in a new terminal):
   ```cmd
   ollama pull llama3.2
   ```
   
   Other good models for Arabic content:
   ```cmd
   ollama pull qwen2.5
   ollama pull mistral
   ```

### 2. Install Python Dependencies

```cmd
cd "d:\INTERNSHIP\pdf_analyzer"
pip install -r requirements.txt
```

### 3. Test the Setup

1. **Test Ollama**:
   ```cmd
   ollama list
   ```
   You should see the models you pulled.

2. **Test with a simple question**:
   ```cmd
   ollama run llama3.2 "Hello, how are you?"
   ```

## Running the Application

### Option 1: Web Interface (Recommended)

1. **Start the Streamlit app**:
   ```cmd
   streamlit run pdf_analyzer.py
   ```

2. **Open browser** to: http://localhost:8501

3. **Upload your PDF** and start asking questions!

### Option 2: Command Line

```cmd
python cli.py "1-النظام الأساسي للحكم.pdf"
```

### Option 3: Batch File (Windows)

Double-click `run_app.bat` to start the web interface.

## Tips for Best Results

### For Arabic Documents:
- Use models like `qwen2.5` which have better Arabic support
- Ask questions in Arabic for better understanding
- The system will maintain the language of the source document

### Chunk Size Optimization:
- **Large documents**: Use chunk size 800-1000
- **Small documents**: Use chunk size 300-500
- **Technical documents**: Use smaller chunks (200-400)

### Model Selection:
- **General documents**: `llama3.2`, `mistral`
- **Technical/Legal**: `qwen2.5`
- **Code documentation**: `codellama`

## Troubleshooting

### "Ollama not found"
- Make sure Ollama is installed and in your PATH
- Restart your terminal after installation
- Try running `ollama serve` first

### "Model not found"
- Run `ollama pull model-name` to download the model
- Check available models with `ollama list`

### "No text extracted from PDF"
- Ensure the PDF contains selectable text (not just images)
- Try with a different PDF file
- Check if the PDF is password protected

### Memory Issues
- Reduce chunk size in the web interface
- Close other applications to free memory
- Use a smaller model like `llama3.2:8b` instead of larger variants

## Example Usage

1. **Upload the Arabic PDF**: `1-النظام الأساسي للحكم.pdf`

2. **Ask questions like**:
   - "ما هو موضوع هذه الوثيقة؟"
   - "اذكر أهم النقاط في النظام"
   - "ما هي المواد المتعلقة بالحكم؟"

3. **The system will**:
   - Find relevant sections in the PDF
   - Generate answers based only on the PDF content
   - Show you the source pages for verification
