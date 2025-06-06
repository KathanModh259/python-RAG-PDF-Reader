@echo off
echo Starting PDF Analyzer...
echo.
echo Make sure Ollama is running first!
echo If not, run: ollama serve
echo.
pause
echo.
echo Starting Streamlit app...
streamlit run pdf_analyzer.py
