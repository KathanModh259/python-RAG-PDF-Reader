@echo off
echo 🚀 Starting PDF Legal Q&A FastAPI Server
echo.
echo Server will be available at:
echo   - API: http://localhost:8000
echo   - Docs: http://localhost:8000/docs
echo   - Home: http://localhost:8000/
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
python fastapi_app.py

pause
