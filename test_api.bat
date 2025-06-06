@echo off
echo 🧪 Testing PDF Legal Q&A FastAPI Server
echo.
echo This will test all API endpoints
echo Make sure the FastAPI server is running first!
echo.

cd /d "%~dp0"
python test_api.py

pause
