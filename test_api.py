"""
Test script for the FastAPI PDF Legal Q&A API
Tests basic functionality of all endpoints
"""

import requests
import json
import time
import os

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['message']}")
            print(f"   - Ollama available: {data['ollama_available']}")
            print(f"   - PDF loaded: {data['pdf_loaded']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_upload_pdf():
    """Test PDF upload - requires a PDF file in the current directory"""
    print("📄 Testing PDF upload...")
    
    # Look for PDF files in current directory
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️  No PDF files found in current directory. Skipping upload test.")
        return False
    
    pdf_file = pdf_files[0]
    print(f"   Using PDF: {pdf_file}")
    
    try:
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file, f, 'application/pdf')}
            response = requests.post(f"{BASE_URL}/upload-pdf", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ PDF upload successful: {data['message']}")
            if data.get('pdf_info'):
                info = data['pdf_info']
                print(f"   - Chunks: {info['chunks_count']}")
                print(f"   - Characters: {info['total_characters']}")
            return True
        else:
            print(f"❌ PDF upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ PDF upload error: {str(e)}")
        return False

def test_pdf_info():
    """Test getting PDF info"""
    print("ℹ️  Testing PDF info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/pdf-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ PDF info retrieved")
            print(f"   - Filename: {data.get('filename', 'N/A')}")
            print(f"   - Upload time: {data.get('upload_time', 'N/A')}")
            return True
        elif response.status_code == 404:
            print("⚠️  No PDF loaded")
            return False
        else:
            print(f"❌ PDF info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ PDF info error: {str(e)}")
        return False

def test_ask_question():
    """Test asking a question"""
    print("❓ Testing question endpoint...")
    
    test_questions = [
        "What is this document about?",
        "Can you summarize the main points?",
        "What are the key legal requirements mentioned?"
    ]
    
    for question in test_questions:
        print(f"   Asking: {question}")
        try:
            payload = {
                "question": question,
                "use_enhanced_reasoning": True
            }
            response = requests.post(f"{BASE_URL}/ask", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Question answered")
                print(f"   - Answer length: {len(data['answer'])} characters")
                print(f"   - Processing time: {data.get('processing_time', 'N/A')} seconds")
                return True
            else:
                print(f"❌ Question failed: {response.status_code}")
                print(f"   Response: {response.text}")
                continue
                
        except Exception as e:
            print(f"❌ Question error: {str(e)}")
            continue
    
    return False

def test_models():
    """Test Ollama models endpoint"""
    print("🤖 Testing models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models retrieved")
            if 'models' in data:
                models = data['models']
                print(f"   - Found {len(models.get('models', []))} models")
            return True
        else:
            print(f"❌ Models failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models error: {str(e)}")
        return False

def test_ollama():
    """Test Ollama connection"""
    print("🦙 Testing Ollama connection...")
    try:
        response = requests.post(f"{BASE_URL}/test-ollama")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Ollama test passed: {data['message']}")
            print(f"   - Response: {data.get('response', 'N/A')}")
            return True
        else:
            print(f"❌ Ollama test failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ollama test error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting FastAPI PDF Legal Q&A API Tests")
    print("=" * 50)
    
    # Give the server time to start if just launched
    print("⏳ Waiting 3 seconds for server to be ready...")
    time.sleep(3)
    
    tests = [
        ("Health Check", test_health),
        ("Ollama Connection", test_ollama),
        ("Models List", test_models),
        ("PDF Upload", test_upload_pdf),
        ("PDF Info", test_pdf_info),
        ("Ask Question", test_ask_question),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! API is working correctly.")
    elif passed > len(tests) // 2:
        print("⚠️  Most tests passed. Check failed tests above.")
    else:
        print("❌ Many tests failed. Check server status and dependencies.")

if __name__ == "__main__":
    main()
