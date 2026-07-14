"""Test the FastAPI backend directly."""
import sys, json, asyncio, urllib.request
sys.path.insert(0, '.')

import uvicorn
from uvicorn import Config, Server

async def main():
    from app.di.container import ApplicationContainer
    from app.api.service import create_app

    container = ApplicationContainer()
    container.init_resources()
    app = create_app(container)

    config = Config(app=app, host='127.0.0.1', port=8766, log_level='error')
    server = Server(config=config)
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(3)

    def http_get(path):
        r = urllib.request.urlopen(f'http://127.0.0.1:8766{path}', timeout=10)
        return json.loads(r.read())

    def http_post(path, body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(f'http://127.0.0.1:8766{path}', data=data,
            headers={'Content-Type': 'application/json'})
        r = urllib.request.urlopen(req, timeout=60)
        return json.loads(r.read())

    print("=== Health Check ===")
    try:
        result = http_get('/health')
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Health failed: {e}")

    print("\n=== Query Test ===")
    try:
        result = http_post('/query', {'question': 'What is Article 14 of the Indian Constitution?', 'mode': 'standard'})
        print(f"Answer: {result['answer'][:300]}")
        print(f"Confidence: {result['confidence']}")
        print(f"Sources: {len(result['sources'])}")
    except Exception as e:
        print(f"Query error: {e}")

    print("\n=== Stats ===")
    try:
        result = http_get('/stats')
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Stats failed: {e}")

    server.should_exit = True
    await task

asyncio.run(main())
