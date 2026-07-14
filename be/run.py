"""Nyaya Mitra — Launch the web app."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    from app.di.container import ApplicationContainer
    from app.api.service import create_app
    import uvicorn

    print("Starting Nyaya Mitra...")
    container = ApplicationContainer()
    container.init_resources()
    container.wire(modules=["app.api.service"])
    app = create_app(container)

    print("Web app: http://127.0.0.1:8765")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
