"""
Legal AI - Offline-First Legal Desktop Application

Entry point. Starts either the GUI or the API server based on CLI arguments.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.di.container import ApplicationContainer
from app.infra.config import Settings
from app.infra.logging import logger


def run_api(container: ApplicationContainer) -> None:
    import uvicorn
    from app.api.service import create_app

    settings = container.config()
    app = create_app(container)
    logger.info(
        "Starting API server on %s:%s",
        settings.api_host,
        settings.api_port,
    )
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )


def run_gui(container: ApplicationContainer) -> None:
    from app.ui.main_window import run_ui
    run_ui(container)


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal AI Desktop Application")
    parser.add_argument(
        "--mode",
        choices=["gui", "api"],
        default="gui",
        help="Start in GUI or API mode (default: gui)",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default=None,
        help="Path to settings file (optional)",
    )
    args = parser.parse_args()

    try:
        container = ApplicationContainer()
        container.init_resources()
        container.wire(modules=["app.api.service", "app.ui.main_window"])

        settings = container.config()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.raw_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == "api":
            run_api(container)
        else:
            run_gui(container)
    except Exception as e:
        logger.critical("Failed to start application: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
