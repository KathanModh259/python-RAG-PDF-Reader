# Building from Source

## Prerequisites

- Python 3.10+
- Poetry 1.5+
- Windows 10/11 SDK (for Windows builds)
- Inno Setup 6+ (for installer)
- Git

## Setup

```bash
# Clone the repository
git clone https://github.com/your-org/legal-ai.git
cd legal-ai

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## Build the Knowledge Base

```bash
# Download and index legal documents
python scripts/build_knowledge_base.py

# Or use your own documents
python scripts/build_knowledge_base.py --data-dir ./my_legal_docs
```

## Download the LLM Model

```bash
# Download the quantized GGUF model
python scripts/download_model.py
```

## Run in Development Mode

```bash
# GUI mode
python app/main.py --mode gui

# API mode (headless)
python app/main.py --mode api
```

## Run Tests

```bash
pytest
```

## Lint and Type Check

```bash
ruff check app/
mypy app/
```

## Package as .exe (Windows)

```bash
# Build the executable with PyInstaller
python scripts/package.py

# This creates dist/LegalAI/ with the standalone executable
```

## Create Installer

1. Open installer/setup.iss in Inno Setup
2. Compile (Ctrl+F9)
3. Output: installer/Output/LegalAI-Setup-{version}.exe

## Code Signing

If you have a code signing certificate:

```bash
signtool sign /fd SHA256 /a /f certificate.pfx /p password dist/LegalAI/LegalAI.exe
```

## CI/CD

The repository includes GitHub Actions workflows (add your own):

- `.github/workflows/test.yml` — Run tests on PR
- `.github/workflows/build.yml` — Build installer on tag
