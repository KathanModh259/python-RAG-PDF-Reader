#!/usr/bin/env python3
"""
PyInstaller-based packaging script for Legal AI.

Produces a standalone Windows .exe in one-folder mode.

Usage:
    python scripts/package.py [--onefile] [--name NAME]
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Package Legal AI as Windows .exe")
    parser.add_argument("--onefile", action="store_true", help="Build single .exe (not recommended for large models)")
    parser.add_argument("--name", default="LegalAI", help="Output executable name")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "dist"
    spec_path = repo_root / "legal-ai.spec"

    pyinstaller_args = [
        sys.executable, "-m", "PyInstaller",
        "--name", args.name,
        "--noconfirm",
        "--clean",
        "--add-data", f"{repo_root / 'app'}{os.pathsep}app",
        "--add-data", f"{repo_root / 'data'}{os.pathsep}data",
        "--hidden-import", "app.di.container",
        "--hidden-import", "app.api.service",
        "--hidden-import", "app.ui.main_window",
        "--hidden-import", "chromadb",
        "--hidden-import", "sentence_transformers",
        "--hidden-import", "llama_cpp",
    ]

    if not args.onefile:
        pyinstaller_args.append("--onedir")
    else:
        pyinstaller_args.append("--onefile")
        pyinstaller_args.append("--upx-dir")  # optional

    pyinstaller_args.append(str(repo_root / "app" / "main.py"))

    print("Running PyInstaller...")
    subprocess.check_call(pyinstaller_args, cwd=repo_root)

    print(f"\nBuild complete. Output: {output_dir / args.name}")
    print("To create installer, run the Inno Setup script in installer/")


if __name__ == "__main__":
    main()
