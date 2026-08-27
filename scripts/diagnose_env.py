# -*- coding: utf-8 -*-
"""
Diagnose why GROQ_API_KEY isn't being detected.
=================================================
Run from the project root:

    python scripts/diagnose_env.py

Never prints your actual key -- only whether it was found and how long it is,
so it's safe to paste the output back for help.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

print("=" * 60)
print(f"Looking for .env at: {ENV_PATH}")
print(f"Exists: {ENV_PATH.exists()}")

if ENV_PATH.exists():
    print(f"Size: {ENV_PATH.stat().st_size} bytes")
    raw = ENV_PATH.read_bytes()
    print(f"First 4 bytes (hex): {raw[:4].hex()}")
    if raw[:3] == b"\xef\xbb\xbf":
        print("!! File starts with a UTF-8 BOM -- this can break parsing on some")
        print("   dotenv versions. Re-save the file as plain UTF-8 (no BOM) in")
        print("   VS Code: bottom-right corner -> click the encoding -> 'Save with Encoding' -> UTF-8.")

    text = ENV_PATH.read_text(encoding="utf-8-sig", errors="replace")
    print("\n--- .env contents (values masked) ---")
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            print(line)
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            masked = f"{value.strip()[:4]}...({len(value.strip())} chars)" if value.strip() else "(EMPTY)"
            print(f"{key}={masked}")
        else:
            print(f"!! Line without '=': {line!r}")
else:
    print("\n!! .env not found at that exact path.")
    print("   Check File Explorer with 'File name extensions' turned on --")
    print("   it may actually be saved as '.env.txt'.")

print("=" * 60)
print("Now checking whether python-dotenv can actually load it...")

try:
    import dotenv
    from importlib.metadata import version as pkg_version
    try:
        print(f"python-dotenv version: {pkg_version('python-dotenv')}")
    except Exception:
        print("python-dotenv is installed (version lookup unavailable)")
except ImportError:
    print("!! python-dotenv is NOT installed. Run: pip install -r requirements.txt")
    raise SystemExit(1)

from dotenv import load_dotenv
loaded = load_dotenv(ENV_PATH)
print(f"load_dotenv() returned: {loaded}  (True means it found and parsed the file)")

key = os.environ.get("GROQ_API_KEY")
print("=" * 60)
if key and key != "your_groq_api_key_here":
    print(f"✅ GROQ_API_KEY is loaded ({len(key)} characters, starts with '{key[:4]}...')")
else:
    print("❌ GROQ_API_KEY is still not set after load_dotenv(). See details above.")
print("=" * 60)
