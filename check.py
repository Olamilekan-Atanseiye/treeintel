# -*- coding: utf-8 -*-
"""
Project setup checker for TreeIntel
=================================================
Run this from the project root (next to app.py) to sanity-check that
everything is where it should be before starting the Flask app:

    python check_setup.py

It never modifies anything -- it only reports. Fix whatever it flags, then
re-run it until everything passes.
"""

import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CHECK = "  \u2713"   # ✓
CROSS = "  \u2717"   # ✗
WARN = "  \u26a0"    # ⚠

passed = []
failed = []
warnings = []


def ok(msg):
    passed.append(msg)
    print(f"{CHECK} {msg}")


def fail(msg):
    failed.append(msg)
    print(f"{CROSS} {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"{WARN} {msg}")


def section(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# =====================================================
# 1. CORE FILES / FOLDER LAYOUT
# =====================================================
section("1. Core files and folders")

required_files = ["app.py", "requirements.txt"]
for f in required_files:
    if (BASE_DIR / f).exists():
        ok(f"{f} found")
    else:
        fail(f"{f} is MISSING from the project root")

required_dirs = ["templates", "static", "models", "knowledge_docs", "scripts"]
for d in required_dirs:
    if (BASE_DIR / d).is_dir():
        ok(f"{d}/ folder found")
    else:
        fail(f"{d}/ folder is MISSING (should sit directly next to app.py)")

if (BASE_DIR / "templates" / "index.html").exists():
    ok("templates/index.html found")
else:
    fail("templates/index.html is MISSING -- this causes TemplateNotFound at runtime")

for f in ["style.css", "script.js"]:
    if (BASE_DIR / "static" / f).exists():
        ok(f"static/{f} found")
    else:
        fail(f"static/{f} is MISSING")


# =====================================================
# 2. CNN MODEL
# =====================================================
section("2. CNN model")

model_files = list((BASE_DIR / "models").glob("*.h5")) if (BASE_DIR / "models").is_dir() else []
if model_files:
    for m in model_files:
        ok(f"Model file found: models/{m.name}")
    if len(model_files) > 1:
        warn("Multiple .h5 files found in models/ -- make sure app.py's MODEL_PATH points to the right one")
else:
    fail("No .h5 model file found in models/ -- app.py will fail to start")


# =====================================================
# 3. KNOWLEDGE ENGINE FILES
# =====================================================
section("3. Knowledge engine (RAG + LLM)")

for f in ["knowledge_engine.py", "species_map.json", "scripts/build_knowledge_base.py"]:
    if (BASE_DIR / f).exists():
        ok(f"{f} found")
    else:
        fail(f"{f} is MISSING")

species_map = {}
map_path = BASE_DIR / "species_map.json"
if map_path.exists():
    try:
        species_map = json.loads(map_path.read_text(encoding="utf-8"))
        species_map.pop("_comment", None)
        ok(f"species_map.json is valid JSON ({len(species_map)} classes listed)")
    except json.JSONDecodeError as e:
        fail(f"species_map.json is not valid JSON: {e}")


# =====================================================
# 4. KNOWLEDGE DOCS <-> SPECIES MAP CONSISTENCY
# =====================================================
section("4. knowledge_docs/ vs species_map.json consistency")

docs_dir = BASE_DIR / "knowledge_docs"
mapped_count = 0
missing_files = []
unmapped_docs = []

if species_map and docs_dir.is_dir():
    mapped_filenames = set()
    for class_name, filename in species_map.items():
        if not filename:
            continue
        mapped_count += 1
        mapped_filenames.add(filename)
        if not (docs_dir / filename).exists():
            missing_files.append((class_name, filename))

    actual_docx_files = {p.name for p in docs_dir.glob("*.docx")}
    unmapped_docs = sorted(actual_docx_files - mapped_filenames)

    ok(f"{mapped_count} classes are mapped to a document in species_map.json")

    if missing_files:
        fail(f"{len(missing_files)} mapped file(s) not found in knowledge_docs/:")
        for class_name, filename in missing_files:
            print(f"      - '{class_name}' -> {filename}")
    else:
        ok("Every mapped filename exists in knowledge_docs/")

    if unmapped_docs:
        warn(f"{len(unmapped_docs)} .docx file(s) in knowledge_docs/ aren't referenced by any class:")
        for f in unmapped_docs:
            print(f"      - {f}")

    null_classes = [c for c, f in species_map.items() if not f]
    if null_classes:
        print(f"    ({len(null_classes)} classes intentionally set to null -- skipped, not an error)")
else:
    warn("Skipped consistency check (species_map.json or knowledge_docs/ missing)")


# =====================================================
# 5. VECTOR STORE / CACHE
# =====================================================
section("5. Vector store")

chroma_dir = BASE_DIR / "knowledge_base" / "chroma_db"
if chroma_dir.exists():
    ok("Vector store found at knowledge_base/chroma_db -- knowledge base has been built")
else:
    warn("Vector store not built yet -- run: python scripts/build_knowledge_base.py")


# =====================================================
# 6. ENVIRONMENT / API KEY
# =====================================================
section("6. Environment variables")

env_path = BASE_DIR / ".env"
if env_path.exists():
    ok(".env file found")
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        warn("python-dotenv not installed -- run: pip install -r requirements.txt")

    key = os.environ.get("GROQ_API_KEY", "")
    if key and key != "your_groq_api_key_here":
        ok("GROQ_API_KEY is set")
    else:
        fail("GROQ_API_KEY is missing or still the placeholder value in .env")
else:
    fail(".env file is MISSING -- copy .env.example to .env and add your real key")

gitignore_path = BASE_DIR / ".gitignore"
if gitignore_path.exists():
    content = gitignore_path.read_text(encoding="utf-8")
    if ".env" in content:
        ok(".gitignore exists and excludes .env")
    else:
        warn(".gitignore exists but does not exclude .env -- add a line with just: .env")
else:
    warn("No .gitignore found -- your .env / API key could get committed by accident")


# =====================================================
# SUMMARY
# =====================================================
section("Summary")
print(f"Passed:   {len(passed)}")
print(f"Warnings: {len(warnings)}")
print(f"Failed:   {len(failed)}")

if failed:
    print("\nFix the ✗ items above before running the app.")
    sys.exit(1)
elif warnings:
    print("\nNo blocking issues, but review the ⚠ items above.")
    sys.exit(0)
else:
    print("\nEverything looks good.")
    sys.exit(0)
