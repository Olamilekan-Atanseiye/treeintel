from pathlib import Path
import json

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_DOCS = BASE_DIR / "knowledge_docs"
SPECIES_MAP_FILE = BASE_DIR / "species_map.json"


# ============================================================
# LOAD SPECIES MAP
# ============================================================

with open(SPECIES_MAP_FILE, "r", encoding="utf-8") as f:
    species_map = json.load(f)


# ============================================================
# MANUAL FILE CORRECTIONS
# Existing filename -> Expected filename
# ============================================================

FILE_RENAMES = {
    "Sp_2_Astonia_Boonie.docx": "Alstonia_Boonei.docx",

    "Camarindus Indica.docx": "Camarindus_Indica.docx",

    "Sp_7_Entandrophragma_Utili.docx":
        "Entandrophragma_Utili.docx",

    "Sp_8_Entandrophrangama_Cylindrical.docx":
        "Entandrophrangama_Cylindrical.docx",

    "Sp_6_Irvigia.docx":
        "Irvingia_Gabonensis.docx",

    "Sp_3_Mansonia_altisal.docx":
        "Mansonia_altissima.docx",

    "Sp_5_Termilala.docx":
        "Terminalia_Catappa.docx",

    "Tetrapleura_tetraptera.docx":
        "Tetraploura_Tereptera.docx",

    "Sp_4_Triplocyton.docx":
        "Triplochiton_Scleroxylon.docx",
}


# ============================================================
# RENAME FILES
# ============================================================

print("\nRENAMING KNOWLEDGE DOCUMENTS")
print("=" * 60)

for old_name, new_name in FILE_RENAMES.items():

    old_path = KNOWLEDGE_DOCS / old_name
    new_path = KNOWLEDGE_DOCS / new_name

    if old_path.exists():

        if new_path.exists():
            print(
                f"⚠ SKIPPED: {new_name} already exists"
            )

        else:
            old_path.rename(new_path)

            print(
                f"✓ RENAMED:\n"
                f"  {old_name}\n"
                f"  → {new_name}\n"
            )

    else:
        print(f"✗ SOURCE FILE NOT FOUND: {old_name}")


# ============================================================
# REMOVE MICROSOFT WORD TEMP FILES
# ============================================================

print("\nREMOVING WORD TEMPORARY FILES")
print("=" * 60)

for file in KNOWLEDGE_DOCS.glob("~$*.docx"):

    try:
        file.unlink()
        print(f"✓ Removed temporary file: {file.name}")

    except Exception as e:
        print(f"✗ Could not remove {file.name}: {e}")


# ============================================================
# CHECK REMAINING FILES
# ============================================================

print("\nCHECKING REMAINING MISSING FILES")
print("=" * 60)

missing_files = []

for class_name, filename in species_map.items():

    file_path = KNOWLEDGE_DOCS / filename

    if not file_path.exists():

        missing_files.append(
            (class_name, filename)
        )


if missing_files:

    print(
        f"\n⚠ {len(missing_files)} files still need attention:\n"
    )

    for class_name, filename in missing_files:

        print(
            f"- '{class_name}' -> {filename}"
        )

else:

    print(
        "\n✓ All species_map.json files exist!"
    )


# ============================================================
# FIND UNUSED DOCX FILES
# ============================================================

print("\nCHECKING UNUSED DOCUMENTS")
print("=" * 60)

referenced_files = set(species_map.values())

for doc_file in KNOWLEDGE_DOCS.glob("*.docx"):

    if doc_file.name.startswith("~$"):
        continue

    if doc_file.name not in referenced_files:

        print(f"⚠ Unused: {doc_file.name}")


print("\nDone!")