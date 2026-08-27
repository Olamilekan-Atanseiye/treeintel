# -*- coding: utf-8 -*-
r"""
Check CLASS_NAMES order against the dataset's folder order.
=================================================
Keras' flow_from_directory (and most other Keras data loaders) assign class
index 0, 1, 2... by SORTED folder name order -- not whatever order you typed
into CLASS_NAMES. If these two orders don't match exactly, the model itself
may be predicting correctly while app.py reports the wrong species name for
every single prediction. This is the first thing to rule out, before
suspecting the model.

Run from the project root:

    python scripts/check_class_order.py "C:\data\Species_JPG"

(pass your training dataset directory -- the one with one subfolder per class)
"""

import ast
import re
import sys
from pathlib import Path

APP_PY = Path(__file__).resolve().parent.parent / "app.py"


def load_class_names_from_app():
    """
    Parses CLASS_NAMES straight out of app.py's source text rather than
    importing app.py -- importing it would load TensorFlow and the model
    file, which is slow and would fail if the model/class-count mismatch
    from earlier hasn't been resolved yet. This check should work
    independently of that.
    """
    src = APP_PY.read_text(encoding="utf-8")
    match = re.search(r"CLASS_NAMES\s*=\s*\[(.*?)\]", src, re.S)
    if not match:
        sys.exit("Could not find CLASS_NAMES in app.py")
    return ast.literal_eval("[" + match.group(1) + "]")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python scripts/check_class_order.py <path_to_dataset_dir>")

    CLASS_NAMES = load_class_names_from_app()

    dataset_dir = Path(sys.argv[1])
    if not dataset_dir.exists():
        sys.exit(f"Dataset folder not found: {dataset_dir}")

    # This mirrors exactly what flow_from_directory does: list subfolders,
    # sort them alphabetically, that's the index order the model learned.
    folder_names = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())

    print("=" * 70)
    print(f"Dataset folders found: {len(folder_names)}")
    print(f"CLASS_NAMES entries:   {len(CLASS_NAMES)}")
    print("=" * 70)

    if len(folder_names) != len(CLASS_NAMES):
        print("MISMATCH: different counts. Fix this before checking order.")
        only_in_folders = set(folder_names) - set(CLASS_NAMES)
        only_in_classnames = set(CLASS_NAMES) - set(folder_names)
        if only_in_folders:
            print(f"\nFolders with no matching CLASS_NAMES entry ({len(only_in_folders)}):")
            for f in sorted(only_in_folders):
                print(f"  - {f}")
        if only_in_classnames:
            print(f"\nCLASS_NAMES entries with no matching folder ({len(only_in_classnames)}):")
            for c in sorted(only_in_classnames):
                print(f"  - {c}")
        sys.exit(1)

    mismatches = []
    for i, (folder, class_name) in enumerate(zip(folder_names, CLASS_NAMES)):
        if folder != class_name:
            mismatches.append((i, folder, class_name))

    if not mismatches:
        print("✅ CLASS_NAMES order exactly matches sorted folder order.")
        print("   This is NOT the source of any misprediction -- look at the")
        print("   model/data quality instead (see evaluate_model.py).")
    else:
        print(f"❌ {len(mismatches)} index/name mismatch(es) found:\n")
        print(f"{'Index':<6}{'Dataset folder (correct)':<35}{'Your CLASS_NAMES has':<35}")
        print("-" * 76)
        for i, folder, class_name in mismatches:
            print(f"{i:<6}{folder:<35}{class_name:<35}")
        print("\nThis means predictions ARE being systematically mislabeled.")
        print("Fix: set CLASS_NAMES in app.py to exactly this order:\n")
        print(folder_names)


if __name__ == "__main__":
    main()
