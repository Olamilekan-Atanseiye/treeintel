# -*- coding: utf-8 -*-
r"""
Evaluate the model against a labeled dataset.
=================================================
Runs every image through the model and compares the prediction to its
true label (the folder it's in), then reports per-class accuracy, overall
accuracy, and a confusion matrix -- so you can see not just THAT the model
is wrong, but WHICH species it confuses with which.

Run check_class_order.py FIRST. If that finds a mismatch, fix it before
running this -- otherwise you'll be evaluating a labeling bug, not the
model.

Ideally, point this at a held-out test set the model never saw during
training (e.g. a 80/20 split you kept aside), not the training folders
themselves -- evaluating on training data will look better than the model
really performs.

Usage:
    python scripts/evaluate_model.py "C:\data\Species_JPG"

Optional: limit images per class for a quick check:
    python scripts/evaluate_model.py "G:\...\Species_JPG" --max-per-class 20
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="Path to the labeled dataset (one subfolder per class)")
    parser.add_argument("--max-per-class", type=int, default=None,
                         help="Only evaluate up to N images per class (faster spot-check)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        sys.exit(f"Dataset folder not found: {dataset_dir}")

    print("Loading model and CLASS_NAMES from app.py (this loads TensorFlow, may take a moment)...")
    from app import model, CLASS_NAMES, IMG_SIZE as APP_IMG_SIZE  # noqa: E402
    from tensorflow.keras.preprocessing import image as keras_image

    class_index = {name: i for i, name in enumerate(CLASS_NAMES)}

    y_true, y_pred, wrong_examples = [], [], []
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)

    class_folders = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    print(f"Found {len(class_folders)} class folders. Evaluating...\n")

    for folder in class_folders:
        true_class = folder.name
        if true_class not in class_index:
            print(f"  ! Skipping '{true_class}' -- not in CLASS_NAMES (check class order first)")
            continue

        image_paths = [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
        if args.max_per_class:
            image_paths = image_paths[: args.max_per_class]

        for img_path in image_paths:
            try:
                img = keras_image.load_img(img_path, target_size=APP_IMG_SIZE)
                arr = keras_image.img_to_array(img).astype("float32") / 255.0
                arr = np.expand_dims(arr, axis=0)
                pred = model.predict(arr, verbose=0)
                pred_idx = int(np.argmax(pred))
                pred_class = CLASS_NAMES[pred_idx]
            except Exception as e:
                print(f"  ! Failed on {img_path.name}: {e}")
                continue

            y_true.append(true_class)
            y_pred.append(pred_class)
            per_class_total[true_class] += 1
            if pred_class == true_class:
                per_class_correct[true_class] += 1
            else:
                wrong_examples.append((true_class, pred_class, img_path.name, float(pred[0][pred_idx] * 100)))

    if not y_true:
        sys.exit("No images were evaluated -- check the dataset path and folder names.")

    overall_acc = sum(per_class_correct.values()) / len(y_true)

    print("=" * 70)
    print(f"Overall accuracy: {overall_acc:.1%}  ({sum(per_class_correct.values())}/{len(y_true)})")
    print("=" * 70)

    print(f"\n{'Class':<35}{'Accuracy':<12}{'N'}")
    print("-" * 55)
    for cls in sorted(per_class_total):
        n = per_class_total[cls]
        correct = per_class_correct.get(cls, 0)
        acc = correct / n if n else 0
        acc_str = f"{acc:.1%}"
        flag = "  <-- LOW" if acc < 0.5 else ""
        print(f"{cls:<35}{acc_str:<12}{n}{flag}")

    if wrong_examples:
        print(f"\n{'='*70}\nSample misclassifications (up to 20):\n{'='*70}")
        for true_c, pred_c, fname, conf in wrong_examples[:20]:
            print(f"  {fname:<30} true={true_c:<25} predicted={pred_c:<25} ({conf:.1f}%)")

    # Confusion pairs: which classes get swapped most often
    confusion_pairs = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t != p:
            confusion_pairs[(t, p)] += 1

    if confusion_pairs:
        print(f"\n{'='*70}\nMost common confusions (true -> predicted : count):\n{'='*70}")
        for (t, p), count in sorted(confusion_pairs.items(), key=lambda x: -x[1])[:15]:
            print(f"  {t:<30} -> {p:<30} : {count}")

    try:
        from sklearn.metrics import classification_report
        print(f"\n{'='*70}\nFull classification report:\n{'='*70}")
        print(classification_report(y_true, y_pred, zero_division=0))
    except ImportError:
        print("\n(Install scikit-learn for a full precision/recall/F1 report: pip install scikit-learn)")


if __name__ == "__main__":
    main()
