"""Convert the Keras classifier into the lightweight runtime model.

Run this only in a development environment with TensorFlow installed:
    python scripts/convert_model_to_tflite.py
"""

from pathlib import Path

import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_MODEL = BASE_DIR / "models" / "Image_classifier_model.h5"
OUTPUT_MODEL = BASE_DIR / "models" / "Image_classifier_model.tflite"


def main():
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(f"Keras model not found: {SOURCE_MODEL}")

    model = tf.keras.models.load_model(SOURCE_MODEL, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Keep float weights so the exported model preserves the classifier's
    # confidence scores. Runtime memory still drops substantially because
    # Render uses the small TensorFlow Lite runtime instead of TensorFlow.
    OUTPUT_MODEL.write_bytes(converter.convert())

    print(f"Wrote {OUTPUT_MODEL} ({OUTPUT_MODEL.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
