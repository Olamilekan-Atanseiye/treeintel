"""
TreeIntel Species Classifier - Flask Web Application
=================================================
Serves an upload page where a user submits a leaf/plant image and
receives the predicted species + confidence score from a trained
Keras (.h5) CNN model.

Run locally:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import io
import os
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from tensorflow.keras.models import load_model

import knowledge_engine

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

# Model ships inside the project (relative path) instead of a
# hardcoded machine-specific path. Place your retrained .h5 file at:
#   models/Image_classifier_model.h5
MODEL_PATH = BASE_DIR / "models" / "Image_classifier_model.h5"

IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB upload limit

# =====================================================
# CLASS NAMES
# =====================================================

CLASS_NAMES = [
    '105CANON', '105CANON_A', '107CANON', '108CANON', '110CANON',
    '111CANON', '112CANON', '113CANON', '114CANON', '115CANON',
    '116CANON', '117CANON', '118CANON', '119CANON', '121CANON',
    '123CANON', '124CANON', '125CANON', '127CANON', '128CANON',
    '129CANON', '131CANON', '132CANON', '133CANON', 'ATA',
    'Aficulia', 'Alstonia Boonei', 'Caloncoba Glauci', 'Camarindus Indica',
    'Carapa Procera', 'Entandrophragma Utili', 'Entandrophragma_angolense',
    'Entandrophrangama Cylindrical', 'Indy', 'Irvingia Gabonensis',
    'Kigelia africana', 'Mansonia altissima', 'Millettia_aboensis',
    'Nauclea_diderrichii', 'Pentaclethra_macrophylla', 'Plumeria Alba',
    'Polyalthia longifolia', 'Sp_1', 'Sub', 'Terminalia Catappa',
    'Terminalia Mantaly', 'Terminalia Superba', 'Tetraploura_Tereptera',
    'Treculia_africana', 'Triplochiton Scleroxylon', 'Unknow_small',
    'White_flower_beside_house', 'Zik', 'red n black fruit'
]

# =====================================================
# APP + MODEL INITIALIZATION (loaded ONCE at startup)
# =====================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

print("=" * 60)
print("Loading model from:", MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}.\n"
        "Copy your .h5 file into the 'models' folder before starting the app."
    )

model = load_model(str(MODEL_PATH))

model_output_units = model.output_shape[-1]
if model_output_units != len(CLASS_NAMES):
    raise ValueError(
        f"CLASS_NAMES has {len(CLASS_NAMES)} entries but the model's output "
        f"layer has {model_output_units} units. These must match exactly, "
        "or predictions will silently map to the wrong species. Update "
        "CLASS_NAMES to the real list from your training generator's "
        "class_indices, in the same order."
    )

print(f"Model loaded. {len(CLASS_NAMES)} classes registered (matches model output).")
print("=" * 60)


# =====================================================
# HELPERS
# =====================================================

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_species(pil_image: Image.Image):
    """Run inference on a PIL image and return (species, confidence%)."""
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    img_array = np.asarray(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(prediction))
    predicted_species = CLASS_NAMES[predicted_index]
    confidence = float(prediction[0][predicted_index] * 100)

    # Also return the top-5 so the UI can show alternatives
    top5_idx = np.argsort(prediction[0])[::-1][:5]
    top5 = [
        {"species": CLASS_NAMES[i], "confidence": float(prediction[0][i] * 100)}
        for i in top5_idx
    ]

    return predicted_species, confidence, top5


# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file included in the request."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    try:
        pil_image = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "Could not read image file. It may be corrupted."}), 400

    try:
        species, confidence, top5 = predict_species(pil_image)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "species": species,
        "confidence": round(confidence, 2),
        "top5": [{"species": t["species"], "confidence": round(t["confidence"], 2)} for t in top5],
    })


@app.route("/knowledge/overview", methods=["POST"])
def knowledge_overview():
    """
    Returns the cached (or freshly generated) structured write-up for a
    species, used to populate the AI Knowledge panel right after a
    prediction. Fails gracefully -- with a clear reason -- rather than
    crashing if the knowledge base isn't built or the species lacks docs.
    """
    payload = request.get_json(silent=True) or {}
    species = payload.get("species")

    if not species:
        return jsonify({"error": "Missing 'species' in request body."}), 400
    if species not in CLASS_NAMES:
        return jsonify({"error": f"Unknown species '{species}'."}), 400

    if not knowledge_engine.knowledge_base_ready():
        return jsonify({
            "available": False,
            "reason": "Knowledge base not built yet. Run scripts/build_knowledge_base.py.",
        }), 200

    if not knowledge_engine.has_documentation(species):
        return jsonify({
            "available": False,
            "reason": "No documentation is mapped for this species in species_map.json.",
        }), 200

    try:
        overview = knowledge_engine.get_overview(species)
    except Exception as e:
        return jsonify({"available": False, "reason": str(e)}), 200

    return jsonify({"available": True, "species": species, "overview": overview})


@app.route("/knowledge/ask", methods=["POST"])
def knowledge_ask():
    """
    Answers a free-form question about a specific species, scoped to that
    species' documents only (via metadata-filtered retrieval).
    """
    payload = request.get_json(silent=True) or {}
    species = payload.get("species")
    question = (payload.get("question") or "").strip()

    if not species or species not in CLASS_NAMES:
        return jsonify({"error": "Missing or unknown 'species'."}), 400
    if not question:
        return jsonify({"error": "Missing 'question'."}), 400

    if not knowledge_engine.knowledge_base_ready():
        return jsonify({
            "error": "Knowledge base not built yet. Run scripts/build_knowledge_base.py."
        }), 200

    try:
        answer = knowledge_engine.ask(species, question)
    except Exception as e:
        return jsonify({"error": str(e)}), 200

    return jsonify({"species": species, "question": question, "answer": answer})


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "num_classes": len(CLASS_NAMES),
        "knowledge_base_ready": knowledge_engine.knowledge_base_ready(),
    })


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    # debug=True is fine for local dev; turn off in production
    app.run(debug=True, host="127.0.0.1", port=5000)
