import os
from pathlib import Path
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "Image_classifier_model.h5"
DATASET_DIR = BASE_DIR / "Species_JPG"

IMG_SIZE = (224, 224)

# =====================================================
# CHECK PATHS
# =====================================================

print("=" * 60)
print("Loading Model...")

print("MODEL PATH:", MODEL_PATH)
print("MODEL EXISTS:", MODEL_PATH.exists())

print("DATASET PATH:", DATASET_DIR)
print("DATASET EXISTS:", DATASET_DIR.exists())

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found:\n{MODEL_PATH}")

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found:\n{DATASET_DIR}")

# =====================================================
# LOAD MODEL
# =====================================================

model = load_model(str(MODEL_PATH))

print("✅ Model Loaded Successfully")

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
print(f"✅ Classes Loaded: {len(class_names)}")

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_species(image_path):

    img = image.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = image.img_to_array(img)

    img_array = img_array.astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)

    predicted_species = class_names[predicted_index]

    confidence = float(prediction[0][predicted_index] * 100)

    return predicted_species, confidence
