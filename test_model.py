import os
import random
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "Image_classifier_model.h5"

DATASET_DIR = BASE_DIR / "Species_JPG"

# Image you want to test
IMAGE_PATH = BASE_DIR / "IMG_6177.JPG"

IMG_SIZE = (224, 224)

# =====================================================
# LOAD MODEL
# =====================================================

print("=" * 60)
print("Loading model...")

model = load_model(str(MODEL_PATH))

print("Model loaded successfully!")
print("=" * 60)

# =====================================================
# CLASS NAMES
# =====================================================

class_names = ['105CANON', '105CANON_A', '107CANON', '108CANON', '110CANON', '111CANON', '112CANON', '113CANON', '114CANON', '115CANON', '116CANON', '117CANON', '118CANON', '119CANON', '121CANON', '123CANON', '124CANON', '125CANON', '127CANON', '128CANON', '129CANON', '131CANON', '132CANON', '133CANON', 'ATA', 'Aficulia', 'Alstonia Boonei', 'Caloncoba Glauci', 'Camarindus Indica', 'Carapa Procera', 'Entandrophragma Utili', 'Entandrophragma_angolense', 'Entandrophrangama Cylindrical', 'Indy', 'Irvingia Gabonensis', 'Kigelia africana', 'Mansonia altissima', 'Millettia_aboensis', 'Nauclea_diderrichii', 'Pentaclethra_macrophylla', 'Plumeria Alba', 'Polyalthia longifolia', 'Sp_1', 'Sub', 'Terminalia Catappa', 'Terminalia Mantaly', 'Terminalia Superba', 'Tetraploura_Tereptera', 'Treculia_africana', 'Triplochiton Scleroxylon', 'Unknow_small', 'White_flower_beside_house', 'Zik', 'red n black fruit']

print("Number of Classes:", len(class_names))

# =====================================================
# LOAD IMAGE
# =====================================================

img = image.load_img(
    IMAGE_PATH,
    target_size=IMG_SIZE
)

img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

img_array = np.expand_dims(
    img_array,
    axis=0
)

# =====================================================
# PREDICT
# =====================================================

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)

predicted_species = class_names[predicted_index]

confidence = prediction[0][predicted_index] * 100

print("\nPrediction Results")
print("------------------------------")
print("Species :", predicted_species)
print(f"Confidence : {confidence:.2f}%")

# =====================================================
# FIND TRAINING IMAGES
# =====================================================

species_folder = os.path.join(
    DATASET_DIR,
    predicted_species
)

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
)

training_images = [

    os.path.join(species_folder, file)

    for file in os.listdir(species_folder)

    if file.endswith(valid_extensions)

]

if len(training_images) >= 2:

    sample_images = random.sample(training_images, 2)

else:

    sample_images = training_images

# =====================================================
# DISPLAY RESULTS
# =====================================================

plt.figure(figsize=(18,6))

# Uploaded image
plt.subplot(1,3,1)

plt.imshow(img)

plt.axis("off")

plt.title(
    f"Uploaded Image\n\nPrediction:\n{predicted_species}\n\nConfidence: {confidence:.2f}%"
)

# Reference Image 1
if len(sample_images) >= 1:

    ref1 = image.load_img(
        sample_images[0],
        target_size=IMG_SIZE
    )

    plt.subplot(1,3,2)

    plt.imshow(ref1)

    plt.axis("off")

    plt.title("Reference Image 1")

# Reference Image 2
if len(sample_images) >= 2:

    ref2 = image.load_img(
        sample_images[1],
        target_size=IMG_SIZE
    )

    plt.subplot(1,3,3)

    plt.imshow(ref2)

    plt.axis("off")

    plt.title("Reference Image 2")

plt.tight_layout()

plt.show()
