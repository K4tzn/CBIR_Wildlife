import os
import json
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter

json_path = "desert_lion_camera_traps.json"  # Pfad zur JSON-Datei
image_base_path = "desert-lion-camera-traps-images"  # Ordner mit den Bildern
output_dir = "organized_dataset"  # Zielordner für sortierte Daten
test_size = 0.2  # Anteil der Daten im Testset


train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


with open(json_path, "r") as f:
    data = json.load(f)


# Bilder nach Kategorien sortieren
images = data["images"]
categories = {}

for img in images:
    file_name = img["file_name"]
    category = file_name.split("/")[0]  # Kategorie aus dem Dateipfad (z. B. 'acinonyx jubatus')
    if category not in categories:
        categories[category] = []
    categories[category].append(file_name)


# Daten aufteilen und kopieren
for category, files in categories.items():
    print(f"Organizing category: {category}")
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)

    for split, split_dir in [("train", train_dir), ("test", test_dir)]:
        category_dir = os.path.join(split_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        for file in (train_files if split == "train" else test_files):
            src = os.path.join(image_base_path, file)  # Vollständiger Pfad zum Bild
            if os.path.exists(src):  # Datei überprüfen
                # Kategorie zum Dateinamen hinzufügen
                original_name = os.path.basename(file)
                new_name = f"{category}_{original_name}"
                dst = os.path.join(category_dir, new_name)
                shutil.copy(src, dst)
            else:
                print(f"Datei nicht gefunden: {src}")

print(f"Dataset organized in '{output_dir}'.")



