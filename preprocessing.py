from collections import Counter
import os
import json
import shutil
import random


# Pfad zur JSON-Datei
json_path = "desert_lion_camera_traps.json"

# Image-Ordner
images_path = "organized_dataset"

# subset 
subset_dir = "subset_dataset"


train_path = os.path.join(images_path, 'train')
subset_train_path = os.path.join(images_path, 'subset_train')


def show_datatypes(images_path):
    # Container für Dateitypen
    file_types = Counter()

    # Über alle Dateien im Ordner iterieren
    for root, dirs, files in os.walk(images_path):
        for file in files:
            _, ext = os.path.splitext(file)  # Dateiendung extrahieren
            file_types[ext.lower()] += 1  # Endung normalisieren (klein schreiben) und zählen

    # Ergebnis ausgeben
    print("Dateitypen und ihre Häufigkeiten:")
    for file_type, count in file_types.items():
        print(f"{file_type if file_type else 'Kein Typ'}: {count}")
    

def extract_non_jpg_files(images_path):
    # Zielordner für Nicht-JPG-Dateien
    non_jpg_dir = "non_jpg_files"

    # Sicherstellen, dass der Zielordner existiert
    os.makedirs(non_jpg_dir, exist_ok=True)

    # Dateien sammeln und verschieben
    for root, dirs, files in os.walk(images_path):
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            
            # Wenn die Datei keine .jpg (oder .jpeg) ist, verschiebe sie
            if ext.lower() not in ['.jpg', '.jpeg']:
                target_path = os.path.join(non_jpg_dir, file)
                shutil.move(file_path, target_path)
                print(f"Verschoben: {file_path} -> {target_path}")

    print("Verschieben abgeschlossen.")


def create_subset(train_path, subset_path, n_samples=10):
    """
    Erstelle ein Subset aus dem Trainingsdatensatz, das eine festgelegte Anzahl von Bildern pro Tierkategorie enthält.
    
    :param train_path: Der Pfad zum `train` Ordner des Originaldatensatzes.
    :param subset_path: Der Pfad zum Ordner, in dem das Subset gespeichert werden soll.
    :param n_samples: Die Anzahl an Bildern, die pro Tierkategorie ausgewählt werden sollen.
    """
    # Stelle sicher, dass der Zielordner existiert, wenn nicht, erstelle ihn
    if not os.path.exists(subset_path):
        os.makedirs(subset_path)

    # Durchlaufe alle Tierkategorien im train Ordner
    for category in os.listdir(train_path):
        category_path = os.path.join(train_path, category)
        
        # Überprüfe, ob der Ordner eine Kategorie ist
        if os.path.isdir(category_path):
            
            # Liste der Bilder in dieser Kategorie
            images = os.listdir(category_path)
            
            # Wenn weniger als n_samples Bilder vorhanden sind, wähle alle aus
            selected_images = random.sample(images, 10) if len(images) > 10 else images

            
            # Zielordner für diese Kategorie im subset erstellen
            category_subset_path = os.path.join(subset_path, category)
            if not os.path.exists(category_subset_path):
                os.makedirs(category_subset_path)
            
            # Ausgewählten Bilder in den neuen Ordner kopieren
            for image in selected_images:
                src_image_path = os.path.join(category_path, image)
                dst_image_path = os.path.join(category_subset_path, image)
                shutil.copy(src_image_path, dst_image_path)

    print(f"Subset erstellt und gespeichert in: {subset_path}")


# Aufruf der Methode mit n_samples=10
#create_subset(train_path, subset_train_path, n_samples=10)


def remove_apostrophes(root_folder):
    # Durchläuft alle Unterordner und Dateien
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if "'" in filename:
                new_filename = filename.replace("'", "")
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, new_filename)
                print(f"Umbenennen: {old_path} -> {new_path}")
                os.rename(old_path, new_path)


folder = "balanced_dataset"
#remove_apostrophes(folder)

def update_json_filenames(input_json_path, output_json_path):
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # Aktualisiere die Dateinamen in "images"
    if "images" in data:
        for img in data["images"]:
            if "file" in img and "'" in img["file"]:
                old_file = img["file"]
                new_file = old_file.replace("'", "")
                print(f"Updating image file: {old_file} -> {new_file}")
                img["file"] = new_file

    # Aktualisiere die Bild-IDs in "annotations"
    if "annotations" in data:
        for ann in data["annotations"]:
            if "image_id" in ann and "'" in ann["image_id"]:
                old_image_id = ann["image_id"]
                new_image_id = old_image_id.replace("'", "")
                print(f"Updating annotation image_id: {old_image_id} -> {new_image_id}")
                ann["image_id"] = new_image_id

    # Speichere die aktualisierte JSON-Datei im schreibbaren Bereich
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Updated JSON saved to {output_json_path}")

update_json_filenames("JSONs/test_annotations.json", "updated_test_annotations.json")