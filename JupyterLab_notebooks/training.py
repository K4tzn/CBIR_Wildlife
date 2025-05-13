import numpy as np 
import pandas as pd 
import tensorflow as tf
import os
from PIL import ImageFile
import os, json, pandas as pd
from fastai.vision.all import *
import matplotlib.pyplot as plt
import torch

from torchvision.models import resnet50, vgg16, vgg19, googlenet, alexnet
import neptune

# === Pfade definieren ===

DATASET_DIR = "balanced_dataset_split/balanced_dataset_split" 
JSON_PATH = "train_test_split_new.json"  
OUTPUT_DIR = "basemodel_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXTRA_IMAGES_DIR = "augmented_dataset_new/augmented_dataset_new" # Augmentierte Bilder für unterrepräsentierte Klassen

ImageFile.LOAD_TRUNCATED_IMAGES = True

# === JSON mit Train/Test-Split laden ===
with open(JSON_PATH, "r") as f:
    split_data = json.load(f)

# DataFrames für Train/Test erstellen
df_train = pd.DataFrame(split_data["train"])
df_test = pd.DataFrame(split_data["test"])

# vollständigen Bildpfad setzen
df_train["image_path"] = df_train["file_path"].apply(lambda x: os.path.join(DATASET_DIR, x))
df_test["image_path"] = df_test["file_path"].apply(lambda x: os.path.join(DATASET_DIR, x))


print(f"Trainingsdaten: {len(df_train)} Bilder, Testdaten: {len(df_test)} Bilder")
print(df_train.head())

# Vokabular aus Labels erstellen
vocab = df_train["label"].unique().tolist()
#vocab = df_train_sampled["label"].unique().tolist()

batch_size = 256

device_id = 3
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
print(device)

# === DataBlock für das Training definieren ===
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock(vocab=vocab)),
    get_x=ColReader('image_path'),
    get_y=ColReader('label'),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),  # 20% als Validierung
    item_tfms=Resize(224),
)
#dls = dblock.dataloaders(df_train_sampled, bs=8, num_workers=8, device="cuda")
dls = dblock.dataloaders(df_train, bs=batch_size, num_workers=8, device=device)
dls.show_batch(max_n=6)

# === Test DataLoader erstellen ===
#test_dl = dls.test_dl(df_test_sampled, device="cuda", with_labels=True)
test_dl = dls.test_dl(df_test, device="cuda", with_labels=True)

print(f"Train set size: {len(dls.train_ds)}")
print(f"Validation set size: {len(dls.valid_ds)}")

# === Neptune.ai initialisieren ===
run = neptune.init_run(
    project="insert_your_project_here",
    api_token="insert_your_api_token_here",
    tags=["googlenet_bs256"]
)

# Logge allgemeine Parameter
run["global/parameters"] = {
    "dataset_dir": DATASET_DIR,
    "json_path": JSON_PATH,
    "batch_size": batch_size,
    "epochs": 100,
}

models_to_eval = {
    #"resnet50": resnet50, 
    #"vgg16": vgg16_bn, 
    #"vgg19": vgg19_bn,
    "googlenet": googlenet
    #"alexnet": alexnet
}

config = {"epochs": 100}

results = {}  # Testgenauigkeiten der Modelle

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Schleife über die Modelle ===
for name, arch in models_to_eval.items():
    try:
        print(f"\n=== Training und Evaluation für {name} ===")
        run[f"{name}/start"] = "Training gestartet"

        learn = vision_learner(dls, arch, metrics=accuracy, pretrained=True,
                       cbs=[EarlyStoppingCallback(monitor='valid_loss', patience=5)]).to_fp16()

        # Neptune: Modellarchitektur speichern
        run[f"{name}/model_summary"] = str(learn.model)

        # Lernrate suchen und loggen
        lr_find_results = learn.lr_find(start_lr=1e-6, end_lr=1e-2, show_plot=False)
        optimal_lr = lr_find_results.valley
        print(f"{name}: empfohlene Lernrate = {optimal_lr}")
        run[f"{name}/optimal_lr"] = optimal_lr

        
        # === Training starten (einmal für alle Epochen) ===
        learn.fit_one_cycle(config["epochs"], lr_max=optimal_lr)
        
        # Train-Loss pro Epoche loggen
        for epoch, loss in enumerate(learn.recorder.losses):
            run[f"{name}/train_loss"].append(loss.item())  # Neptune speichert als Chart
        
        # Validation-Loss & Accuracy pro Epoche loggen
        if hasattr(learn.recorder, "values") and len(learn.recorder.values) > 0:
            for epoch, values in enumerate(learn.recorder.values):
                if len(values) > 1:  # Falls Valid-Loss existiert
                    run[f"{name}/valid_loss"].append(values[1])
                if len(values) > 2:  # Falls Accuracy existiert
                    run[f"{name}/accuracy"].append(values[2])

        # Neptune-Logging überprüfen
        print(f"Logging abgeschlossen für {name}.")


        # Entferne Early Stopping nach dem Training
        for cb in learn.cbs:
            if isinstance(cb, EarlyStoppingCallback):
                learn.remove_cb(cb)

        # Test Evaluation
        test_loss, test_acc = learn.validate(dl=test_dl)
        print(f"{name}: Testgenauigkeit = {test_acc:.4f}")
        results[name] = test_acc
        run[f"{name}/test_accuracy"] = test_acc

        # Confusion Matrix speichern
        interp = ClassificationInterpretation.from_learner(learn, dl=test_dl)
        interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
        plt.title(f"Confusion Matrix: {name}_bs{batch_size}")
        cm_path = os.path.join(OUTPUT_DIR, f"cm_{name}_{batch_size}.png")
        plt.savefig(cm_path)
        plt.close()
        run[f"{name}/confusion_matrix"].upload(cm_path)

        # Modell speichern
        model_path = os.path.join(OUTPUT_DIR, f"basemodel_{name}_bs{batch_size}.pkl")
        learn.export(model_path)
        print(f"{name} gespeichert unter: {model_path}")
        run[f"{name}/model_checkpoint"].upload(model_path)

    except Exception as e: 
        print(f"Fehler in {name}: {e}")
        run[f"{name}/error"] = str(e)
