import streamlit as st
import numpy as np
import faiss
import os
import torch
from PIL import Image
from torchvision import transforms
from fastai.learner import load_learner
from collections import Counter
from sklearn.preprocessing import normalize
import pathlib

# Windows compatibility for FastAI's PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Model and feature paths
MODEL_PATHS = {
    "GoogLeNet (bs128+aug)": "models/googlenet_bs128+aug.pkl",
    "ResNet50 (bs16+aug+iNat)": "models/resnet50_bs16+aug+iNat.pkl",
    "ViT (bs64+augm+iNat)": "models/ViT_bs64+augm+iNat.pkl"
}

FEATURE_PATHS = {
    "GoogLeNet (bs128+aug)": "features/features_googlenet_bs128+aug_with_path_FIXED.npz",
    "ResNet50 (bs16+aug+iNat)": "features/features_resnet50_bs16+aug+iNat_with_path_FIXED.npz",
    "ViT (bs64+augm+iNat)": "features/features_ViT_bs64+augm+iNat_with_path_FIXED.npz"
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Feature extraction with FastAI Learner
def extract_features(learner, image):
    image = transform(image).unsqueeze(0)
    learner.model.eval()
    with torch.no_grad():
        features = learner.model(image).squeeze().numpy()
    return features.astype('float32')

# Streamlit UI
st.set_page_config(page_title="CBIR Demo", layout="wide")
st.title("🔍 CBIR for Biodiversity Research in the Desert of Northern Namibia")

# Expandable species list
with st.expander("📚 Show list of supported species (37)"):
    species_list = [
        ("acinonyx jubatus", "cheetah"),
        ("antidorcas marsupialis", "springbok"),
        ("canis mesomelas", "black-backed jackal"),
        ("cn-francolins", "francolins (unspecified)"),
        ("cn-owls", "owls (unspecified)"),
        ("cn-raptors", "birds of prey (unspecified)"),
        ("columbidae", "pigeons/doves"),
        ("corvus albus", "pied crow"),
        ("corvus capensis", "cape crow"),
        ("crocuta crocuta", "spotted hyena"),
        ("diceros bicornis", "black rhinoceros"),
        ("equus asinus", "donkey"),
        ("equus zebra hartmannae", "hartmann's mountain zebra"),
        ("eupodotis rueppellii", "rüppell's bustard"),
        ("giraffa camelopardalis", "giraffe"),
        ("hyaena brunnea", "brown hyena"),
        ("hystrix africaeaustralis", "cape porcupine"),
        ("lepus capensis", "cape hare"),
        ("loxodanta africana", "african elephant"),
        ("mellivora capensis", "honey badger"),
        ("neotis ludwigii", "ludwig's bustard"),
        ("numididae", "guineafowl"),
        ("oreotragus oreotragus", "klipspringer"),
        ("oryx gazella", "gemsbok"),
        ("otocyon megalotis", "bat-eared fox"),
        ("panthera leo", "lion"),
        ("panthera pardus", "leopard"),
        ("papio anubis", "olive baboon"),
        ("procavia capensis", "rock hyrax"),
        ("pronolagus randensis", "namibian red rock hare"),
        ("pteroclidae", "sandgrouse"),
        ("raphiceros campestris", "steenbok"),
        ("struthio camelus", "ostrich"),
        ("torgos tracheliotos", "lappet-faced vulture"),
        ("tragelaphus strepsiceros", "greater kudu"),
        ("vulpes chama", "cape fox")
    ]
    for sci, eng in species_list:
        st.markdown(f"- *{sci}* — **{eng}**")

# UI controls
model_choice = st.selectbox("📌 Select a model:", list(MODEL_PATHS.keys()))
k = st.slider("🔢 Number of similar images (k):", 1, 10, 5)
uploaded_file = st.file_uploader("📤 Upload a query image:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Query Image", use_container_width=False, width=300)

    # Load model and features
    learner = load_learner(MODEL_PATHS[model_choice])
    data = np.load(FEATURE_PATHS[model_choice], allow_pickle=True)
    features = data["features"].astype("float32")
    labels = data["labels"]
    paths = data["paths"]

    # Normalize features for cosine similarity
    features = normalize(features, axis=1)

    # Extract and normalize query vector
    query_image = Image.open(uploaded_file).convert("RGB")
    query_vector = extract_features(learner, query_image)
    query_vector = normalize(query_vector.reshape(1, -1), axis=1)

    # FAISS cosine similarity via inner product
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    S, I = index.search(query_vector, k)  # S = similarity scores

    # Prepare results sorted by similarity
    results = list(zip(S[0], I[0]))
    results.sort(key=lambda x: -x[0])  # descending similarity

    st.subheader("📷 Top-k similar images (sorted by cosine similarity):")
    cols = st.columns(k)
    for i, col in enumerate(cols):
        similarity, idx = results[i]
        label = labels[idx]
        path = paths[idx]
        try:
            col.image(path, use_container_width=True)
            col.markdown(f"**{label}**")
            col.markdown(f"Cosine similarity: `{similarity:.4f}`")
        except:
            col.write(f"⚠️ Image not found:\n{path}")
            col.write(f"Cosine similarity: {similarity:.4f}")

    # Classification suggestion based on top-k
    st.subheader("📌 Classification suggestion:")
    top_k_labels = [labels[idx] for _, idx in results]
    label_counts = Counter(top_k_labels)
    total = sum(label_counts.values())
    top3 = label_counts.most_common(3)

    for label, count in top3:
        percent = 100 * count / total
        st.markdown(f"- **{label}** ({percent:.1f}%)")

    if len(top3) > 1 and top3[0][1] == top3[1][1]:
        st.markdown("⚠️ *Uncertain suggestion: multiple classes have the same top count.*")
