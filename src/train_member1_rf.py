import os
import time
import pickle
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
from torchvision import transforms, models

# Page configuration
st.set_page_config(page_title="DFU Severity Benchmark", layout="wide")

st.title("🩺 Diabetic Foot Ulcer (DFU) Severity Benchmark Dashboard")
st.write("Upload a foot image to evaluate severity predictions and compare model performance metrics in real time.")

# 1. Image Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = ["Healthy / Normal", "DFU Detected"]

# 2. Safe Model Loading Function
@st.cache_resource
def load_all_models():
    loaded_models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def find_target(name):
        for item in os.listdir("models"):
            if item.lower().startswith(name.lower()) and not item.endswith(".zip"):
                return os.path.join("models", item)
        return None

    # 1. MobileNetV2 (Tasmiya)
    mb_path = find_target("mobilenet_v2")
    if mb_path:
        try:
            mb = models.mobilenet_v2()
            mb.classifier[1] = nn.Linear(mb.last_channel, len(CLASS_NAMES))
            mb.load_state_dict(torch.load(mb_path, map_location=device, weights_only=False))
            mb.eval()
            loaded_models["MobileNetV2 (Tasmiya)"] = ("pytorch", mb, 99.53)
        except Exception as e:
            st.warning(f"Could not load MobileNetV2: {e}")

    # 2. EfficientNet-B0 (Sushmita)
    eff_path = find_target("efficientnet_b0")
    if eff_path:
        try:
            eff = models.efficientnet_b0()
            eff.classifier[1] = nn.Linear(eff.classifier[1].in_features, len(CLASS_NAMES))
            eff.load_state_dict(torch.load(eff_path, map_location=device, weights_only=False))
            eff.eval()
            loaded_models["EfficientNet-B0 (Sushmita)"] = ("pytorch", eff, 98.10)
        except Exception as e:
            st.warning(f"Could not load EfficientNet-B0: {e}")

    # 3. DenseNet-121 (Bharati)
    dn_path = find_target("densenet121")
    if dn_path:
        try:
            dn = models.densenet121()
            dn.classifier = nn.Linear(dn.classifier.in_features, len(CLASS_NAMES))
            dn.load_state_dict(torch.load(dn_path, map_location=device, weights_only=False))
            dn.eval()
            loaded_models["DenseNet-121 (Bharati)"] = ("pytorch", dn, 96.68)
        except Exception as e:
            st.warning(f"Could not load DenseNet-121: {e}")

    # 4. Random Forest (Shilpa)
    rf_path = find_target("random_forest")
    if rf_path:
        try:
            with open(rf_path, "rb") as f:
                rf = pickle.load(f)
            loaded_models["Random Forest (Shilpa)"] = ("sklearn", rf, 93.36)
        except Exception as e:
            st.warning(f"Skipping Random Forest: {e}")

    return loaded_models, device

loaded_models, device = load_all_models()

# 3. Streamlit UI
uploaded_file = st.file_uploader("Choose a foot image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("📊 Live Multi-Model Predictions")
        img_tensor = transform(image).unsqueeze(0).to(device)
        results = []

        for name, (model_type, model_obj, benchmark_acc) in loaded_models.items():
            start_time = time.time()

            if model_type == "pytorch":
                with torch.no_grad():
                    outputs = model_obj(img_tensor)
                    _, pred = torch.max(outputs, 1)
                    pred_class = CLASS_NAMES[pred.item()]
            else:
                flat_img = img_tensor.cpu().numpy().reshape(1, -1)
                if hasattr(model_obj, "n_features_in_"):
                    flat_img = flat_img[:, :model_obj.n_features_in_]
                pred = model_obj.predict(flat_img)
                pred_class = CLASS_NAMES[pred[0]]

            latency_ms = (time.time() - start_time) * 1000

            results.append({
                "Model Architecture": name,
                "Prediction": pred_class,
                "Live Latency (ms)": f"{latency_ms:.2f} ms",
                "Training Accuracy": f"{benchmark_acc}%" if isinstance(benchmark_acc, float) else benchmark_acc
            })

        st.table(results)
        st.success("🏆 **Primary Recommendation:** **MobileNetV2** is selected for production (99.53% Accuracy | ~26ms Latency).")