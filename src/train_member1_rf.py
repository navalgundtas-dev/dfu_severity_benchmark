import os
import time
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import joblib

DATA_DIR = os.path.join("data", "dataset")
MODEL_SAVE_DIR = "models"
MODEL_FILENAME = "random_forest.pkl"

def load_images_for_rf(data_dir):
    images, labels = [], []
    classes = sorted(os.listdir(data_dir))
    for label_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).resize((64, 64)).convert("L")
                images.append(np.array(img).flatten())
                labels.append(label_idx)
            except Exception:
                pass
    return np.array(images), np.array(labels)

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found!")
        return

    print("Loading images for Random Forest baseline...")
    X, y = load_images_for_rf(DATA_DIR)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("Training Random Forest...")
    model.fit(X_train, y_train)

    start_time = time.time()
    preds = model.predict(X_val)
    total_time_ms = (time.time() - start_time) * 1000
    latency = total_time_ms / len(X_val)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='weighted', zero_division=0)
    precision = precision_score(y_val, preds, average='weighted', zero_division=0)
    recall = recall_score(y_val, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(y_val, preds)

    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS (Random Forest)")
    print("="*50)
    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"F1-Score:        {f1:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"Latency:         {latency:.2f} ms/image")
    print("\nConfusion Matrix:")
    print(cm)
    print("="*50)

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
    joblib.dump(model, save_path)
    print(f"\n[SUCCESS] Model saved directly to: {save_path}")

if __name__ == "__main__":
    main()