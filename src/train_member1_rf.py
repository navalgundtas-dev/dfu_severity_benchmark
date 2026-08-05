import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join("data", "dataset") 
IMAGE_SIZE = (64, 64)

def load_data(data_dir):
    X, y = [], []
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for label_idx, category in enumerate(categories):
        cat_path = os.path.join(data_dir, category)
        for img_name in os.listdir(cat_path):
            img_path = os.path.join(cat_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                X.append(img.flatten())
                y.append(label_idx)
                
    return np.array(X), np.array(y), categories

if __name__ == "__main__":
    print("Loading dataset for Random Forest...")
    X, y, class_names = load_data(DATA_DIR)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*40)
    print(f"Random Forest Accuracy: {acc * 100:.2f}% | F1-Score: {f1:.4f}")
    print("="*40)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))