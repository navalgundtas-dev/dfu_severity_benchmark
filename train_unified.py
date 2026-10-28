import argparse
import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders(data_dir, batch_size=32, val_split=0.2):
    # Match your exact folder path: data/dataset
    dataset_path = os.path.join(data_dir, "dataset") if not data_dir.endswith("dataset") else data_dir
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    
    # Calculate train/val split sizes
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Loaded dataset from: {dataset_path}")
    print(f"Detected Classes & Index Mapping: {full_dataset.class_to_idx}")
    return train_loader, val_loader, full_dataset


def build_and_freeze_model(model_name):
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(model.last_channel, 2)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)

    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 2)

    elif model_name == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 2)

    return model.to(device)


def train_pytorch_model(model_name, epochs, data_dir, save_path):
    train_loader, val_loader, full_dataset = get_data_loaders(data_dir)
    model = build_and_freeze_model(model_name)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    print(f"\n🚀 Fast-Training {model_name} for {epochs} epochs on {device}...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss, correct = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += torch.sum(preds == labels.data)

        train_acc = correct.double() / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader.dataset):.4f} | Train Acc: {train_acc*100:.2f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "class_to_idx": full_dataset.class_to_idx
    }, save_path)
    print(f"✅ Saved {model_name} to {save_path} in {time.time() - start_time:.2f} seconds!")


def train_random_forest(data_dir, save_path):
    from sklearn.ensemble import RandomForestClassifier

    print("\n🚀 Training Random Forest...")
    train_loader, _, _ = get_data_loaders(data_dir, batch_size=64)

    X_list, y_list = [], []
    for inputs, labels in train_loader:
        inputs_resized = nn.functional.interpolate(inputs, size=(64, 64)).numpy().reshape(inputs.size(0), -1)
        X_list.append(inputs_resized)
        y_list.append(labels.numpy())

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"✅ Saved Random Forest to {save_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mobilenet_v2", "efficientnet_b0", "densenet121", "swin_t", "random_forest"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default="data/dataset")
    args = parser.parse_args()

    output_filename = os.path.join("models", f"{args.model}_unified.pth")
    if args.model == "random_forest":
        train_random_forest(args.data_dir, os.path.join("models", "rf_unified.pkl"))
    else:
        train_pytorch_model(args.model, args.epochs, args.data_dir, output_filename)