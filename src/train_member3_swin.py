import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}...")

    # Paths & Hyperparameters
    DATA_DIR = "data"
    BATCH_SIZE = 16
    EPOCHS = 8
    LEARNING_RATE = 0.0001  # Lower LR for Transformer stability
    MODEL_SAVE_DIR = "models"
    MODEL_FILENAME = "swin_t.pth"

    # Data Augmentation & Normalization for Swin Transformer
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Datasets
    full_dataset = datasets.ImageFolder(root=DATA_DIR)
    num_classes = len(full_dataset.classes)
    
    # Train / Val Split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Apply respective transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Swin Transformer Model
    model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    
    # Replace head classifier for exact class count
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # AdamW optimizer prevents weight explosion in Transformers
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

    # Evaluation phase
    model.eval()
    all_preds = []
    all_labels = []
    start_time = time.time()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()
    total_samples = len(val_dataset)
    latency = ((end_time - start_time) / total_samples) * 1000  # ms per image

    # Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS (Swin Transformer - Swin-T)")
    print("="*50)
    print(f"Accuracy:     {accuracy * 100:.2f}%")
    print(f"F1-Score:     {f1:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"Latency:      {latency:.2f} ms/image")
    print("\nConfusion Matrix:")
    print(cm)
    print("="*50)

    # Save Model Weights
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
    torch.save(model.state_dict(), save_path)
    print(f"\n[SUCCESS] Saved weights directly to: {save_path}")

if __name__ == "__main__":
    main()