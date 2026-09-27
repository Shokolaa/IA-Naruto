import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. Définition du Dataset PyTorch
# ---------------------------------------------------------
class NarutoDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ---------------------------------------------------------
# 2. Architecture du Réseau (MLP)
# ---------------------------------------------------------
class NarutoHandSignClassifier(nn.Module):
    def __init__(self, input_dim=126, num_classes=4, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 3. Fonction d'entraînement
# ---------------------------------------------------------
def train(epochs=40, batch_size=32, lr=0.001):
    csv_file = "naruto_dataset.csv"
    if not os.path.exists(csv_file):
        print(f"Erreur : Le fichier {csv_file} est introuvable. Collecte des données d'abord !")
        return

    # Lecture des données
    df = pd.read_csv(csv_file)
    if len(df) == 0:
        print("Erreur : Le fichier CSV est vide.")
        return

    X = df.iloc[:, :126].values
    y = df["label"].values

    num_classes = len(np.unique(y))
    print(f"Dataset charge : {len(df)} echantillons | {num_classes} classes detectees.")

    # Split Train / Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # DataLoaders
    train_dataset = NarutoDataset(X_train, y_train)
    val_dataset = NarutoDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Configuration GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrainement execute sur : {device}")

    # Initialisation du modèle
    model = NarutoHandSignClassifier(input_dim=126, num_classes=num_classes).to(device)

    # Fonction de perte et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0

    print(f"\n--- Debut de l'entrainement ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        # Phase Train
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        epoch_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100

        # Phase Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        epoch_val_loss = val_loss / total_val
        val_acc = (correct_val / total_val) * 100

        # Affichage regulier
        if epoch % 5 == 0 or epoch == epochs or epochs <= 10:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | "
                  f"Train Loss: {epoch_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        # Sauvegarde des meilleurs poids
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'input_dim': 126
            }, "naruto_model.pth")

    print("\n--- Entrainement termine ---")
    print(f"Meilleure precision en validation : {best_val_acc:.2f}%")
    print("Modele sauvegarde dans : naruto_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrainement du classifieur de signes Naruto")
    parser.add_argument("--epochs", "-e", type=int, default=40, help="Nombre d'epochs pour l'entrainement (defaut: 40)")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Taille des batchs (defaut: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (defaut: 0.001)")
    
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)