import torch
import torch.nn as nn

class NarutoHandSignClassifier(nn.Module):
    def __init__(self, input_dim=126, num_classes=12, dropout_rate=0.3):
        """
        input_dim : 126 si deux mains (2 * 21 points * 3 coordonnees x, y, z),
                    63 si une seule main.
        num_classes : nombre de signes a detecter (ex: les 12 mudras du zodiaque).
        """
        super().__init__()
        
        self.net = nn.Sequential(
            # Bloc 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Bloc 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Bloc 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Sortie (Logits bruts, CrossEntropyLoss applique le softmax en interne)
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instanciation du modele pour 12 signes (Tigre, Serpent, Dragon, etc.)
    model = NarutoHandSignClassifier(input_dim=126, num_classes=12).to(device)
    
    # Test avec un batch fictif de 8 echantillons
    dummy_input = torch.randn(8, 126, device=device)
    output = model(dummy_input)
    
    print(f"Modele deploye sur : {device}")
    print("Dimensions en sortie (batch_size, num_classes) :", output.shape)