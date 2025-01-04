import pickle
import torch

filepath = "train/checkpoint2.pth"  # ou le chemin de ton fichier
try:
    with open(filepath, 'rb') as f:
        data = torch.load(f, map_location='cpu')
    print("Contenu du fichier:", data)
except Exception as e:
    print(f"Erreur lors de la lecture du fichier avec pickle: {e}")
