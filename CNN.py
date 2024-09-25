import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd

# Chargement des données des kanjis
train_data = pd.read_csv("/home/etud/Bureau/projet_python/kanji_train_data.csv", sep=",", header=None)
train_labels = pd.read_csv("/home/etud/Bureau/projet_python/kanji_train_target.csv",sep=",", header=None)
test_data = pd.read_csv("/home/etud/Bureau/projet_python/kanji_test_data.csv", sep=",",header=None)

# Convertir les données en tableaux numpy
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)

# Prétraitement des données
img_width, img_height = 64, 64  # Nouvelles dimensions des images

# Redimensionner les données d'entraînement et de test
train_data_resized = []
for img in train_data:
    img_tensor = torch.tensor(img.reshape(64, 64)).unsqueeze(0).unsqueeze(0).float()
    img_resized = F.interpolate(img_tensor, size=(img_height, img_width), mode='bicubic')
    train_data_resized.append(img_resized.squeeze().numpy())
train_data_resized = np.array(train_data_resized)

test_data_resized = []
for img in test_data:
    img_tensor = torch.tensor(img.reshape(64, 64)).unsqueeze(0).unsqueeze(0).float()
    img_resized = F.interpolate(img_tensor, size=(img_height, img_width), mode='bicubic')
    test_data_resized.append(img_resized.squeeze().numpy())
test_data_resized = np.array(test_data_resized)

# Normaliser les données entre 0 et 1
train_data_resized = train_data_resized.astype('float32') / 255
test_data_resized = test_data_resized.astype('float32') / 255

# Convertir les étiquettes en un vecteur torch approprié
train_labels = torch.tensor(train_labels.flatten(), dtype=torch.long)

# Affichage des dimensions pour vérification
print("Dimensions des données d'entraînement :", train_data_resized.shape)
print("Dimensions des étiquettes d'entraînement :", train_labels.shape)
print("Dimensions des données de test :", test_data_resized.shape)

# Définition du modèle CNN pour les kanjis


# Définition de la classe CNN pour les kanjis
class CNNKanjis(nn.Module):
    def __init__(self):
        super(CNNKanjis, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
       
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Modification de la taille des couches linéaires
        self.fc2 = nn.Linear(128, 20)  # Adapté pour 20 classes de kanjis

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
    
        x = x.view(-1, 64 * 14 * 14)  # Réajustement de la taille avant les couches linéaires
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Création d'une instance du modèle CNN pour les kanjis
model_kanjis = CNNKanjis()

# Spécification du matériel utilisé
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model_kanjis.to(device)

# Définition de la fonction de perte et de l'optimiseur
criterion_kanjis = nn.CrossEntropyLoss()
#optimizer_kanjis = optim.Adam(model_kanjis.parameters(), lr=0.001)
optimizer_kanjis = optim.SGD(model_kanjis.parameters(), lr=0.01)

# Conversion des données en tenseurs torch
train_data_tensor = torch.tensor(train_data_resized, dtype=torch.float32).unsqueeze(1)
test_data_tensor = torch.tensor(test_data_resized, dtype=torch.float32).unsqueeze(1)

# Création des datasets
train_dataset_kanjis = TensorDataset(train_data_tensor, train_labels)
test_dataset_kanjis = TensorDataset(test_data_tensor, torch.tensor(test_data[:,0], dtype=torch.long))  # Ajout des étiquettes

# Création des dataloaders
trainloader_kanjis = DataLoader(train_dataset_kanjis, batch_size=64, shuffle=True)
testloader_kanjis = DataLoader(test_dataset_kanjis, batch_size=64, shuffle=False)

# Entraînement du modèle pour les kanjis
epochs = 2
for epoch in range(epochs):
    model_kanjis.train()
    running_loss = 0.0
    for images, labels in tqdm(trainloader_kanjis):
        images, labels = images.to(device), labels.to(device)
        optimizer_kanjis.zero_grad()
        outputs = model_kanjis(images)
        loss = criterion_kanjis(outputs, labels)
        loss.backward()
        optimizer_kanjis.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader_kanjis):.4f}")

# Évaluation du modèle pour les kanjis
model_kanjis.eval()
predictions = []
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(testloader_kanjis):
        images, labels = images.to(device), labels.to(device)
        outputs = model_kanjis(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        correct += (predicted == labels).sum().item()  # Calcul du nombre de prédictions correctes
        total += labels.size(0)  # Calcul du nombre total d'échantillons dans le test set

# Calcul de l'accuracy
accuracy = correct / total
print(f"Accuracy on test set: {100 * accuracy:.2f}%")

# Enregistrer toutes les prédictions dans un fichier CSV
np.savetxt("kanji_test_predictions_cnn.csv", predictions, delimiter=",")

# Affichage de l'occurrence
print(f"Occurrence: {accuracy}")



# Convertir les prédictions en numpy array
