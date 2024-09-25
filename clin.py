import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch as th
import torch.nn as nn
import torch.optim as optim

# Chargement et préparation des données
data_train = pd.read_csv("/home/etud/Bureau/projet_python/kanji_train_data.csv", sep=",", header=None)
data_target = pd.read_csv("/home/etud/Bureau/projet_python/kanji_train_target.csv", sep=",", header=None)

# Séparation des données en ensembles d'entraînement et de validation
X_train, X_test, y_train, y_test = train_test_split(data_train, data_target, test_size=0.2, random_state=42)

# Normalisation des données
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Normalisation des données de test
test_data = pd.read_csv("/home/etud/Bureau/projet_python/kanji_test_data.csv", sep=",", header=None)
test_mean = test_data.mean()
test_std = test_data.std()
test_data = (test_data - test_mean) / test_std

# Conversion en tenseur
X_t = th.tensor(X_train.values, dtype=th.float32)
y_t = th.tensor(y_train.values.squeeze(), dtype=th.long)
X_v = th.tensor(X_test.values, dtype=th.float32)
y_v = th.tensor(y_test.values.squeeze(), dtype=th.long)
t_t = th.tensor(test_data.values, dtype=th.float32)


class NeuralNetwork(nn.Module):
    def __init__(self, nb_colonnes, nb_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(nb_colonnes, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, nb_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

nb_colonnes = X_t.shape[1]
nb_classes = 20

# Initialisation du modèle
model = NeuralNetwork(nb_colonnes, nb_classes)

# Fonction de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Entraînement du modèle
num_iterations = 1000
for i in range(num_iterations):
    optimizer.zero_grad()
    outputs = model(X_t)
    loss = criterion(outputs, y_t)
    loss.backward()
    optimizer.step()

# Prédiction sur l'ensemble de test
with th.no_grad():
    logits = model(X_v)
    predictions = th.argmax(logits, dim=1)

# Calcul de la précision
accuracy = accuracy_score(y_test, predictions)
#accuracy = accuracy_score(y_test.numpy(), predictions.numpy())

print("Précision du modèle :", accuracy)

# Prédiction sur les données de test
with th.no_grad():
    logits = model(t_t)
    predictions = th.argmax(logits, dim=1)
    # Transformation en numpy
    predictions_numpy = predictions.numpy()
    np.savetxt("kanji_test_predictions.csv", predictions_numpy)
