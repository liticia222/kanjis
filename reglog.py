import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Chargement des données
train_data = pd.read_csv("/home/etud/Bureau/projet_python/kanji_train_data.csv", sep=",", header=None)
train_target = pd.read_csv("/home/etud/Bureau/projet_python/kanji_train_target.csv", sep=",", header=None)

# Fractionnement des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.2, random_state=42)

# Réduction de la dimensionnalité avec PCA
pca = PCA(n_components=0.95)  # Garder 95% de la variance
X_train_pca = pca.fit_transform(X_train)
#Les données d'entraînement et de test sont transformées en utilisant les composantes principales obtenues.
X_test_pca = pca.transform(X_test)

# Normalisation des fonctionnalités
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_pca)
X_test_scaled = scaler.transform(X_test_pca)

# Initialisation du modèle de régression logistique
log_reg = LogisticRegression(max_iter=1000, solver='saga', penalty='l2', n_jobs=-1)  # Utiliser tous les cœurs disponibles

# Entraînement du modèle
log_reg.fit(X_train_scaled, y_train.values.ravel())

# Prédiction sur les données de test
y_pred = log_reg.predict(X_test_scaled)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle de régression logistique :", accuracy)

# Chargement des données de test
test_data = pd.read_csv("/home/etud/Bureau/projet_python/kanji_test_data.csv", header=None)

# Réduction de la dimensionnalité avec PCA pour les données de test
test_data_pca = pca.transform(test_data)

# Normalisation des fonctionnalités pour les données de test
test_data_scaled = scaler.transform(test_data_pca)

# Prédire les classes des kanjis dans l'ensemble de test
test_predictions = log_reg.predict(test_data_scaled).round().astype(int)

# Enregistrement des prédictions dans un fichier CSV
np.savetxt("kanji_test_predictionsreg.csv", test_predictions, delimiter=",", fmt='%d')
