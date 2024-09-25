import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

train_data = pd.read_csv("/home/etud/Bureau/projet_python/kanji_train_data.csv", sep=",")
train_target = pd.read_csv("/home/etud/Bureau/projet_python/kanji_train_target.csv", sep=",")
# Réduire la dimensionnalité avec PCA en 2 dim 
pca = PCA(n_components=2) 
train_data_2d = pca.fit_transform(train_data)

# Tracer les données en 2D avec des couleurs distinctes pour chaque classe
plt.figure(figsize=(10, 8))
for i in range(20):
    plt.scatter(train_data_2d[train_target.iloc[:, 0] == i, 0], train_data_2d[train_target.iloc[:, 0] == i, 1], label=f"Class {i}")

plt.title("Visualisation des Kanjis en 2D")
plt.xlabel("Composante Principale 1")
plt.ylabel("Composante Principale 2")
plt.legend()
plt.show()

# Fonction pour afficher la représentation graphique d'un kanji à partir de son vecteur
def plot_kanji(vector):
    kanji_matrix = np.array(vector).reshape(64, 64)
    plt.imshow(kanji_matrix, cmap='gray')
    plt.axis('off')
    plt.show()
    

# Exemple d'utilisation de la fonction pour afficher un kanji
plot_kanji(train_data.iloc[0])


