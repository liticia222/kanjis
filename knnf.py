import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Charger les données d'entraînement

train_data = pd.read_csv("/home/etud/Bureau/projet_python/kanji_train_data.csv", sep=",", header=None)
train_target = pd.read_csv("/home/etud/Bureau/projet_python/kanji_train_target.csv", sep=",", header=None)




# Diviser les données d'entraînement en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(train_data, train_target, test_size=0.2, random_state=42)

# Tester différentes valeurs de k pour trouver le meilleur nombre de voisins
k_values = [1, 3, 5, 7, 9]
accuracies = []

for k in k_values:
    # Initialiser le classifieur KNN
    knn = KNeighborsClassifier(n_neighbors=k)
   
    # Entraîner le modèle KNN sur l'ensemble d'entraînement
    knn.fit(X_train, y_train.values.ravel())
   
    # Prédire les étiquettes sur l'ensemble de validation
    y_pred = knn.predict(X_val)
   
    # Calculer l'exactitude du modèle
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)
   
# Tracer la courbe de précision en fonction du nombre de voisins
plt.plot(k_values, accuracies, marker='o')
plt.title('Précision du modèle en fonction du nombre de voisins')
plt.xlabel('Nombre de voisins (k)')
plt.ylabel('Précision')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Trouver le meilleur k
best_k = k_values[np.argmax(accuracies)]
print("Meilleur nombre de voisins (k) :", best_k)

# Entraîner le modèle KNN avec le meilleur k sur l'ensemble d'entraînement complet
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(train_data, train_target.values.ravel())




#---------------------------------------------------------------------------------------------------------
# Charger les données de test
test_data = pd.read_csv("/home/etud/Bureau/projet_python/kanji_test_data.csv", header=None)

print(test_data.shape)
accuracy = accuracy_score(y_val, y_pred)
print("Exactitude du modèle sur l'ensemble de validation :", accuracy)

# Prédire les classes des kanjis dans l'ensemble de test
test_predictions = best_knn.predict(test_data)

np.savetxt("kanji_test_predictions.csv",test_predictions )
#lastuce est de reutiliser tout les donner a la fin
