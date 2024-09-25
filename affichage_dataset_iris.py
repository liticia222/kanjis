import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

iris=datasets.load_iris()

print(iris)

X=iris.data
Y=iris.target
Y=Y.reshape(Y.shape[0],1)
list_col=iris.feature_names+ ["target"]
df=pd.DataFrame(data=np.hstack((X,Y)),columns=list_col)

print(df)


list_color= ["blue","green","red"]




for i in range(3):
    df_subclass=df.loc[df["target"] ==i]
    plt.scatter(df_subclass["sepal length (cm)"],df_subclass["sepal width (cm)"],color=list_color[i], label=iris.target_names[i])

plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Sepal Width Vs Sepal Length")
plt.legend()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in range(3):
    df_subclass=df.loc[df["target"] ==i]
    ax.scatter(df_subclass["sepal length (cm)"],df_subclass["sepal width (cm)"],df_subclass["petal length (cm)"],color=list_color[i],label=iris.target_names[i])

ax.set_xlabel("Sepal length")
ax.set_ylabel("Sepal width")
ax.set_zlabel("Petal length")
plt.legend()
plt.title("Iris 3D plot")

plt.show()





