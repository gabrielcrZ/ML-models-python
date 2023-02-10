from sklearn import cluster as cl
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics as m
import numpy as np


fisier = pd.read_csv("world_championship_Cluster.csv")
fisier.head()
x = fisier.iloc[:, [2, 1]].values

def elbow(x):
    k = range(1, 10)
    inertii = []
    for i in k:
        km = cl.KMeans(n_clusters=i, n_init='auto')
        km.fit(x)
        inertii.append(km.inertia_)

    plt.plot(k, inertii, color="darkviolet")
    plt.title('Calcul elbow')
    plt.show()
elbow(x)


km = cl.KMeans(n_clusters=2, n_init='auto')
y_predicted = km.fit_predict(fisier[['Creep Score', 'Gold Earned']].values)

fisier['cluster'] = y_predicted
#print(y_predicted)
fisier.head()

x_nou = [[300, 16500]]
x_nou_x = 300
x_nou_y = 16500
print(km.predict(x_nou))
print("Silhouette score = " + str(m.silhouette_score(x, km.labels_)))

fisier1 = fisier[fisier.cluster == 0]
fisier2 = fisier[fisier.cluster == 1]
# fisier3 = fisier[fisier.cluster == 2]
# fisier4 = fisier[fisier.cluster == 3]
plt.scatter(fisier1['Creep Score'], fisier1['Gold Earned'], color='darkviolet')
plt.scatter(fisier2['Creep Score'], fisier2['Gold Earned'], color='cyan')
# plt.scatter(fisier3['Creep Score'], fisier3['Gold Earned'],color='yellow')
# plt.scatter(fisier4['Creep Score'], fisier4['Gold Earned'],color='red')
plt.scatter(x_nou_x, x_nou_y, color='black', label='km predict')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='black', marker="*", label='centroid')

plt.title('Clustering - Gold generat de fiecare player in functie de Creep Score')
plt.xlabel('Creep Score')
plt.ylabel('Gold generat')
plt.legend()
plt.show()
