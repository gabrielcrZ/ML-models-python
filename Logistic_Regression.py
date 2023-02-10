import matplotlib.pyplot as plt
import sklearn.linear_model as ln
import pandas as pd
import numpy as np

fisier = pd.read_csv("high_diamond_ranked_Logistic.csv")
kills = fisier.iloc[:, [0]].values
# deaths = fisier.iloc[:, [1]].values
# kills_deaths = fisier.iloc[:, [0, 1]].values
result = fisier.iloc[:, [2]].values
# result_result = fisier.iloc[:, [2, 2]].values



plt.scatter(kills, result, color="darkviolet")
plt.xlabel("Kill-uri in primele 10 minute")
plt.ylabel("Rezultat meci")
plt.title("Regresie logistica - Raport Kills/Deaths -> Rezultat meci ")
#plt.show()
regression = ln.LogisticRegression()
regression.fit(kills, result.ravel())

#Intre 6-7 kills in primele 10 minute pentru a genera victorie
x_nou = [[7]]
y_nou = regression.predict(x_nou)
print("Probabilitati" + " " + str(regression.predict_proba(x_nou)))
print("Rezultat (0 = Infrangere, 1 = Victorie) = " + str(regression.predict(x_nou)))

# regression.fit(kills_deaths, result_result)
# kills_deaths_nou = [[5,1]]
# result_nou = regression.predict(kills_deaths_nou)
# print(regression.predict_proba(kills_deaths_nou))
# print(regression.predict(kills_deaths_nou))

