import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model as ln

fisier = pd.read_csv("high_diamond_ranked_Liniar.csv")

kills = fisier.iloc[:, [0]].values
minions = fisier.iloc[:, [1]].values
jungle_minions = fisier.iloc[:, [2]].values
gold = fisier.iloc[:, [3]].values
kills_set = fisier.iloc[:, [4]].values
gold_set = fisier.iloc[:, [5]].values
state = fisier.iloc[:, [0, 1, 2]].values

# Tgold_set = np.array(gold_set).reshape(-1, 1)
# Tkills_set = np.array(kills_set).reshape(-1, 1)
# Tgold = np.array(gold).reshape(-1, 1)
# Tkills = np.array(kills).reshape(-1, 1)

regression = ln.LinearRegression()
regression.fit(kills, gold)
new_kills = [[23]]
new_gold = regression.predict(new_kills)
r_square = regression.score(kills, gold)
print("R_squared = " + str(r_square))
print("Gold generat pentru " + str(new_kills) + " " + "kills = " + str(new_gold))

gold_prezis = regression.predict(kills)
plt.scatter(kills_set, gold_set, s=40, c="darkviolet")
plt.scatter(new_kills, new_gold, c="black", s=50, label="rgr predict")
y = regression.coef_ * kills_set + regression.intercept_


plt.plot(kills_set, y, linestyle='dashed', c="black", linewidth=4)
plt.plot(kills, gold_prezis, c="cyan")
plt.title("Regresie liniara - Gold generat in functie de kill-uri")
plt.ylabel("Gold generat")
plt.xlabel("Kill-uri in primele 10 minute")
plt.legend()
plt.show()

# regression.fit(state, gold)
# new_state = [[5, 250, 10]]
# new_gold = regression.predict(new_state)
# r_squared= regression.score(state, gold)
# print(r_squared)
# print(new_gold)










