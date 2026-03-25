import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Date: suprafata (mp) si pret (euro)
suprafete = np.array([50, 60, 70, 80, 90, 100])
preturi = np.array([50000, 60000, 70000, 80000, 85000, 100000])
suprafete_2d = suprafete.reshape(-1, 1)
model = LinearRegression()
model.fit(suprafete_2d, preturi)

pret_prezis = model.predict([[120]])
print(f"Pret prezis pentru 120 mp: {pret_prezis[0]}")
print(f"Panta: {model.coef_[0]}")
print(f"Interceptul:{model.intercept_}")


plt.scatter(suprafete, preturi, color='blue', label='Date reale')
plt.plot(suprafete, model.predict(suprafete_2d), color='red', label='Linia modelului')
plt.xlabel('Suprafata (mp)')
plt.ylabel('Pret (euro)')
plt.legend()
plt.savefig('grafic.png')
print("Grafic salvat!")

from sklearn.metrics import mean_absolute_error, mean_squared_error

predictii = model.predict(suprafete_2d)
mae = mean_absolute_error(preturi, predictii)
rmse = mean_squared_error(preturi, predictii) ** 0.5

print(f"MAE: {mae} euro")
print(f"RMSE: {rmse} euro")
