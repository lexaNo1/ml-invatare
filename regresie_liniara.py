import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Date
suprafete = np.array([50, 60, 70, 80, 90, 100])
preturi = np.array([50000, 60000, 70000, 80000, 85000, 100000])
suprafete_2d = suprafete.reshape(-1, 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(suprafete_2d, preturi, test_size=0.2, random_state=42)

# Antrenare
model = LinearRegression()
model.fit(X_train, y_train)

# Predictie
pret_prezis = model.predict([[120]])
print(f"Pret prezis pentru 120 mp: {pret_prezis[0]}")
print(f"Panta: {model.coef_[0]}")
print(f"Interceptul: {model.intercept_}")

# Grafic
plt.scatter(suprafete, preturi, color='blue', label='Date reale')
plt.plot(suprafete, model.predict(suprafete_2d), color='red', label='Linia modelului')
plt.xlabel('Suprafata (mp)')
plt.ylabel('Pret (euro)')
plt.legend()
plt.savefig('grafic.png')
print("Grafic salvat!")

# Metrici
predictii = model.predict(X_test)
mae = mean_absolute_error(y_test, predictii)
rmse = mean_squared_error(y_test, predictii) ** 0.5

print(f"MAE: {mae} euro")
print(f"RMSE: {rmse} euro")
