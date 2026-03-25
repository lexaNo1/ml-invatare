import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Date: suprafata (mp) si pret (euro)
suprafete = np.array([50, 60, 70, 80, 90, 100])
preturi = np.array([50000, 60000, 70000, 80000, 85000, 100000])
suprafete_2d = suprafete.reshape(-1, 1)
model = LinearRegression()
model.fit(suprafete_2d, preturi)
