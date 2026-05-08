import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X = np.array([
    [1, 1, 100], [2, 1, 200], [1, 2, 150], [2, 2, 180], [1, 0, 90],
    [8, 5, 50],  [7, 4, 60],  [9, 6, 40],  [8, 4, 55],  [9, 5, 45]
])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

model = RandomForestClassifier(n_estimators=100)
scoruri = cross_val_score(model, X, y, cv=5)

print(f"Scoruri: {scoruri}")
print(f"Media: {scoruri.mean():.3f}")
print(f"Deviatie standard: {scoruri.std():.3f}")


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100))
])

scoruri_pipe = cross_val_score(pipeline, X, y, cv=5)
print(f"Scoruri pipeline: {scoruri_pipe}")
print(f"Media pipeline: {scoruri_pipe.mean():.3f}")

from sklearn.model_selection import GridSearchCV

parametri = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 2, 5]
}

grid = GridSearchCV(RandomForestClassifier(), parametri, cv=5)
grid.fit(X, y)

print(f"Cei mai buni parametri: {grid.best_params_}")
print(f"Cel mai bun scor: {grid.best_score_:.3f}")