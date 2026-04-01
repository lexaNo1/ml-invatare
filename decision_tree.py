import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.array([
    [1, 1, 100], [2, 1, 200], [1, 2, 150], [2, 2, 180], [1, 0, 90],
    [8, 5, 50],  [7, 4, 60],  [9, 6, 40],  [8, 4, 55],  [9, 5, 45]
])

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictii = model.predict(X_test)
acuratete = accuracy_score(y_test, predictii)
print(f"Acuratete: {acuratete * 100}%")

model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(X_train, y_train)

predictii_rf = model_rf.predict(X_test)
acuratete_rf = accuracy_score(y_test, predictii_rf)
print(f"Acuratete Random Forest: {acuratete_rf * 100}%")


importante = model_rf.feature_importances_
caracteristici = ['cuvinte_suspicious', 'numar_linkuri', 'lungime_email']

for i, caracteristica in enumerate(caracteristici):
    print(f"{caracteristica}: {importante[i]:.3f}")