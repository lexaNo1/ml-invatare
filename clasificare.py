import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# [cuvinte_suspicious, numar_linkuri]
X = np.array([
    [1, 1], [2, 1], [1, 2],   # nu spam
    [8, 5], [7, 4], [9, 6]    # spam
])

# 0 = nu spam, 1 = spam
y = np.array([0, 0, 0, 1, 1, 1])

#split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model
model = LogisticRegression()
model.fit(X_train, Y_train)

predictii = model.predict(X_test)
acuratete = accuracy_score(Y_test, predictii)
print(f"Acuratete: {acuratete * 100}%")

email_nou = np.array([[6, 3]])
predictie = model.predict(email_nou)
print(f"Email: {'SPAM' if predictie[0] == 1 else 'NU SPAM'}")
