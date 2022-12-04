import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
import sklearn.preprocessing as skl_pre
import sklearn.model_selection as skl_ms
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

np.random.seed(1)
# Create array of classifiers to test bagging
classifierName = [tree.DecisionTreeClassifier(max_depth=7),
                  tree.DecisionTreeClassifier(max_leaf_nodes=30),
                  RandomForestClassifier(n_estimators=50)]

Train = pd.read_csv("/home/toidface/Documents/ML_proj/train.csv")
# Colab: Train = pd.read_csv("/content/train.csv")
print(Train.shape)

Y_train = Train['Lead']
X_train = Train.drop(columns=['Lead'])
# 1. Use scaler on the data set (just parameters)
scaler = skl_pre.StandardScaler().fit(X_train)
# 2. Scale the training data
X_train_norm = scaler.transform(X_train)

# n-fold runs
n_fold = 10
cv = skl_ms.KFold(n_splits=n_fold, random_state=2, shuffle=True)
K = np.arange(2, 3)
misclassification = np.zeros((len(classifierName), len(K)))
print(cv)
# run all models and subplot
for m in np.arange(len(classifierName)):
    print(m)
    # then run for different depths
    for train_index, val_index in cv.split(X_train):
        x_train, x_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train, y_val = Y_train.iloc[train_index], Y_train.iloc[val_index]

        # Normalise
        X_train_norm = scaler.transform(x_train)
        X_val_norm = scaler.transform(x_val)
        for j, k in enumerate(K):
            model = BaggingClassifier(
                base_estimator=classifierName[m],
                n_estimators=k)
            model.fit(X_train_norm, y_train)
            prediction = model.predict(X_val_norm)
            misclassification[m, j] += np.mean(prediction != y_val)
    print(m)

# Plot cross-val-error per method, assuming three models, change if necessary
misclassification /= n_fold
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(K, misclassification[0, :])
ax1.set_title('DT max_d = 7')
ax1.set(xlabel='', ylabel='Validation error')
ax2.plot(K, misclassification[1, :])
ax2.set_title('DT max_leaf = 30')
ax2.set(xlabel='n_estimators', ylabel='')
ax3.plot(K, misclassification[2, :])
ax3.set_title('RFC n_est = 50')
ax3.set(xlabel='', ylabel='')
fig.set_figwidth(20)
plt.show()
