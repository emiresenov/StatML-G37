import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import graphviz

from sklearn import tree
import sklearn.preprocessing as skl_pre
import sklearn.model_selection as skl_ms

np.random.seed(1)

dir = '/home/toidface/Documents/ML_proj/Tree based/'
Train = pd.read_csv("/home/toidface/Documents/ML_proj/train.csv")
print(Train.shape)

Y_train = Train['Lead']
X_train = Train.drop(columns=['Lead'])
# 1. Use scaler on the data set (just parameters)
scaler = skl_pre.StandardScaler().fit(X_train)
# 2. Scale the training data
X_train_norm = scaler.transform(X_train)

# First clear the output-file,
output_clear = open(dir + 'output.txt', 'w')
output_clear.write(' ')
# k-fold runs
n_fold = 10
cv = skl_ms.KFold(n_splits=n_fold, random_state=2, shuffle=True)
K = np.arange(1, 75)
misclassification = np.zeros(len(K))
print(cv)
# then run for different depths
for train_index, val_index in cv.split(X_train):
    x_train, x_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train, y_val = Y_train.iloc[train_index], Y_train.iloc[val_index]

    # Normalise
    X_train_norm = scaler.transform(x_train)
    X_val_norm = scaler.transform(x_val)
    for j, k in enumerate(K):
        model = tree.DecisionTreeClassifier(
            max_depth=k)
        model.fit(X_train_norm, y_train)
        prediction = model.predict(X_val_norm)
        misclassification[j] += np.mean(prediction != y_val)
        # if the tree-structure is of interest
        dot_data = tree.export_graphviz(model,
                                        out_file=dir + "None_" + str(k),
                                        feature_names=X_train.columns,
                                        class_names=model.classes_,
                                        filled=True,
                                        rounded=True,
                                        leaves_parallel=True,
                                        proportion=True)

        graph = graphviz.Source(dot_data)

# Save Acc. rate & crosstab to File
misclassification /= n_fold
plt.plot(K, misclassification)
plt.title('Cross validation error for DecisionTree')
plt.xlabel('max_depth')
plt.ylabel('Validation error')
plt.show()
