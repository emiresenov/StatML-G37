import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import graphviz

from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

np.random.seed(1)

dir = '/home/toidface/Documents/ML_proj/Tree based/DecisionTree_leaf/'
Train = pd.read_csv("/home/toidface/Documents/ML_proj/train.csv")
# print(Train.shape)

trainIndex = np.random.choice(Train.shape[0], size=250, replace=False)
train = Train.iloc[trainIndex]
test = Train.iloc[~Train.index.isin(trainIndex)]
Y_train = train['Lead']
X_train = train.drop(columns=['Lead'])
# print(X_train)

# First clear the output-file,
output_clear = open(dir + 'output.txt', 'w')
output_clear.write(' ')
# then run for different depths
for i in range(4, 22, 4):
    model = tree.DecisionTreeClassifier(max_leaf_nodes=i)
    model.fit(X_train, Y_train)
    dot_data = tree.export_graphviz(model,
                                    out_file=dir + "None_" + str(i),
                                    feature_names=X_train.columns,
                                    class_names=model.classes_,
                                    filled=True,
                                    rounded=True,
                                    leaves_parallel=True,
                                    proportion=True)

    graph = graphviz.Source(dot_data)

# Test the model
    Y_test = test['Lead']
    X_test = test.drop(columns=['Lead'])
    Y_predict = model.predict(X_test)

# Save Acc. rate & crosstab to File
    output = open(dir + 'output.txt', 'a')
    output.write('Res for tree-leaves:' + str(i) + "\n")
    output.write('Accuracy rate of: '
                 + str((int)(100 * np.mean(Y_predict == Y_test))) + "\n")
    output.write(
        str(100*pd.crosstab(Y_predict, Y_test, normalize='index')) + "\n\n\n")
