import csv
import numpy as np
import collections
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import metrics

import pandas as pd

f_train = open("cancer_train.csv")
f_test = open("cancer_test.csv")

reader_train = csv.reader(f_train)
reader_test = csv.reader(f_test)

train_data = []
for line in reader_train:
    train_data.append(line)
train_data = np.array(train_data)

test_data = []
for line in reader_test:
    test_data.append(line)
test_data = np.array(test_data)

"""
cancer data explanation:

[0] Data Label ( 0: benign, 1: malignant )
[1] Clump Thickness
[2] Uniformity of Cell Size
[3] Uniformity of Cell Shape
[4] Marginal Adhesion
[5] Single Epithelial Cell Size
[6] Bare Nuclei
[7] Bland Chromatin
[8] Normal Nucleoli
[9] Mitoses

all the features ranges from 1 to 10
"""

# K Nearest Neighbors ( K = 5)

knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(train_data[:, 1:], train_data[:, 0])

knn_train_mat = train_data[:, 0] == knn.predict(train_data[:, 1:])
knn_test_mat = test_data[:, 0] == knn.predict(test_data[:, 1:])

# Decision Tree

dt = tree.DecisionTreeClassifier()
dt.fit(train_data[:, 1:], train_data[:, 0])

dt_train_mat = train_data[:, 0] == dt.predict(train_data[:, 1:])
dt_test_mat = test_data[:, 0] == dt.predict(test_data[:, 1:])

# SVM

svmc = svm.SVC()
svmc.fit(train_data[:, 1:], train_data[:, 0])

svmc_train_mat = train_data[:, 0] == svmc.predict(train_data[:, 1:])
svmc_test_mat = test_data[:, 0] == svmc.predict(test_data[:, 1:])

print("###### Train Dataset Accuracy ######\n")
print("KNN classifier: %.4lf" % (collections.Counter(knn_train_mat)[True] / len(train_data)))
# print("%.4lf" % metrics.accuracy_score(train_data[:, 0], knn.predict(train_data[:, 1:])))
print("Decision Tree classifier: %.4lf" % (collections.Counter(dt_train_mat)[True] / len(train_data)))
# print("%.4lf" % metrics.accuracy_score(train_data[:, 0], dt.predict(train_data[:, 1:])))
print("SVM classifier: %.4lf" % (collections.Counter(svmc_train_mat)[True] / len(train_data)))
# print("%.4lf" % metrics.accuracy_score(train_data[:, 0], svmc.predict(train_data[:, 1:])))

print("\n###### Test Dataset Accuracy ######\n")
print("KNN classifier: %.4lf" % (collections.Counter(knn_test_mat)[True] / len(test_data)))
# print("%.4lf" % metrics.accuracy_score(test_data[:, 0], knn.predict(test_data[:, 1:])))
print("Decision Tree classifier: %.4lf" % (collections.Counter(dt_test_mat)[True] / len(test_data)))
# print("%.4lf" % metrics.accuracy_score(test_data[:, 0], dt.predict(test_data[:, 1:])))
print("SVM classifier: %.4lf" % (collections.Counter(svmc_test_mat)[True] / len(test_data)))
# print("%.4lf" % metrics.accuracy_score(test_data[:, 0], svmc.predict(test_data[:, 1:])))

# SVM classifier가 가장 Accuracy가 높아 최적의 분류기로 판단됨.

modified_svm = svm.SVC(C=2.0)
modified_svm.fit(train_data[:, 1:], train_data[:, 0])

modified_svm_test_mat = test_data[:, 0] == modified_svm.predict(test_data[:, 1:])

print("\n###### Tuned Test Dataset Accuracy ######\n")
print("SVM with C=2.0 : %.4lf" % (collections.Counter(modified_svm_test_mat)[True] / len(test_data)))
# print("%.4lf" % metrics.accuracy_score(test_data[:, 0], modified_svm.predict(test_data[:, 1:])))

a = test_data[:, 0].astype(float)

p = modified_svm.predict(test_data[:, 1:]).astype(float)

con_mat = {

    'Predicted 0': [collections.Counter((a+1)*(p+1))[1], collections.Counter(p)[0]-collections.Counter((a+1)*(p+1))[1]],
    'Predicted 1': [collections.Counter(p)[1] - collections.Counter(a*p)[1], collections.Counter(a*p)[1]]
}

# test_data[:, 0]
# modified_svm.predict(test_data[:, 1:] )

con_index = ['Actual 0', 'Actual 1']

con_mat = pd.DataFrame(con_mat, index=con_index)

print("\nconfusion matrix:")
print(con_mat)
print("\nPrecision: %.4lf" % metrics.precision_score(a, p))
print("Recall: %.4lf" % metrics.recall_score(a, p))
print("F1_Score: %.4lf" % metrics.f1_score(a, p))

# print(metrics.confusion_matrix(a, p))
f_train.close()
f_test.close()
