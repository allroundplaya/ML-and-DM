import csv
import knn
import numpy as np
import collections


def cal_accuracy(a_arr, p_arr, c_arr):
    """
    :param a_arr:
        Data Type: 1d numpy array
        Array of actual values
    :param p_arr:
        Data Type: 1d numpy array
        Array of predicted values
    :param c_arr:
        Data Type: 1d numpy array
        Array of classes
    :return:
        Data Type: float
    """

    acc_arr = np.array([])
    for value in c_arr:
        acc_arr = np.append(acc_arr, collections.Counter( (a_arr == value) * (p_arr == value) )[True] / len(a_arr) )

    return np.sum(acc_arr)


def cal_precision(a_arr, p_arr, c_arr):
    """
    :param a_arr:
        Data Type: 1d numpy array
        Array of actual values
    :param p_arr:
        Data Type: 1d numpy array
        Array of predicted values
    :param c_arr:
        Data Type: 1d numpy array
        Array of classes
    :return:
        Data Type: float
    """

    pre_arr = np.array([])
    for value in c_arr:
        pre_arr = np.append( pre_arr, collections.Counter( (a_arr == value) * (p_arr == value) )[True] / collections.Counter(p_arr == value)[True])

    # print(pre_arr)
    return np.average(pre_arr)


def cal_recall(a_arr, p_arr, c_arr):
    """
    :param a_arr:
        Data Type: 1d numpy array
        Array of actual values
    :param p_arr:
        Data Type: 1d numpy array
        Array of predicted values
    :param c_arr:
        Data Type: 1d numpy array
        Array of classes
    :return:
        Data Type: float
    """
    re_arr = np.array([])
    for value in c_arr:
        re_arr = np.append(re_arr, collections.Counter((a_arr == value) * (p_arr == value))[True] / collections.Counter(a_arr == value)[True])

    return np.average(re_arr)


def cal_f1(a_arr, p_arr, c_arr):
    """
    :param a_arr:
        Data Type: 1d numpy array
        Array of actual values
    :param p_arr:
        Data Type: 1d numpy array
        Array of predicted values
    :param c_arr:
        Data Type: 1d numpy array
        Array of classes
    :return:
        Data Type: float
    """
    f1_arr = np.array([])
    for value in c_arr:
        x = collections.Counter((a_arr == value) * (p_arr == value) )[True] / collections.Counter(p_arr == value)[True]
        y = collections.Counter((a_arr == value) * (p_arr == value))[True] / collections.Counter(a_arr == value)[True]
        f1_arr = np.append(f1_arr, 2 / ((1/x) + (1/y)))


    return np.average(f1_arr)


def cal_result(k, dist_type, train_data):
    """
    :param k:
        Data Type: int

    :param dist_type: string
        Data Type:
    :param train_data:
        Data Type:
    :return:
        Data Type:
    """
    f_txt = open("result_02.txt", "a")
    clf = knn.KNN(k, dist_type)

    print("Saving...")

    f_txt.write("============================================\n\n")
    f_txt.write("# of K: %d\n" % k)

    if dist_type == 'e' or dist_type == 'E':
        f_txt.write("Distance Type: Euclidean\n")

    elif dist_type == 'm' or dist_type == 'M':
        f_txt.write("Distance Type: Manhattan\n")

    elif dist_type == 'l' or dist_type == 'L':
        f_txt.write("Distance Type: L∞\n")

    clf.train(train_data)

    a_arr_train = train_data[:, 0]
    a_arr_test = test_data[:, 0]

    p_arr_train = np.array([])
    p_arr_test = np.array([])

    c_arr = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

    for row in train_data:
        p_arr_train = np.append(p_arr_train, clf.predict(row))

    for row in test_data:
        p_arr_test = np.append(p_arr_test, clf.predict(row))

    f_txt.write("\n1. Metrics of Train Data\n")
    f_txt.write("Accuracy: %.4lf\n" % cal_accuracy(a_arr_train, p_arr_train, c_arr))
    f_txt.write("Precision: %.4lf\n" % cal_precision(a_arr_train, p_arr_train, c_arr))
    f_txt.write("Recall: %.4lf\n" % cal_recall(a_arr_train, p_arr_train, c_arr))
    f_txt.write("F-1 Score: %.4lf\n" % cal_f1(a_arr_train, p_arr_train, c_arr))

    f_txt.write("\n2. Metrics of Test Data\n")
    f_txt.write("Accuracy: %.4lf\n" % cal_accuracy(a_arr_test, p_arr_test, c_arr))
    f_txt.write("Precision: %.4lf\n" % cal_precision(a_arr_test, p_arr_test, c_arr))
    f_txt.write("Recall: %.4lf\n" % cal_recall(a_arr_test, p_arr_test, c_arr))
    f_txt.write("F-1 Score: %.4lf\n" % cal_f1(a_arr_test, p_arr_test, c_arr))
    f_txt.write("\n============================================\n\n")

    print("Done!")
    f_txt.close()
    del clf
    return

def cal_5fold(k, dist_type, train_data):
    f_5 = open("result_02_5fold.txt", "a")
    print("Saving...")
    clf_5 = knn.KNN(k, dist_type)

    f_5.write("============================================\n\n")
    f_5.write("# of K: %d\n" % k)

    if dist_type == 'e' or dist_type == 'E':
        f_5.write("Distance Type: Euclidean\n")

    elif dist_type == 'm' or dist_type == 'M':
        f_5.write("Distance Type: Manhattan\n")

    elif dist_type == 'l' or dist_type == 'L':
        f_5.write("Distance Type: L∞\n")

    c_arr = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

    acc_test_arr = np.array([])
    pre_test_arr = np.array([])
    re_test_arr = np.array([])
    f1_test_arr = np.array([])

    t_data = train_data
    np.random.shuffle(t_data)
    for i in range(5):
        train5_data = np.delete(t_data, np.s_[160*i: 160*(i+1)], axis=0)
        test5_data = t_data[160*i: 160*(i+1)]

        clf_5.train(train5_data)

        a_arr_test5 = test5_data[:, 0]
        p_arr_test5 = np.array([])


        for row in test5_data:
            p_arr_test5 = np.append(p_arr_test5, clf_5.predict(row))

        acc_test_arr = np.append(acc_test_arr, cal_accuracy(a_arr_test5, p_arr_test5, c_arr))
        pre_test_arr = np.append(pre_test_arr, cal_precision(a_arr_test5, p_arr_test5, c_arr))
        re_test_arr = np.append(re_test_arr, cal_recall(a_arr_test5, p_arr_test5, c_arr))
        f1_test_arr = np.append(f1_test_arr, cal_f1(a_arr_test5, p_arr_test5, c_arr))

    f_5.write("\n5-fold Cross Validation Metrics\n")
    f_5.write("Accuracy: %.4lf\n" % np.average(acc_test_arr))
    f_5.write("Precision: %.4lf\n" % np.average(pre_test_arr))
    f_5.write("Recall: %.4lf\n" % np.average(re_test_arr))
    f_5.write("F-1 Score: %.4lf\n" % np.average(f1_test_arr))
    f_5.write("\n============================================\n\n")
    print("Done!")
    f_5.close()
    del clf_5
    return


# CSV -> numpy Array
f_train = open("digits_train.csv")
f_test = open("digits_test.csv")

reader_train = csv.reader(f_train)
reader_test = csv.reader(f_test)

train_data = []
for line in reader_train:
    train_data.append(line)
train_data = np.array(train_data)
train_data = train_data.astype(float)

test_data = []
for line in reader_test:
    test_data.append(line)
test_data = np.array(test_data)
test_data = test_data.astype(float)


dtype_list = ['e', 'm', 'l']
# dtype_list = ['l']
k_list = [3, 4, 5]


print("===== KNN Classifier Metrics =====")
# for dtype in dtype_list:
#     for k_num in k_list:
#         print("(k, type) = (%d, %c)" % (k_num, dtype))
#         cal_result(k_num, dtype, train_data)


print("===== 5-fold Cross Validation =====")
for dtype in dtype_list:
    for k_num in k_list:
        cal_5fold(k_num, dtype, train_data)
f_train.close()
f_test.close()
