import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def load_dataset():
    data_mat = []
    label_mat = []
    with open('testSet.txt') as f:
        for line in f:
            line = line.strip().split()
            data_mat.append([1.0, float(line[0]), float(line[1])])
            label_mat.append(int(line[2]))
    return data_mat, label_mat


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def grad_ascent(data_mat, class_labels):
    data_matrix = np.mat(data_mat)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(data_matrix)

    alpha = 0.001
    max_cycles = 500
    # 系数
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def stoc_grad_ascent(data_mat, class_labels):
    m, n = data_mat.shape
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid((data_mat[i] * weights).sum())
        error = class_labels[i] - h
        weights = weights + alpha * error * data_mat[i]
    return weights


def stoc_grad_ascent2(data_mat, class_labels):
    m, n = data_mat.shape
    data_mat = data_mat.as_matrix()
    class_labels = class_labels.as_matrix()
    weights = np.ones(n)
    max_cycles = 150
    for k in range(max_cycles):
        data_index = list(range(m))
        for i in range(m):
            # 减小步长
            alpha = 4 / (1.0 + k + i) + 0.0001
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(np.sum(data_mat[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_mat[rand_index]
            del(data_index[rand_index])
    return weights

# data_mat, label_mat = load_dataset()
# print(stoc_grad_ascent2(DataFrame(data_mat), DataFrame(label_mat)))


def colic_test():
    train_mat = pd.read_csv('horseColicTraining.txt', sep='\t', header=None)
    m, n = train_mat.shape
    train_label = train_mat.pop(n-1)

    train_weights = stoc_grad_ascent2(train_mat, train_label)

    test_mat = pd.read_csv('horseColicTest.txt', sep='\t', header=None)
    test_label = test_mat.pop(test_mat.shape[1]-1)

    err_count = 0
    result = (test_mat.as_matrix() * train_weights).sum(axis=1)
    for index, r in enumerate(result):
        res = 1 if sigmoid(r) > 0.5 else 0
        if res != test_label[index]:
            err_count += 1
        
    print(err_count)
    print('err rate is', err_count / len(test_label))

colic_test()
