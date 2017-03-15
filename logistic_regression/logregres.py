import numpy as np


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

data_mat, label_mat = load_dataset()
print(grad_ascent(data_mat, label_mat))
