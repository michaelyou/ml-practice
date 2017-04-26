import numpy as np


def load_dataset(file_name):
    data_mat = []
    with open(file_name) as f:
        for line in f:
            line = line.strip().split('\t')
            line = list(map(float, line))
            data_mat.append(line)
    return np.mat(data_mat)


def distance(vec_a, vec_b):
    return np.sqrt(np.sum(np.power(vec_a - vec_b, 2)))


def rand_center(data_set, k):
    # 列数
    n = np.shape(data_set)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_j = min(data_set[:, j])
        range_j = float(max(data_set[:, j]) - min_j)
        # np.random.rand生成一个k行1列的np.array
        centroids[:, j] = min_j + range_j * np.random.rand(k, 1)
    return centroids


data_set = load_dataset('testSet.txt')


def kmeans(data_set, k):
    m = np.shape(data_set)[0]
    cluster_assignment = np.mat(np.zeros((m, 2)))
    centroids = rand_center(data_set, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):

