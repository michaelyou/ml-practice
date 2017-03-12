import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def create_dataset():

    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables


def create_dataset2():
    df = pd.read_table('datingTestSet.txt', sep='\s+', names=['flymiles', 'gametime', 'icecream', 'label'])
    label = df.pop('label')
    return df, label

# def classify0(target, dataset, labels, k):
#     dataset_size = group.shape[0]
#     diffmat = np.tile(target, (dataset_size, 1)) - dataset
#     sq_diffmat = diffmat ** 2
#     sq_distance = sq_diffmat.sum(axis=1)
#     distance = sq_distance ** 0.5
# 
#     df = pd.concat([Series(distance), Series(labels)], axis=1)
#     df.columns = ['distance', 'label']
#     df = df.sort_values(by='distance')
#     return df['label'][:k].value_counts().idxmax()


def classify0(target, dataset, labels, k):
    diffmat = DataFrame(dataset) - target
    sq_diffmat = diffmat ** 2
    sq_distance = sq_diffmat.sum(axis=1)
    distance = sq_distance ** 0.5

    df = pd.concat([distance, Series(labels)], axis=1)
    df.columns = ['distance', 'label']
    df = df.sort_values(by='distance')
    return df['label'][:k].value_counts().idxmax()


def norm(df):
    max_value = df.max()
    min_value = df.min()
    gap = max_value - min_value
    norm_df = (df - min_value) / gap
    return norm_df, min_value, gap


def data_test():
    df, label = create_dataset2()
    label = label.replace('didntLike', 0).replace('smallDoses', 1).replace('largeDoses', 2)
    norm_df, min_value, gap = norm(df)

    num_test = int(0.1 * df.shape[0])
    wrong_count = 0
    for i in range(num_test):
        test_result = classify0(norm_df.iloc[i], norm_df.iloc[i+num_test:], label[i+num_test:], 3) 
        if test_result != label[i]:
            print(test_result, label[i])
            wrong_count += 1

    # 0.05
    return float(wrong_count) / num_test

print(data_test())
# group, labels = create_dataset()
# print(classify0([0, 0], group, labels, 3))
