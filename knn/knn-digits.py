import os

import pandas as pd
from pandas import Series, DataFrame


def classify0(target, dataset, labels, k):
    diffmat = DataFrame(dataset) - target
    sq_diffmat = diffmat ** 2
    sq_distance = sq_diffmat.sum(axis=1)
    distance = sq_distance ** 0.5

    df = pd.concat([distance, Series(labels)], axis=1)
    df.columns = ['distance', 'label']
    df = df.sort_values(by='distance')
    return df['label'][:k].value_counts().idxmax()


def img2vector(filepath):
    digit_l = []
    with open(filepath) as f:
        for l in f:
            digit_l += list(map(int, list(l.strip())))
    return digit_l

# print(img2vector('digits/trainingDigits/1_42.txt'))


def get_dataset(dir_path):
    ds = []
    labels = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            file_path = os.path.join(root, f)
            vec = img2vector(file_path)
            ds.append(vec)
            labels.append(int(f[0]))
    return DataFrame(ds), Series(labels)


def test_digit():
    training_dataset, training_labels = get_dataset('digits/trainingDigits')
    test_dataset, test_labels = get_dataset('digits/testDigits')

    def get_digit_test_result(target):
        return classify0(target, training_dataset, training_labels, 3)

    test_results = test_dataset.apply(get_digit_test_result, axis=1)

    diff = test_results - test_labels
    wrong = diff[diff != 0].count()

    print(wrong, len(test_labels))
    return float(wrong) / len(test_labels)

print(test_digit())


            
