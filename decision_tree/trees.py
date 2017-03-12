from pandas import Series, DataFrame
from math import log
import pandas as pd


def calculate_shannon_entropy(labels):
    labels = Series(labels)
    label_occurance = labels.value_counts()
    label_size = len(labels)

    entropys = label_occurance.apply(lambda x: (x/label_size) * (-log(x/label_size,2)))
    return entropys.sum()


def split_dataframe(df, label, value):
    return df[label == value]


def choose_best_feature_to_split(df):

    def get_entropy_each_feature(feature):
        entropy = 0.0
        feature_values = feature.unique()
        for value in feature_values:
            sub_df = split_dataframe(df, feature, value)
            prob = len(sub_df) / float(len(feature))
            entropy += prob * calculate_shannon_entropy(sub_df[sub_df.columns[-1]])
        return entropy

    entropys = df.apply(get_entropy_each_feature)
    # 最好的划分熵最小，划分应该是一个熵减的过程
    return entropys[:-1].idxmin()


def majority_count(labels):
    return labels.value_counts().idxmax()


def create_tree(df):
    class_list = df[df.columns[-1]]
    if len(class_list.unique()) == 1:
        return class_list.iloc[0]
    if len(df.columns) == 1:
        # now df has just the label column
        return majority_count(df)
    best_feature = choose_best_feature_to_split(df)
    tree = {best_feature: {}}
    feature_values = df[best_feature].unique()
    for value in feature_values:
        tree[best_feature][value] = create_tree(split_dataframe(df, df[best_feature], value))
    return tree


def classify(tree, test_df):
    first_key = list(tree.keys())[0]
    second_tree = tree[first_key]
    for key in second_tree.keys():
        if test_df[first_key][0] == key:
            if type(second_tree[key]) == dict:
                return classify(second_tree[key], test_df)
            else:
                return second_tree[key]
    return None


# test
# df = DataFrame([[1, 2, 3, 1, 2], [1, 3, 2, 1, 1], [1, 2, 2, 2, 2], [2, 2, 2, 3, 1]], columns=['A', 'B', 'C', 'D', 'E'])
# labels = features = ['surface', 'flippers', 'fish']
# df = DataFrame([[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']], columns=labels)
# print(calculate_shannon_entropy([1, 1, 1, 0, 0, 1, 1, 2, 3, 4, 2, 5]))
# classify_tree = create_tree(df)
# print(classify(classify_tree, DataFrame([[1, 1]], columns=['surface', 'flippers'])))

def glass_classify():
    glass_df = pd.read_csv('lenses.txt', sep='\t', names=['age', 'prescript', 'astigmatic', 'tearRate', 'glass_type'])
    glass_tree = create_tree(glass_df)
    print(glass_tree)

print(glass_classify())
