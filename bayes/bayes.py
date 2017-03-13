import numpy as np


training_docs = ['my dog has flea problems help please', 'maybe not take him to dog park stupid', 'my dalmation is so cute i love him',
                 'stop posting stupid worthless garbage', 'mr licks ate my steak how to stop him', 'quit buying worthless dog food stupid']


def load_dataset():
    splited_docs = []
    for index, doc in enumerate(training_docs):
        splited_docs.append(doc.split())

    class_vec = [0, 1, 0, 1, 0, 1]

    return splited_docs, class_vec


def create_vocab_list():
    docs = ' '.join(training_docs)
    return list(set(docs.split()))


def words2vec(vocab_list, input_doc):
    # input_vec = input_doc.split()
    vec = [0] * len(vocab_list)
    for index, w in enumerate(vocab_list):
        if w in input_doc:
            vec[index] = 1

    return vec


def train_naive_bayes(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = np.zeros(num_words)
    p1_num = np.zeros(num_words)
    p0_denom = 0.0
    p1_denom = 0.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vect = p1_num / p1_denom
    p0_vect = p0_num / p0_denom

    print(p0_vect, p1_vect, p_abusive)
    return p0_vect, p1_vect, p_abusive


splited_docs, class_vec = load_dataset()
for index, doc in enumerate(splited_docs):
    splited_docs[index] = words2vec(create_vocab_list(), doc)

train_naive_bayes(splited_docs, class_vec)
