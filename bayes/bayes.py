import os
import numpy as np
from math import log


training_docs = ['my dog has flea problems help please', 'maybe not take him to dog park stupid', 'my dalmation is so cute i love him',
                 'stop posting stupid worthless garbage', 'mr licks ate my steak how to stop him', 'quit buying worthless dog food stupid']


def load_dataset():
    splited_docs = []
    for index, doc in enumerate(training_docs):
        splited_docs.append(doc.split())

    class_vec = [0, 1, 0, 1, 0, 1]

    return splited_docs, class_vec


# 最好移除掉stop words
def create_vocab_list():
    docs = ' '.join(training_docs)
    return list(set(docs.split()))


def create_vocab_list2(dataset):
    vocablist = []
    for doc in dataset:
        vocablist.extend(doc)
    return list(set(vocablist))


# 词集模型
def setofwords2vec(vocab_list, input_doc):
    # input_vec = input_doc.split()
    vec = [0] * len(vocab_list)
    for index, w in enumerate(vocab_list):
        if w in input_doc:
            vec[index] = 1

    return vec


# 词袋模型
def bagofwords2vec(vocab_list, input_doc):
    # input_vec = input_doc.split()
    vec = [0] * len(vocab_list)
    for index, w in enumerate(vocab_list):
        if w in input_doc:
            vec[index] += 1

    return vec


def train_naive_bayes(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    # p0_num = np.zeros(num_words)
    # p1_num = np.zeros(num_words)
    # p0_denom = 0.0
    # p1_denom = 0.0
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            # P(w|spam)
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            # P(w|norm)
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    
    #用log避免概率相乘时的下溢出
    p1_vect = np.log(p1_num/p1_denom)
    p0_vect = np.log(p0_num/p0_denom)

    # print(p0_vect, p1_vect, p_abusive)
    return p0_vect, p1_vect, p_abusive


def classify_naive_bayes(vec2classify, p0_vect, p1_vect, pclass1):
    p1 = sum(vec2classify * p1_vect) + log(pclass1)
    p0 = sum(vec2classify * p0_vect) + log(1.0 - pclass1)
    return 1 if p1 > p0 else 0


def test_naive_bayes():
    splited_docs, class_vec = load_dataset()
    for index, doc in enumerate(splited_docs):
        splited_docs[index] = setofwords2vec(create_vocab_list(), doc)

    p0_vect, p1_vect, p_abusive = train_naive_bayes(splited_docs, class_vec)
    test_entry = ['love', 'my', 'dalmation']
    entry_vec = setofwords2vec(create_vocab_list(), test_entry)
    print(' '.join(test_entry), 'classified as', classify_naive_bayes(entry_vec, p0_vect, p1_vect, p_abusive))
    test_entry = ['stupid', 'garbage']
    entry_vec = setofwords2vec(create_vocab_list(), test_entry)
    print(' '.join(test_entry), 'classified as', classify_naive_bayes(entry_vec, p0_vect, p1_vect, p_abusive))


def email_classify():
    class_vec = []
    splited_docs = []
    for root, dirs, files in os.walk('email/ham'):
        for f in files:
            f_path = os.path.join(root, f)
            with open(f_path, 'r', encoding='ISO-8859-1') as ff:
                content = ff.read()
            content = [x.strip('.,?!()') for x in content.split() if len(x)]
            splited_docs.append(content)
            class_vec.append(0)
    for root, dirs, files in os.walk('email/spam'):
        for f in files:
            f_path = os.path.join(root, f)
            with open(f_path, 'r', encoding='ISO-8859-1') as ff:
                content = ff.read()
            content = [x.strip('.,?!()') for x in content.split() if len(x)]
            splited_docs.append(content)
            class_vec.append(1)

    training_dataset = splited_docs[:20] + splited_docs[25:35]
    training_class_vec = class_vec[:20] + class_vec[25:35]
    test_dataset = splited_docs[20:25] + splited_docs[35:50]
    test_class_vec = class_vec[20:25] + class_vec[35:50]

    vocablist = create_vocab_list2(training_dataset)
    for index, doc in enumerate(training_dataset):
        training_dataset[index] = bagofwords2vec(vocablist, doc)
    p0_vect, p1_vect, p_abusive = train_naive_bayes(training_dataset, training_class_vec)

    err_count = 0
    for index, email in enumerate(test_dataset):
        email = bagofwords2vec(vocablist, email)
        class_result = classify_naive_bayes(email, p0_vect, p1_vect, p_abusive)
        if class_result != test_class_vec[index]:
            err_count += 1

    print(err_count)

email_classify()
