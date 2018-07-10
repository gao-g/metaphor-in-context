import csv
import ast
import numpy as np

"""
Compute the lexical baseline for VUA dataset across all pos tags.
Each word is considered as as an instance.
"""


def get_model_for_each_pos_tag(dataset, all_pos):
    """

    :param dataset: a list of triples: [string sentence, label_seq, pos_seq]
    :param all_pos: list of all pos tags: assume the complete set for the dataset
    :return: a dictionary of pog tag to metaphorical words in that category
    """
    # (NOUN, apple) --> [sum, total] sum is the number of metaphorical labels and total is the number of occurrences
    pos_word2sum_total = {}
    # NOUN --> metaphorical set of words [apple, water, ... ]
    pos2words = {pos: set() for pos in all_pos}

    for sen, label_seq, pos_seq in dataset:
        words = sen.split()
        for i in range(len(words)):
            key = (pos_seq[i], words[i])
            if key in pos_word2sum_total:
                pos_word2sum_total[key][0] += label_seq[i]
                pos_word2sum_total[key][1] += 1
            else:
                pos_word2sum_total[key] = [label_seq[i], 1]

    for pos_word, sum_total in pos_word2sum_total.items():
        if sum_total[0] / sum_total[1] > 0.5:
            pos2words[pos_word[0]].add(pos_word[1])
    return pos2words


def getModel(dataset):
    """

    :param dataset: a list of (verb, label) pairs
    :return: a dictioanry: verb --> number that shows the probability of being metaphor
    """
    model = {}
    for verb, label in dataset:
        if verb in model:
            model[verb].append(label)
        else:
            model[verb] = [label]

    final_model = {}
    for key in model.keys():
        value = model[key]
        prob = sum(value) / len(value)
        if prob > 0.5:
            final_model[key] = prob
    return final_model


def predict(model, dataset):
    """

    :param model: a dictionary of metaphorical verbs
    :param dataset: a list of verb-label pairs
    :return: two lists, predictions, and labels with alignment
    """
    # predict
    predictions = []
    labels = []
    for verb, label in dataset:
        labels.append(label)
        if verb in model:
            predictions.append(1)
        else:
            predictions.append(0)
    return evaluate(predictions, labels)


def evaluate(predictions, labels):
    """

    :param predictions: a list of 1s and 0s
    :param labels: a list of 1s and 0s
    :return: 4 numbers: precision, recall, met_f1, accuracy
    assume the given two lists have alignment
    """
    # Set model to eval mode, which turns off dropout.
    assert(len(predictions) == len(labels))
    total_examples = len(predictions)

    num_correct = 0
    confusion_matrix = np.zeros((2, 2))
    for i in range(total_examples):
        if predictions[i] == labels[i]:
            num_correct += 1
        confusion_matrix[predictions[i], labels[i]] += 1

    assert(num_correct == confusion_matrix[0, 0] + confusion_matrix[1, 1])
    accuracy = 100 * num_correct / total_examples
    precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
    recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
    met_f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, met_f1, accuracy

raw_train_vua = []
with open('../data/VUAsequence/VUA_seq_formatted_train.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        label_seq = ast.literal_eval(line[3])
        words = line[2].split()
        for i in range(len(words)):
            raw_train_vua.append([words[i], label_seq[i]])
raw_val_vua = []
with open('../data/VUAsequence/VUA_seq_formatted_val.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        label_seq = ast.literal_eval(line[3])
        words = line[2].split()
        for i in range(len(words)):
            raw_val_vua.append([words[i], label_seq[i]])
raw_test_vua = []
with open('../data/VUAsequence/VUA_seq_formatted_test.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        label_seq = ast.literal_eval(line[3])
        words = line[2].split()
        for i in range(len(words)):
            raw_test_vua.append([words[i], label_seq[i]])

print('VUA dataset division: ', len(raw_train_vua), len(raw_val_vua), len(raw_test_vua))

vua_model = getModel(raw_train_vua+raw_val_vua)
print('P, R, F1, Acc. for VUA test dataset: ', predict(vua_model, raw_test_vua))
