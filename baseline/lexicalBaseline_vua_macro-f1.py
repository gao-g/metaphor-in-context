import csv
import numpy as np

"""
Compute the lexical baseline for VUA verb dataset across 4 genres
Report the macro averaged performance
"""

genre2idx = {'news': 0, 'fiction': 1, 'academic': 2, 'conversation': 3}
idx2genre = {genre2idx[x]: x for x in genre2idx}
# sentence ID --> genre
ID2genre = {}
with open('../data/VUAsequence/VUA_seq_formatted_test.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        ID2genre[(line[0], line[1])] = line[6]

raw_train_vua = []
with open('../data/VUA/VUA_formatted_train.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        raw_train_vua.append([line[2], int(line[5])])

raw_val_vua = []
with open('../data/VUA/VUA_formatted_val.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        raw_val_vua.append([line[2], int(line[5])])

raw_test_vua = []
with open('../data/VUA/VUA_formatted_test.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        raw_test_vua.append([line[2], int(line[5]), (line[0], line[1])])

print('VUA dataset division: ', len(raw_train_vua), len(raw_val_vua), len(raw_test_vua))


def getModel(dataset):
    """

    :param dataset:
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
    :param dataset: a list of verb-label-ID pairs
    """
    # predict
    predictions = []
    labels = []
    genres = []
    for verb, label, ID in dataset:
        labels.append(label)
        genres.append(ID2genre[ID])
        if verb in model:
            predictions.append(1)
        else:
            predictions.append(0)
    evaluate(predictions, labels, genres)


def evaluate(predictions, labels, genres):
    """

    :param predictions: a list of prediction per instance
    :param labels: a list of label per instance
    :param genres: a list of genres per instance
    :return: 4 numbers: macro-averaging: precision, recall, met_f1, accuracy
    """
    # Set model to eval mode, which turns off dropout.
    assert(len(predictions) == len(labels))
    assert(len(predictions) == len(genres))

    confusion_matrix = np.zeros((len(genre2idx), 2, 2))
    for i in range(len(predictions)):
        confusion_matrix[genre2idx[genres[i]],predictions[i], labels[i]] += 1

    for i in range(len(idx2genre)):
        accuracy = 100 * (confusion_matrix[i, 1, 1] + confusion_matrix[i, 0, 0]) / np.sum(confusion_matrix[i])
        precision = 100 * confusion_matrix[i, 1, 1] / np.sum(confusion_matrix[i, 1])
        recall = 100 * confusion_matrix[i, 1, 1] / np.sum(confusion_matrix[i, :, 1])
        met_f1 = 2 * precision * recall / (precision + recall)
        print('genre: {} PRFA test performance: {} {} {} {}'.format(idx2genre[i],  precision, recall, met_f1, accuracy))


vua_model = getModel(raw_train_vua+raw_val_vua)
predict(vua_model, raw_test_vua)
