import csv
import numpy as np
import random

"""
Compute the lexical baseline for VUA-verb, MOH-X, TroFi dataset (just verbs).
"""


"""
1. Data pre-processing
get raw dataset as a list:
  Each element is a pair:
    a verb
    a label: int 1 or 0
"""
'''
1.1 VUA

'''
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
        raw_test_vua.append([line[2], int(line[5])])

print('VUA dataset division: ', len(raw_train_vua), len(raw_val_vua), len(raw_test_vua))

"""
2. Establish model
model is a dictionary of metaphorical verbs
"""


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


vua_model = getModel(raw_train_vua+raw_val_vua)


"""
3. predict
"""


def predict(model, dataset):
    """

    :param model: a dictionary of metaphorical verbs
    :param dataset: a list of verb-label pairs
    :return: two lists, predictions, and labels
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

    :param predictions: a list
    :param labels: a list
    :return: 4 numbers: precision, recall, met_f1, accuracy
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


print('P, R, F1, Acc. for VUA test dataset: ', predict(vua_model, raw_test_vua))



"""
MOH-X lexical baseline with ten-fold cross validation
"""
# load dataset
raw_mohX = []
with open('../data/MOH-X/MOH-X_formatted_svo_cleaned.csv') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        # line[3] is the sentence, which starts from a space
        raw_mohX.append([line[2], int(line[5])])
print('MOH-X dataset size:', len(raw_mohX))

# shuffle
random.seed(3)
random.shuffle(raw_mohX)

# prepare 10 folds
ten_folds = []
for i in range(10):
    ten_folds.append(raw_mohX[i*65:(i+1)*65])

# 10 fold
PRFA_list = []
for i in range(10):
    raw_train_mohX = []
    raw_val_mohX = []
    # seperate training and validation data
    for j in range(10):
        if j != i:
            raw_train_mohX.extend(ten_folds[j])
        else:
            raw_val_mohX = ten_folds[j]
    # make model, predict, and evaluate
    mohX_model = getModel(raw_train_mohX)
    PRFA_list.append(predict(mohX_model, raw_val_mohX))

# average result
PRFA = np.array(PRFA_list)
print('P, R, F1, Acc. for MOH-X dataset: ', np.mean(PRFA, axis=0))

"""
MOH-X lexical baseline with ten-fold cross validation
"""
# load dataset
raw_trofi = []  # [verb, label]
with open('../data/TroFi/TroFi_formatted_all3737.csv') as f:
    # verb	sentence	verb_idx	label
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        # line[3] is the sentence, which starts from a space
        raw_trofi.append([line[0], int(line[3])])
print('TroFi dataset size:', len(raw_trofi))

# shuffle
random.seed(0)
random.shuffle(raw_trofi)

# prepare 10 folds
ten_folds = []
fold_size = int(3737/10)
for i in range(10):
    ten_folds.append(raw_trofi[i * fold_size: (i + 1) * fold_size])

# 10 fold
PRFA_list = []
for i in range(10):
    raw_train_trofi = []
    raw_val_trofi = []
    # separate training and validation data
    for j in range(10):
        if j != i:
            raw_train_trofi.extend(ten_folds[j])
        else:
            raw_val_trofi = ten_folds[j]
    # make model, predict, and evaluate
    trofi_model = getModel(raw_train_trofi)
    PRFA_list.append(predict(trofi_model, raw_val_trofi))

# average result
PRFA = np.array(PRFA_list)
print('P, R, F1, Acc. for TroFi dataset: ', np.mean(PRFA, axis=0))

