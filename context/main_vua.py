from util import get_num_lines, get_vocab, embed_sequence, get_word2idx_idx2word, get_embedding_matrix
from util import TextDatasetWithGloveElmoSuffix as TextDataset
from util import evaluate
from model import RNNSequenceClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import csv
import h5py
# import matplotlib
# matplotlib.use('Agg')  # to avoid the error: _tkinter.TclError: no display name and no $DISPLAY environment variable
# matplotlib.use('tkagg') # to display the graph on remote server
import matplotlib.pyplot as plt

print("PyTorch version:")
print(torch.__version__)
print("GPU Detected:")
print(torch.cuda.is_available())
using_GPU = True

"""
1. Data pre-processing
"""
'''
1.1 VUA
get raw dataset as a list:
  Each element is a triple:
    a sentence: string
    a index: int: idx of the focus verb
    a label: int 1 or 0

'''
raw_train_vua = []
with open('../data/VUA/VUA_formatted_train.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        raw_train_vua.append([line[3], int(line[4]), int(line[5])])

raw_val_vua = []
with open('../data/VUA/VUA_formatted_val.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        raw_val_vua.append([line[3], int(line[4]), int(line[5])])

raw_test_vua = []
with open('../data/VUA/VUA_formatted_test.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        raw_test_vua.append([line[3], int(line[4]), int(line[5])])
print('VUA dataset division: ', len(raw_train_vua), len(raw_val_vua), len(raw_test_vua))

"""
2. Data preparation
"""
'''
2. 1
get vocabulary and glove embeddings in raw dataset 
'''
# vocab is a set of words
vocab = get_vocab(raw_train_vua + raw_val_vua + raw_test_vua)
# two dictionaries. <PAD>: 0, <UNK>: 1
word2idx, idx2word = get_word2idx_idx2word(vocab)
# glove_embeddings a nn.Embeddings
glove_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
# elmo_embeddings
elmos_train_vua = h5py.File('../elmo/VUA_train.hdf5', 'r')
elmos_val_vua = h5py.File('../elmo/VUA_val.hdf5', 'r')
# suffix_embeddings: number of suffix tag is 2, and the suffix embedding dimension is 50
suffix_embeddings = nn.Embedding(2, 50)

'''
2. 2
embed the datasets
'''
embedded_train_vua = [[embed_sequence(example[0], example[1], word2idx,
                                      glove_embeddings, elmos_train_vua, suffix_embeddings), example[2]]
                      for example in raw_train_vua]
embedded_val_vua = [[embed_sequence(example[0], example[1], word2idx,
                                    glove_embeddings, elmos_val_vua, suffix_embeddings), example[2]]
                    for example in raw_val_vua]

'''
2. 3
set up Dataloader for batching
'''
# Separate the input (embedded_sequence) and labels in the indexed train sets.
train_dataset_vua = TextDataset([example[0] for example in embedded_train_vua],
                                [example[1] for example in embedded_train_vua])
val_dataset_vua = TextDataset([example[0] for example in embedded_val_vua],
                              [example[1] for example in embedded_val_vua])

# Data-related hyperparameters
batch_size = 64
# Set up a DataLoader for the training, validation, and test dataset
train_dataloader_vua = DataLoader(dataset=train_dataset_vua, batch_size=batch_size, shuffle=True,
                                  collate_fn=TextDataset.collate_fn)
val_dataloader_vua = DataLoader(dataset=val_dataset_vua, batch_size=batch_size,
                                collate_fn=TextDataset.collate_fn)

"""
3. Model training
"""
'''
3. 1 
set up model, loss criterion, optimizer
'''
# Instantiate the model
# embedding_dim = glove + elmo + suffix indicator
# dropout1: dropout on input to RNN
# dropout2: dropout in RNN; would be used if num_layers=1
# dropout3: dropout on hidden state of RNN to linear layer
rnn_clf = RNNSequenceClassifier(num_classes=2, embedding_dim=300 + 1024 + 50, hidden_size=300, num_layers=1, bidir=True,
                                dropout1=0.2, dropout2=0.2, dropout3=0.2)
# Move the model to the GPU if available
if using_GPU:
    rnn_clf = rnn_clf.cuda()
# Set up criterion for calculating loss
nll_criterion = nn.NLLLoss()
# Set up an optimizer for updating the parameters of the rnn_clf
rnn_clf_optimizer = optim.SGD(rnn_clf.parameters(), lr=0.08, momentum=0.9)
# Number of epochs (passes through the dataset) to train the model for.
num_epochs = 15

'''
3. 2
train model
'''
training_loss = []
val_loss = []
training_f1 = []
val_f1 = []
# A counter for the number of gradient updates
num_iter = 0
for epoch in range(num_epochs):
    print("Starting epoch {}".format(epoch + 1))
    for (example_text, example_lengths, labels) in train_dataloader_vua:
        example_text = Variable(example_text)
        example_lengths = Variable(example_lengths)
        labels = Variable(labels)
        if using_GPU:
            example_text = example_text.cuda()
            example_lengths = example_lengths.cuda()
            labels = labels.cuda()
        # predicted shape: (batch_size, 2)
        predicted = rnn_clf(example_text, example_lengths)
        batch_loss = nll_criterion(predicted, labels)
        rnn_clf_optimizer.zero_grad()
        batch_loss.backward()
        rnn_clf_optimizer.step()
        num_iter += 1
        # Calculate validation and training set loss and accuracy every 200 gradient updates
        if num_iter % 200 == 0:
            avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1 = evaluate(val_dataloader_vua, rnn_clf,
                                                                                   nll_criterion, using_GPU)
            val_loss.append(avg_eval_loss)
            val_f1.append(f1)
            print(
                "Iteration {}. Validation Loss {}. Validation Accuracy {}. Validation Precision {}. Validation Recall {}. Validation F1 {}. Validation class-wise F1 {}.".format(
                    num_iter, avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1))
            # filename = '../models/LSTMSuffixElmoAtt_???_all_iter_' + str(num_iter) + '.pt'
            # torch.save(rnn_clf, filename)
            avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1 = evaluate(train_dataloader_vua, rnn_clf,
                                                                                   nll_criterion, using_GPU)
            training_loss.append(avg_eval_loss)
            training_f1.append(f1)
            print(
                "Iteration {}. Training Loss {}. Training Accuracy {}. Training Precision {}. Training Recall {}. Training F1 {}. Training class-wise F1 {}.".format(
                    num_iter, avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1))
print("Training done!")

# cannot display the graph in terminal on remote server
# """
# 3.3
# plot the training process: MET F1 and losses for validation and training dataset
# """
# plt.figure(0)
# plt.title('F1 for VUA dataset')
# plt.xlabel('iteration (unit:200)')
# plt.ylabel('F1')
# plt.plot(val_f1,'g')
# plt.plot(training_f1, 'b')
# plt.legend(['Validation F1', 'Training F1'], loc='upper right')
# plt.show()
#
#
# plt.figure(1)
# plt.title('Loss for VUA dataset')
# plt.xlabel('iteration (unit:200)')
# plt.ylabel('Loss')
# plt.plot(val_loss,'g')
# plt.plot(training_loss, 'b')
# plt.legend(['Validation loss', 'Training loss'], loc='upper right')
# plt.show()

"""
4. test the model
the following code is for test data of VUA
"""
'''
VUA
'''
elmos_test_vua = h5py.File('../elmo/VUA_test.hdf5', 'r')
embedded_test_vua = [[embed_sequence(example[0], example[1], word2idx,
                                     glove_embeddings, elmos_test_vua, suffix_embeddings), example[2]]
                     for example in raw_test_vua]
test_dataset_vua = TextDataset([example[0] for example in embedded_test_vua],
                               [example[1] for example in embedded_test_vua])
test_dataloader_vua = DataLoader(dataset=test_dataset_vua, batch_size=batch_size,
                                 collate_fn=TextDataset.collate_fn)
avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1 = evaluate(test_dataloader_vua, rnn_clf,
                                                                       nll_criterion, using_GPU)
print("Test Accuracy {}. Test Precision {}. Test Recall {}. Test F1 {}. Test class-wise F1 {}.".format(
    eval_accuracy, precision, recall, f1, fus_f1))

