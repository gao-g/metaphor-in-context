from util import get_num_lines, get_vocab, embed_sequence, get_word2idx_idx2word, get_embedding_matrix
from util import TextDatasetWithGloveElmoSuffix as TextDataset
from util import evaluate
from model import RNNSequenceClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import csv
import random
import h5py
# import matplotlib
#
# matplotlib.use('Agg')  # to avoid the error: _tkinter.TclError: no display name and no $DISPLAY environment variable
# matplotlib.use('tkagg')  # to display the graph on remote server
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
1.4 MOH-X
get raw dataset as a list:
  Each element is a triple:
    a sentence: string
    a index: int: idx of the focus verb
    a label: int 1 or 0
'''
raw_mohX = []
with open('../data/MOH-X/MOH-X_formatted_svo_cleaned.csv') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        raw_mohX.append([line[3][1:], int(line[4]), int(line[5])])
print('MOH-X dataset', len(raw_mohX))

"""
2. Data preparation
"""
'''
2. 1
get vocabulary and glove embeddings in raw dataset 
'''
# vocab is a set of words
vocab = get_vocab(raw_mohX)
# two dictionaries. <PAD>: 0, <UNK>: 1
word2idx, idx2word = get_word2idx_idx2word(vocab)
# glove_embeddings a nn.Embeddings
glove_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
# elmo_embeddings
# To exclude elmos: use elmos_mohx=None, and change embedding_dim in later model initiation.
elmos_mohx = h5py.File('../elmo/MOH-X_cleaned.hdf5', 'r')
# suffix_embeddings: number of suffix tag is 2, and the suffix embedding dimension is 50
suffix_embeddings = nn.Embedding(2, 50)

'''
2. 2
embed the datasets
'''
embedded_mohX = [[embed_sequence(example[0], example[1], word2idx,
                                 glove_embeddings, elmos_mohx, suffix_embeddings), example[2]]
                 for example in raw_mohX]


'''
2. 3 10-fold cross validation
'''
# separate the embedded_sentences and labels into 2 list, in order to pass into the TextDataset as argument
sentences = [example[0] for example in embedded_mohX]
labels = [example[1] for example in embedded_mohX]
# ten_folds is a list of 10 tuples, each tuple is (list_of_embedded_sentences, list_of_corresponding_labels)
ten_folds = []
for i in range(10):
    ten_folds.append((sentences[i*65:(i+1)*65], labels[i*65:(i+1)*65]))

optimal_f1s = []
for i in range(10):
    '''
    2. 3
    set up Dataloader for batching
    '''
    training_sentences = []
    training_labels = []
    for j in range(10):
        if j != i:
            training_sentences.extend(ten_folds[j][0])
            training_labels.extend(ten_folds[j][1])
    training_dataset_mohX = TextDataset(training_sentences, training_labels)
    val_dataset_mohX = TextDataset(ten_folds[i][0], ten_folds[i][1])

    # Data-related hyperparameters
    batch_size = 10
    # Set up a DataLoader for the training, validation, and test dataset
    train_dataloader_mohX = DataLoader(dataset=training_dataset_mohX, batch_size=batch_size, shuffle=True,
                                      collate_fn=TextDataset.collate_fn)
    val_dataloader_mohX = DataLoader(dataset=val_dataset_mohX, batch_size=batch_size, shuffle=True,
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
    # dropout2: dropout in RNN; would be used if num_layers!=1
    # dropout3: dropout on hidden state of RNN to linear layer
    rnn_clf = RNNSequenceClassifier(num_classes=2, embedding_dim=300+1024+50, hidden_size=300, num_layers=1, bidir=True,
                     dropout1=0.2, dropout2=0, dropout3=0.2)
    # Move the model to the GPU if available
    if using_GPU:
        rnn_clf = rnn_clf.cuda()
    # Set up criterion for calculating loss
    nll_criterion = nn.NLLLoss()
    # Set up an optimizer for updating the parameters of the rnn_clf
    rnn_clf_optimizer = optim.SGD(rnn_clf.parameters(), lr=0.02, momentum=0.9)
    # Number of epochs (passes through the dataset) to train the model for.
    num_epochs = 30

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
    train_dataloader = train_dataloader_mohX
    val_dataloader = val_dataloader_mohX
    for epoch in range(num_epochs):
        print("Starting epoch {}".format(epoch + 1))
        for (example_text, example_lengths, labels) in train_dataloader:
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
                avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1 = evaluate(val_dataloader, rnn_clf, nll_criterion, using_GPU)
                val_loss.append(avg_eval_loss)
                val_f1.append(f1)
                print(
                    "Iteration {}. Validation Loss {}. Validation Accuracy {}. Validation Precision {}. Validation Recall {}. Validation F1 {}. Validation class-wise F1 {}.".format(
                        num_iter, avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1))
                # filename = '../models/LSTMSuffixElmoAtt_???_all_iter_' + str(num_iter) + '.pt'
                # torch.save(rnn_clf, filename)
                avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1 = evaluate(train_dataloader, rnn_clf, nll_criterion, using_GPU)
                training_loss.append(avg_eval_loss)
                training_f1.append(f1)
                print(
                    "Iteration {}. Training Loss {}. Training Accuracy {}. Training Precision {}. Training Recall {}. Training F1 {}. Training class-wise F1 {}.".format(
                        num_iter, avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1))
    print("Training done for fold {}".format(i))

    """
    3.3
    plot the training process: MET F1 and losses for validation and training dataset
    """
#     plt.figure(0)
#     plt.title('F1 for MOH-X dataset on fold ' + str(i))
#     plt.xlabel('iteration (unit:200)')
#     plt.ylabel('F1')
#     plt.plot(val_f1,'g')
#     plt.plot(training_f1, 'b')
#     plt.legend(['Validation F1', 'Training F1'], loc='upper right')
#     plt.show()


#     plt.figure(1)
#     plt.title('Loss for MOH-X dataset on fold ' + str(i))
#     plt.xlabel('iteration (unit:200)')
#     plt.ylabel('Loss')
#     plt.plot(val_loss,'g')
#     plt.plot(training_loss, 'b')
#     plt.legend(['Validation loss', 'Training loss'], loc='upper right')
#     plt.show()

    """
    store the best f1
    """
    optimal_f1s.append(max(val_f1))

print('F1 on MOH-X by 10-fold = ', optimal_f1s)
print('F1 on MOH-X = ', np.mean(np.array(optimal_f1s)))
# plt.figure(2)
# plt.title('F1 for MOH-X dataset on ten folds')
# plt.xlabel('fold')
# plt.ylabel('F1')
# plt.plot(optimal_f1s,'r')
# plt.plot([np.mean(np.array(optimal_f1s))] * 10, 'y')
# plt.legend(['F1 for each fold', 'Average F1'], loc='upper right')
# plt.show()
