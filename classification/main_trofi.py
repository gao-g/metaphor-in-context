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
import random
import math
import numpy as np
# import matplotlib
#
# matplotlib.use('Agg')  # to avoid the error: _tkinter.TclError: no display name and no $DISPLAY environment variable
# matplotlib.use('tkagg')  # to display the graph on remote server
# import matplotlib.pyplot as plt

print("PyTorch version:")
print(torch.__version__)
print("GPU Detected:")
print(torch.cuda.is_available())
using_GPU = True

"""
1. Data pre-processing
"""
'''
roFi
get raw dataset as a list:
  Each element is a triple:
    a sentence: string
    a index: int: idx of the focus verb
    a label: int 1 or 0
'''
raw_trofi = []

# normal version
with open('../data/TroFi/TroFi_formatted_all3737.csv') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        raw_trofi.append([line[1].strip(), int(line[2]), int(line[3])])
print('TroFi dataset size: ', len(raw_trofi))

"""
2. Data preparation
"""
'''
2. 1
get vocabulary and glove embeddings in raw dataset 
'''
# vocab is a set of words
vocab = get_vocab(raw_trofi)
# two dictionaries. <PAD>: 0, <UNK>: 1
word2idx, idx2word = get_word2idx_idx2word(vocab)
# glove_embeddings a nn.Embeddings
glove_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
# elmo_embeddings
# set elmos_mohx=None to exclude elmo vectors
elmos_trofi = h5py.File('../elmo/TroFi3737.hdf5', 'r')
# suffix_embeddings: number of suffix tag is 2, and the suffix embedding dimension is 50
suffix_embeddings = nn.Embedding(2, 50)

'''
2. 2
embed the datasets
'''
random.seed(0)
random.shuffle(raw_trofi)

embedded_trofi = [[embed_sequence(example[0], example[1], word2idx,
                                 glove_embeddings, elmos_trofi, suffix_embeddings), example[2]]
                 for example in raw_trofi]


'''
2. 3
set up Dataloader for batching
'''
'''
2. 3 10-fold cross validation
'''
# separate the embedded_sentences and labels into 2 list, in order to pass into the TextDataset as argument
sentences = [example[0] for example in embedded_trofi]
labels = [example[1] for example in embedded_trofi]
# ten_folds is a list of 10 tuples, each tuple is (list_of_embedded_sentences, list_of_corresponding_labels)
ten_folds = []
fold_size = int(3737/10)
for i in range(10):
    ten_folds.append((sentences[i*fold_size:(i+1)*fold_size], 
                      labels[i*fold_size:(i+1)*fold_size]))

optimal_f1s = []
optimal_ps = []
optimal_rs = []
optimal_accs = []
predictions_all = []
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
    training_dataset_trofi = TextDataset(training_sentences, training_labels)
    val_dataset_trofi = TextDataset(ten_folds[i][0], ten_folds[i][1])

    # Data-related hyperparameters
    batch_size = 10
    # Set up a DataLoader for the training, validation, and test dataset
    train_dataloader_trofi = DataLoader(dataset=training_dataset_trofi, batch_size=batch_size, shuffle=True,
                                      collate_fn=TextDataset.collate_fn)
    val_dataloader_trofi = DataLoader(dataset=val_dataset_trofi, batch_size=batch_size, shuffle=False,
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
    rnn_clf = RNNSequenceClassifier(num_classes=2, embedding_dim=300+1024+50, hidden_size=300,
                                    num_layers=1, bidir=True,
                                    dropout1=0.2, dropout2=0, dropout3=0)
    # Move the model to the GPU if available
    if using_GPU:
        rnn_clf = rnn_clf.cuda()
    # Set up criterion for calculating loss
    nll_criterion = nn.NLLLoss()
    # Set up an optimizer for updating the parameters of the rnn_clf
    rnn_clf_optimizer = optim.Adam(rnn_clf.parameters(), lr=0.001)
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
    val_p = []
    val_r = []
    val_acc = []
    # A counter for the number of gradient updates
    num_iter = 0
    train_dataloader = train_dataloader_trofi 
    val_dataloader = val_dataloader_trofi
    model_index = 0
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
                val_p.append(precision)
                val_r.append(recall)
                val_acc.append(eval_accuracy)
                print(
                    "Iteration {}. Validation Loss {}. Validation Accuracy {}. Validation Precision {}. Validation Recall {}. Validation F1 {}. Validation class-wise F1 {}.".format(
                        num_iter, avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1))
#                 filename = '../models/LSTMSuffixElmoAtt_MOH_fold_' + str(i) + '_epoch_' + str(model_index) + '.pt'
#                 torch.save(rnn_clf, filename)
                model_index += 1
#                 avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1 = evaluate(train_dataloader, rnn_clf, nll_criterion, using_GPU)
#                 training_loss.append(avg_eval_loss)
#                 training_f1.append(f1)
#                 print(
#                     "Iteration {}. Training Loss {}. Training Accuracy {}. Training Precision {}. Training Recall {}. Training F1 {}. Training class-wise F1 {}.".format(
#                         num_iter, avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1))

    """
    additional trianing!
    """
#     rnn_clf_optimizer = optim.Adam(rnn_clf.parameters(), lr=0.0005)
#     for epoch in range(num_epochs):
#         print("Starting epoch {}".format(epoch + 1))
#         for (example_text, example_lengths, labels) in train_dataloader:
#             example_text = Variable(example_text)
#             example_lengths = Variable(example_lengths)
#             labels = Variable(labels)
#             if using_GPU:
#                 example_text = example_text.cuda()
#                 example_lengths = example_lengths.cuda()
#                 labels = labels.cuda()
#             # predicted shape: (batch_size, 2)
#             predicted = rnn_clf(example_text, example_lengths)
#             batch_loss = nll_criterion(predicted, labels)
#             rnn_clf_optimizer.zero_grad()
#             batch_loss.backward()
#             rnn_clf_optimizer.step()
#             num_iter += 1
#             # Calculate validation and training set loss and accuracy every 200 gradient updates
#             if num_iter % 100 == 0:
#                 avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1 = evaluate(val_dataloader, rnn_clf, nll_criterion, using_GPU)
#                 val_loss.append(avg_eval_loss)
#                 val_f1.append(f1)
#                 val_p.append(precision)
#                 val_r.append(recall)
#                 val_acc.append(eval_accuracy)
#                 print(
#                     "Iteration {}. Validation Loss {}. Validation Accuracy {}. Validation Precision {}. Validation Recall {}. Validation F1 {}. Validation class-wise F1 {}.".format(
#                         num_iter, avg_eval_loss, eval_accuracy, precision, recall, f1, fus_f1))
#                 model_index += 1
    
    print("Training done for fold {}".format(i))

    """
    3.3
    plot the training process: MET F1 and losses for validation and training dataset
    """
#     plt.figure(0)
#     plt.title('F1 for TroFI dataset on fold ' + str(i))
#     plt.xlabel('iteration (unit:200)')
#     plt.ylabel('F1')
#     plt.plot(val_f1,'g')
#     plt.plot(val_p,'r')
#     plt.plot(val_r,'b')
#     plt.plot(val_acc,'c')
#     plt.plot(training_f1, 'b')
#     plt.legend(['Validation F1', 'Validation precision', 'validaiton recall', 'validation accuracy', 'Training F1'], loc='upper right')
#     plt.show()


#     plt.figure(1)
#     plt.title('Loss for TroFi dataset on fold ' + str(i))
#     plt.xlabel('iteration (unit:200)')
#     plt.ylabel('Loss')
#     plt.plot(val_loss,'g')
#     plt.plot(training_loss, 'b')
#     plt.legend(['Validation loss', 'Training loss'], loc='upper right')
#     plt.show()

    """
    store the best f1
    """
    print('val_f1: ', val_f1)
    idx = 0
    if math.isnan(max(val_f1)):
        optimal_f1s.append(max(val_f1[6:]))
        idx = val_f1.index(optimal_f1s[-1])
        optimal_ps.append(val_p[idx])
        optimal_rs.append(val_r[idx])
        optimal_accs.append(val_acc[idx])
    else:
        optimal_f1s.append(max(val_f1))
        idx = val_f1.index(optimal_f1s[-1])
        optimal_ps.append(val_p[idx])
        optimal_rs.append(val_r[idx])
        optimal_accs.append(val_acc[idx])
#     filename = '../models/LSTMSuffixElmoAtt_TroFi_fold_' + str(i) + '_epoch_' + str(idx) + '.pt'
#     temp_model = torch.load(filename)
#     print('best model: ', filename)
#     predictions_all.extend(test(val_dataloader_TroFi, temp_model, using_GPU))
        
print('F1 on TroFi by 10-fold = ', optimal_f1s)
print('Precision on TroFi = ', np.mean(np.array(optimal_ps)))
print('Recall on TroFi = ', np.mean(np.array(optimal_rs)))
print('F1 on TroFi = ', np.mean(np.array(optimal_f1s)))
print('Accuracy on TroFi = ', np.mean(np.array(optimal_accs)))
# plt.figure(2)
# plt.title('F1 for TroFi dataset on ten folds')
# plt.xlabel('fold')
# plt.ylabel('F1')
# plt.plot(optimal_ps,'r')
# plt.plot(optimal_rs,'b')
# plt.plot(optimal_f1s,'g')
# plt.plot(optimal_accs,'c')
# plt.plot([np.mean(np.array(optimal_f1s))] * 10, 'y')
# plt.legend(['precision for each fold', 'recall for each fold', 'F1 for each fold', 'accuracy for each fold', 'Average F1'], loc='upper right')
# plt.show()