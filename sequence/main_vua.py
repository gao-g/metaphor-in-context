from util import get_num_lines, get_pos2idx_idx2pos, index_sequence, get_vocab, embed_indexed_sequence, \
    get_word2idx_idx2word, get_embedding_matrix, write_predictions, get_performance_VUAverb_val
from util import TextDatasetWithGloveElmoSuffix as TextDataset
from util import evaluate
from model import RNNSequenceModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import csv
import h5py
import ast
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
    a list of labels: 
    a list of pos: 

'''
pos_set = set()
raw_train_vua = []
with open('../data/VUAsequence/VUA_seq_formatted_train.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        pos_seq = ast.literal_eval(line[4])
        label_seq = ast.literal_eval(line[3])
        assert (len(pos_seq) == len(label_seq))
        assert (len(line[2].split()) == len(pos_seq))
        raw_train_vua.append([line[2], label_seq, pos_seq])
        pos_set.update(pos_seq)

raw_val_vua = []
with open('../data/VUAsequence/VUA_seq_formatted_val.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        pos_seq = ast.literal_eval(line[4])
        label_seq = ast.literal_eval(line[3])
        assert (len(pos_seq) == len(label_seq))
        assert (len(line[2].split()) == len(pos_seq))
        raw_val_vua.append([line[2], label_seq, pos_seq])
        pos_set.update(pos_seq)

# embed the pos tags
pos2idx, idx2pos = get_pos2idx_idx2pos(pos_set)

for i in range(len(raw_train_vua)):
    raw_train_vua[i][2] = index_sequence(pos2idx, raw_train_vua[i][2])
for i in range(len(raw_val_vua)):
    raw_val_vua[i][2] = index_sequence(pos2idx, raw_val_vua[i][2])
print('size of training set, validation set: ', len(raw_train_vua), len(raw_val_vua))


"""
2. Data preparation
"""
'''
2. 1
get vocabulary and glove embeddings in raw dataset 
'''
# vocab is a set of words
vocab = get_vocab(raw_train_vua)
# two dictionaries. <PAD>: 0, <UNK>: 1
word2idx, idx2word = get_word2idx_idx2word(vocab)
# glove_embeddings a nn.Embeddings
glove_embeddings = get_embedding_matrix(word2idx, idx2word, normalization=False)
# elmo_embeddings
elmos_train_vua = h5py.File('../elmo/VUA_train.hdf5', 'r')
elmos_val_vua = h5py.File('../elmo/VUA_val.hdf5', 'r')


'''
2. 2
embed the datasets
'''
# raw_train_vua: sentence, label_seq, pos_seq
# embedded_train_vua: embedded_sentence, pos, labels
embedded_train_vua = [[embed_indexed_sequence(example[0], example[2], word2idx,
                                      glove_embeddings, elmos_train_vua, suffix_embeddings),
                       example[2], example[1]]
                      for example in raw_train_vua]
embedded_val_vua = [[embed_indexed_sequence(example[0], example[2], word2idx,
                                    glove_embeddings, elmos_val_vua, suffix_embeddings),
                     example[2], example[1]]
                    for example in raw_val_vua]


'''
2. 3
set up Dataloader for batching
'''
# Separate the input (embedded_sequence) and labels in the indexed train sets.
# embedded_train_vua: embedded_sentence, pos, labels
train_dataset_vua = TextDataset([example[0] for example in embedded_train_vua],
                                [example[1] for example in embedded_train_vua],
                                [example[2] for example in embedded_train_vua])
val_dataset_vua = TextDataset([example[0] for example in embedded_val_vua],
                              [example[1] for example in embedded_val_vua],
                              [example[2] for example in embedded_val_vua])

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
# dropout2: dropout in RNN; would be used if num_layers!=1
# dropout3: dropout on hidden state of RNN to linear layer
RNNseq_model = RNNSequenceModel(num_classes=2, embedding_dim=300 + 1024, hidden_size=300, num_layers=1, bidir=True,
                                dropout1=0.5, dropout2=0, dropout3=0.1)
# Move the model to the GPU if available
if using_GPU:
    RNNseq_model = RNNseq_model.cuda()
# Set up criterion for calculating loss
loss_criterion = nn.NLLLoss()
# Set up an optimizer for updating the parameters of the rnn_clf
rnn_optimizer = optim.Adam(RNNseq_model.parameters(), lr=0.001)
# Number of epochs (passes through the dataset) to train the model for.
num_epochs = 10

'''
3. 2
train model
'''
train_loss = []
val_loss = []
performance_matrix = None
val_f1s = []
train_f1s = []
# A counter for the number of gradient updates
num_iter = 0
comparable = []
for epoch in range(num_epochs):
    print("Starting epoch {}".format(epoch + 1))
    for (__, example_text, example_lengths, labels) in train_dataloader_vua:
        example_text = Variable(example_text)
        example_lengths = Variable(example_lengths)
        labels = Variable(labels)
        if using_GPU:
            example_text = example_text.cuda()
            example_lengths = example_lengths.cuda()
            labels = labels.cuda()
        # predicted shape: (batch_size, seq_len, 2)
        predicted = RNNseq_model(example_text, example_lengths)
        batch_loss = loss_criterion(predicted.view(-1, 2), labels.view(-1))
        rnn_optimizer.zero_grad()
        batch_loss.backward()
        rnn_optimizer.step()
        num_iter += 1
        # Calculate validation and training set loss and accuracy every 200 gradient updates
        if num_iter % 200 == 0:
            avg_eval_loss, performance_matrix = evaluate(idx2pos, val_dataloader_vua, RNNseq_model,
                                                         loss_criterion, using_GPU)
            val_loss.append(avg_eval_loss)
            val_f1s.append(performance_matrix[:, 2])
            print("Iteration {}. Validation Loss {}.".format(num_iter, avg_eval_loss))
            avg_eval_loss, performance_matrix = evaluate(idx2pos, train_dataloader_vua, RNNseq_model,
                                                         loss_criterion, using_GPU)
            train_loss.append(avg_eval_loss)
            train_f1s.append(performance_matrix[:, 2])
            print("Iteration {}. Training Loss {}.".format(num_iter, avg_eval_loss))
print("Training done!")

"""
3.3
plot the training process: losses for validation and training dataset
"""
plt.figure(0)
plt.title('Loss for VUA dataset')
plt.xlabel('iteration (unit:200)')
plt.ylabel('Loss')
plt.plot(val_loss, 'g')
plt.plot(train_loss, 'b')
plt.legend(['Validation loss', 'Training loss'], loc='upper right')
plt.show()

plt.figure(1)
plt.title('Validation F1 for VUA dataset')
plt.xlabel('iteration (unit:200)')
plt.ylabel('F1')
for i in range(len(idx2pos)):
    plt.plot([x[i] for x in val_f1s])
plt.legend([idx2pos[i] for i in range(len(idx2pos))], loc='upper left')
plt.show()

plt.figure(2)
plt.title('Training F1 for VUA dataset')
plt.xlabel('iteration (unit:200)')
plt.ylabel('F1')
for i in range(len(idx2pos)):
    plt.plot([x[i] for x in train_f1s])
plt.legend([idx2pos[i] for i in range(len(idx2pos))], loc='upper left')
plt.show()
