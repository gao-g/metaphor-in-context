from tqdm import tqdm
import torch
import numpy as np
import mmap
import ast
import csv
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable


# Misc helper functions
# Get the number of lines from a filepath
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def get_pos2idx_idx2pos(vocab):
    """

    :param vocab: a set of strings: all pos tags
    :return: word2idx: a dictionary: string to an int
             idx2word: a dictionary: int to a string
    """
    word2idx = {}
    idx2word = {}
    for word in vocab:
        assigned_index = len(word2idx)
        word2idx[word] = assigned_index
        idx2word[assigned_index] = word
    return word2idx, idx2word


def index_sequence(item2idx, seq):
    """

    :param item2idx: a dictionary:  string to an int
    :param pos_seq: a list of pos tags
    :return: a list of ints
    """
    embed = []
    for x in seq:
        embed.append(item2idx[x])
    assert (len(seq) == len(embed))
    return embed


def get_embedding_matrix(word2idx, idx2word, normalization=False):
    """
    assume padding index is 0

    :param word2idx: a dictionary: string --> int, includes <PAD> and <UNK>
    :param idx2word: a dictionary: int --> string, includes <PAD> and <UNK>
    :param normalization:
    :return: an embedding matrix: a nn.Embeddings
    """
    # Load the GloVe vectors into a dictionary, keeping only words in vocab
    embedding_dim = 300
    glove_path = "../glove/glove840B300d.txt"
    glove_vectors = {}
    with open(glove_path) as glove_file:
        for line in tqdm(glove_file, total=get_num_lines(glove_path)):
            split_line = line.rstrip().split()
            word = split_line[0]
            if len(split_line) != (embedding_dim + 1) or word not in word2idx:
                continue
            assert (len(split_line) == embedding_dim + 1)
            vector = np.array([float(x) for x in split_line[1:]], dtype="float32")
            if normalization:
                vector = vector / np.linalg.norm(vector)
            assert len(vector) == embedding_dim
            glove_vectors[word] = vector

    print("Number of pre-trained word vectors loaded: ", len(glove_vectors))

    # Calculate mean and stdev of embeddings
    all_embeddings = np.array(list(glove_vectors.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_stdev = float(np.std(all_embeddings))
    print("Embeddings mean: ", embeddings_mean)
    print("Embeddings stdev: ", embeddings_stdev)

    # Randomly initialize an embedding matrix of (vocab_size, embedding_dim) shape
    # with a similar distribution as the pretrained embeddings for words in vocab.
    vocab_size = len(word2idx)
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean, embeddings_stdev)
    # Go through the embedding matrix and replace the random vector with a
    # pretrained one if available. Start iteration at 2 since 0, 1 are PAD, UNK
    for i in range(2, vocab_size):
        word = idx2word[i]
        if word in glove_vectors:
            embedding_matrix[i] = torch.FloatTensor(glove_vectors[word])
    if normalization:
        for i in range(vocab_size):
            embedding_matrix[i] = embedding_matrix[i] / float(np.linalg.norm(embedding_matrix[i]))
    embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    embeddings.weight = nn.Parameter(embedding_matrix)
    return embeddings


def get_vocab(raw_dataset):
    """
    return vocab set, and prints out the vocab size

    :param raw_dataset: a list of lists: each inner list is a triple:
                a sentence: string
                a list of labels:
                a list of pos:
    :return: a set: the vocabulary in the raw_dataset
    """
    vocab = []
    for example in raw_dataset:
        vocab.extend(example[0].split())
    vocab = set(vocab)
    print("vocab size: ", len(vocab))
    return vocab


def get_word2idx_idx2word(vocab):
    """

    :param vocab: a set of strings: vocabulary
    :return: word2idx: string to an int
             idx2word: int to a string
    """
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    for word in vocab:
        assigned_index = len(word2idx)
        word2idx[word] = assigned_index
        idx2word[assigned_index] = word
    return word2idx, idx2word


def embed_indexed_sequence(sentence, pos_seq, word2idx, glove_embeddings, elmo_embeddings,
                           pos_embeddings=None):
    """
    Assume that pos_seq maps well with sentence
    Assume that the given sentence is indexed by word2idx
    Assume that word2idx has 1 mapped to UNK
    Assume that word2idx maps well implicitly with glove_embeddings
    Assume that the given pos_seq is indexed by pos2idx
    Assume that pos2idx maps well implicitly with pos_embeddings
    i.e. the idx for each word is the row number for its corresponding embedding

    :param sentence: a single string: a sentence with space
    :param pos_seq: a list of ints: indexed pos_sequence
    :param word2idx: a dictionary: string --> int
    :param glove_embeddings: a nn.Embedding with padding idx 0
    :param elmo_embeddings: a h5py file
                    each group_key is a string: a sentence
                    each inside group is an np array (seq_len, 1024 elmo)
    :param pos_embeddings: a nn.Embedding without padding idx
    :return: a np.array (seq_len, embed_dim=glove+elmo+suffix)
    """
    words = sentence.split()

    # 1. embed the sequence by glove vector
    # Replace words with tokens, and 1 (UNK index) if words not indexed.
    indexed_sequence = [word2idx.get(x, 1) for x in words]
    # glove_part has shape: (seq_len, glove_dim)
    glove_part = glove_embeddings(Variable(torch.LongTensor(indexed_sequence)))

    # 2. embed the sequence by elmo vectors
    if elmo_embeddings is not None:
        elmo_part = elmo_embeddings[sentence]
        assert (elmo_part.shape == (len(words), 1024))

    # 3. embed the sequence by pos indicators i.e. whether it is a verb or not
    if pos_embeddings is not None:
        pos_part = pos_embeddings(Variable(torch.LongTensor(pos_seq)))

    # concatenate three parts: glove+elmo+suffix along axis 1
    # glove_part and pos_part are Variables, so we need to use .data
    # otherwise, throws weird ValueError: incorrect dimension, zero-dimension, etc..
    if elmo_embeddings is not None and pos_embeddings is not None:
        result = np.concatenate((glove_part.data, elmo_part), axis=1)
        result = np.concatenate((result, pos_part.data), axis=1)
    elif elmo_embeddings is None and pos_embeddings is None:
        result = glove_part.data
    elif pos_embeddings is not None:  # elmo = None, pos != None
        result = np.concatenate((glove_part.data, pos_part.data), axis=1)
    else:  # elmo != None, pos = None
        result = np.concatenate((glove_part.data, elmo_part), axis=1)

    assert (len(words) == result.shape[0])
    return result


def evaluate(idx2pos, evaluation_dataloader, model, criterion, using_GPU):
    """
    Evaluate the model on the given evaluation_dataloader

    :param evaluation_dataloader:
    :param model:
    :param criterion: loss criterion
    :param using_GPU: a boolean
    :return:
     average_eval_loss
     a matrix (#allpostags, 4) each row is the PRFA performance for a pos tag
    """
    # Set model to eval mode, which turns off dropout.
    model.eval()

    # total_examples = total number of words
    total_examples = 0
    total_eval_loss = 0
    confusion_matrix = np.zeros((len(idx2pos), 2, 2))
    for (eval_pos_seqs, eval_text, eval_lengths, eval_labels) in evaluation_dataloader:
        eval_text = Variable(eval_text, volatile=True)
        eval_lengths = Variable(eval_lengths, volatile=True)
        eval_labels = Variable(eval_labels, volatile=True)
        if using_GPU:
            eval_text = eval_text.cuda()
            eval_lengths = eval_lengths.cuda()
            eval_labels = eval_labels.cuda()

        # predicted shape: (batch_size, seq_len, 2)
        predicted = model(eval_text, eval_lengths)
        # Calculate loss for this test batch. This is averaged, so multiply
        # by the number of examples in batch to get a total.
        total_eval_loss += criterion(predicted.view(-1, 2), eval_labels.view(-1))
        # get 0 or 1 predictions
        # predicted_labels: (batch_size, seq_len)
        _, predicted_labels = torch.max(predicted.data, 2)
        total_examples += eval_lengths.size(0)
        confusion_matrix = update_confusion_matrix(confusion_matrix, predicted_labels, eval_labels.data, eval_pos_seqs)

    average_eval_loss = total_eval_loss / evaluation_dataloader.__len__()

    # Set the model back to train mode, which activates dropout again.
    model.train()
    return average_eval_loss.data[0], print_info(confusion_matrix, idx2pos)


def update_confusion_matrix(matrix, predictions, labels, pos_seqs):
    """
    update the confusion matrix based on the given batch

    :param matrix: a 3D numpy array of shape (#pos_tags, 2, 2)
    :param predictions: a numpy array of shape (batch_size, max_seq_len)
    :param labels: a numpy array of shape (batch_size, max_seq_len)
    :param lengths: a numpy array of shape (batch_size)
    :param pos_seqs: a list of variable-length indexed pos sequence
    :param idx2pos: a dictionary: int --> pos tag
    :return: the updated matrix
    """
    for i in range(len(pos_seqs)):
        indexed_pos_sequence = pos_seqs[i]
        prediction = predictions[i]
        label = labels[i]
        for j in range(len(indexed_pos_sequence)):
            indexed_pos = indexed_pos_sequence[j]
            p = prediction[j]
            l = label[j]
            matrix[indexed_pos][p][l] += 1
    return matrix


def get_batch_predictions(predictions, pos_seqs):
    """

    :param predictions: a numpy array of shape (batch_size, max_seq_len)
    :param pos_seqs: a list of variable-length indexed pos sequence
    :return: a list of variable-length predictions. each inner list is prediction for a sentence
    """
    pred_lst = []
    for i in range(len(pos_seqs)):  # each example i.e. each row
        indexed_pos_sequence = pos_seqs[i]
        prediction_padded = predictions[i]
        cur_pred_lst = []
        for j in range(len(indexed_pos_sequence)):  # inside each example: up to sentence length
            cur_pred_lst.append(prediction_padded[j])
        pred_lst.append(cur_pred_lst)
    return pred_lst


def write_predictions(raw_dataset, evaluation_dataloader, model, using_GPU, rawdata_filename):
    """
    Evaluate the model on the given evaluation_dataloader

    :param raw_dataset
    :param evaluation_dataloader:
    :param model:
    :param using_GPU: a boolean
    :return: a list of
    """
    # Set model to eval mode, which turns off dropout.
    model.eval()

    predictions = []
    for (eval_pos_seqs, eval_text, eval_lengths, eval_labels) in evaluation_dataloader:
        eval_text = Variable(eval_text, volatile=True)
        eval_lengths = Variable(eval_lengths, volatile=True)
        eval_labels = Variable(eval_labels, volatile=True)
        if using_GPU:
            eval_text = eval_text.cuda()
            eval_lengths = eval_lengths.cuda()
            eval_labels = eval_labels.cuda()

        # predicted shape: (batch_size, seq_len, 2)
        predicted = model(eval_text, eval_lengths)
        # get 0 or 1 predictions
        # predicted_labels: (batch_size, seq_len)
        _, predicted_labels = torch.max(predicted.data, 2)
        predictions.extend(get_batch_predictions(predicted_labels, eval_pos_seqs))

    # Set the model back to train mode, which activates dropout again.
    model.train()
    assert (len(predictions) == len(raw_dataset))

    # read original data
    data = []
    with open(rawdata_filename, encoding='latin-1') as f:
        lines = csv.reader(f)
        for line in lines:
            data.append(line)

    # append predictions to the original data
    data[0].append('prediction')
    for i in range(len(predictions)):
        data[i + 1].append(predictions[i])
    return data


def print_info(matrix, idx2pos):
    """
    Prints the precision, recall, f1, and accuracy for each pos tag
    Assume that the confusion matrix is implicitly mapped with the idx2pos
    i.e. row 0 in confusion matrix is for the pos tag mapped by int 0 in idx2pos

    :param matrix: a confusion matrix of shape (#pos_tags, 2, 2)
    :param idx2pos: idx2pos: a dictionary: int --> pos tag
    :return: a matrix (#allpostags, 4) each row is the PRFA performance for a pos tag
    """
    result = []
    for idx in range(len(idx2pos)):
        pos_tag = idx2pos[idx]
        grid = matrix[idx]
        precision = 100 * grid[1, 1] / np.sum(grid[1])
        recall = 100 * grid[1, 1] / np.sum(grid[:, 1])
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = 100 * (grid[1, 1] + grid[0, 0]) / np.sum(grid)
        print('PRFA performance for ', pos_tag, precision, recall, f1, accuracy)
        result.append([precision, recall, f1, accuracy])
    return np.array(result)


def get_performance_VUAverb_val(write=False):
    """
    Prints the performance of LSTM sequence model on VUA-verb validation set
    :param: write: a boolean to indicate write or not
    :return:
    """
    # get the VUA-ver validation set
    ID_verbidx_label = []  # ID tuple, verb_idx, label 1 or 0
    with open('../data/VUA/VUA_formatted_val.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            ID_verbidx_label.append([(line[0], line[1]), int(line[4]), int(line[5])])

    # get the prediction from LSTM sequence model
    ID2sen_labelseq = {}  # ID tuple --> [label_sequence]
    with open('../predictions/vua_seq_predictions_LSTMsequence_vua.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            ID2sen_labelseq[(line[0], line[1])] = ast.literal_eval(line[-1])
    # compute confusion_matrix
    predictions = []
    confusion_matrix = np.zeros((2, 2))
    for ID, verbidx, label in ID_verbidx_label:
        pred = ID2sen_labelseq[ID][verbidx]
        predictions.append(pred)
        confusion_matrix[pred][label] += 1
    assert (np.sum(confusion_matrix) == len(ID_verbidx_label))
    precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
    recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = 100 * (confusion_matrix[1, 1] + confusion_matrix[0, 0]) / np.sum(confusion_matrix)
    print('The performance of LSTM sequence model on VUA-verb validation set: ')
    print('Precision, Recall, F1, Accuracy: ', precision, recall, f1, accuracy)

    if write:
        data = []
        with open('../data/VUA/VUA_formatted_val.csv', encoding='latin-1') as f:
            lines = csv.reader(f)
            for line in lines:
                data.append(line)

        # append predictions to the original data
        data[0].append('prediction(by sequence model)')
        for i in range(len(predictions)):
            data[i + 1].append(predictions[i])

        f = open('../predictions/vua_predictions_LSTMsequence_vua.csv', 'w')
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()
        print('Writing the prediction on VUA-verb validation set by sequence model is done.')
    return [precision, recall, f1, accuracy]



def get_performance_VUAverb_test():
    """
    Similar treatment as get_performance_VUAverb_val
    Read the VUA-verb test data, and the VUA-sequence test data.
    Extract the predictions for VUA-verb test data from the VUA-sequence test data.
    Prints the performance of LSTM sequence model on VUA-verb test set based on genre
    Prints the performance of LSTM sequence model on VUA-verb test set regardless of genre

    :return: the averaged performance across genre
    """
    # get the VUA-ver test set
    ID_verbidx_label = []  # ID tuple, verb_idx, label 1 or 0
    with open('../data/VUA/VUA_formatted_test.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            ID_verbidx_label.append([(line[0], line[1]), int(line[4]), int(line[5])])

    # get the prediction from LSTM sequence model
    ID2sen_labelseq = {}  # ID tuple --> [genre, label_sequence]
    with open('../predictions/vua_seq_test_predictions_LSTMsequence_vua.csv', encoding='latin-1') as f:
        # txt_id	sen_ix	sentence	label_seq	pos_seq	labeled_sentence	genre   predictions
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            ID2sen_labelseq[(line[0], line[1])] = [line[6], ast.literal_eval(line[7])]
    # compute confusion_matrix
    predictions = []
    genres = ['news', 'fiction', 'academic', 'conversation']
    confusion_matrix = np.zeros((4, 2, 2))
    for ID, verbidx, label in ID_verbidx_label:
        pred = ID2sen_labelseq[ID][1][verbidx]
        predictions.append(pred)
        genre = ID2sen_labelseq[ID][0]
        genre_idx = genres.index(genre)
        confusion_matrix[genre_idx][pred][label] += 1
    assert (np.sum(confusion_matrix) == len(ID_verbidx_label))

    print('Tagging model performance on test-verb: genre')
    avg_performance = []
    for i in range(len(genres)):
        precision = 100 * confusion_matrix[i, 1, 1] / np.sum(confusion_matrix[i, 1])
        recall = 100 * confusion_matrix[i, 1, 1] / np.sum(confusion_matrix[i, :, 1])
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = 100 * (confusion_matrix[i, 1, 1] + confusion_matrix[i, 0, 0]) / np.sum(confusion_matrix[i])
        print(genres[i], 'Precision, Recall, F1, Accuracy: ', precision, recall, f1, accuracy)
        avg_performance.append([precision, recall, f1, accuracy])
    avg_performance = np.array(avg_performance)

    print('Tagging model performance on test-verb: regardless of genre')
    confusion_matrix = confusion_matrix.sum(axis=0)
    precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
    recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = 100 * (confusion_matrix[1, 1] + confusion_matrix[0, 0]) / np.sum(confusion_matrix)
    print('Precision, Recall, F1, Accuracy: ', precision, recall, f1, accuracy)

    return avg_performance.mean(0)


def get_performance_VUA_test():
    """
    Read the VUA-sequence test data and predictions
    Prints the performance of LSTM sequence model on VUA-sequence test set based on genre
    Prints the performance of LSTM sequence model on VUA-sequence test set regardless of genre

    :return: the averaged performance across genre
    """
    # get the prediction from LSTM sequence model
    ID2sen_labelseq = {}  # ID tuple --> [genre, label_sequence, pred_sequence]
    with open('../predictions/vua_seq_test_predictions_LSTMsequence_vua.csv', encoding='latin-1') as f:
        # txt_id	sen_ix	sentence	label_seq	pos_seq	labeled_sentence	genre   predictions
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            ID2sen_labelseq[(line[0], line[1])] = [line[6], ast.literal_eval(line[3]), ast.literal_eval(line[7])]
    # compute confusion_matrix
    genres = ['news', 'fiction', 'academic', 'conversation']
    confusion_matrix = np.zeros((4, 2, 2))
    for ID in ID2sen_labelseq:
        genre, label_sequence, pred_sequence = ID2sen_labelseq[ID]
        for i in range(len(label_sequence)):
            pred = pred_sequence[i]
            label = label_sequence[i]
            genre_idx = genres.index(genre)
            confusion_matrix[genre_idx][pred][label] += 1

    print('Tagging model performance on test-sequence: genre')
    avg_performance = []
    for i in range(len(genres)):
        precision = 100 * confusion_matrix[i, 1, 1] / np.sum(confusion_matrix[i, 1])
        recall = 100 * confusion_matrix[i, 1, 1] / np.sum(confusion_matrix[i, :, 1])
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = 100 * (confusion_matrix[i, 1, 1] + confusion_matrix[i, 0, 0]) / np.sum(confusion_matrix[i])
        print(genres[i], 'Precision, Recall, F1, Accuracy: ', precision, recall, f1, accuracy)
        avg_performance.append([precision, recall, f1, accuracy])
    avg_performance = np.array(avg_performance)

    print('Tagging model performance on test-sequence: regardless of genre')
    confusion_matrix = confusion_matrix.sum(axis=0)
    precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
    recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = 100 * (confusion_matrix[1, 1] + confusion_matrix[0, 0]) / np.sum(confusion_matrix)
    print('Precision, Recall, F1, Accuracy: ', precision, recall, f1, accuracy)

    return avg_performance.mean(0)


# Make sure to subclass torch.utils.data.Dataset
class TextDatasetWithGloveElmoSuffix(Dataset):
    def __init__(self, embedded_text, pos_seqs, labels):
        """

        :param embedded_text:
        :param pos_seqs:  a list of list: each inner list is a sequence of indexed pos tags
        :param labels: a list of list: each inner list is a sequence of 0, 1.
        :param max_sequence_length: an int
        """
        if len(embedded_text) != len(labels):
            raise ValueError("Differing number of sentences and labels!")
        # A list of numpy arrays, where each inner numpy arrays is sequence_length * embed_dim
        # embedding for each word is : glove + elmo + suffix
        self.embedded_text = embedded_text
        # A list of indexed pos sequences
        # where each inner list is the indexed pos tags for the sentence at the corresponding index.
        self.pos_seqs = pos_seqs
        #  a list of list: each inner list is a sequence of 0, 1.
        # where each inner list is the label for the sentence at the corresponding index.
        self.labels = labels
        # Truncate examples that are longer than max_sequence_length.
        # Long sequences are expensive and might blow up GPU memory usage.

    def __getitem__(self, idx):
        """
        Return the Dataset example at index `idx`.

        Returns
        -------
        example_pos_seq:
            a list of indexed pos tag sequence
        example_text: numpy array
        length: int
            The length of the (possibly truncated) example_text.
        example_label_seq: a list of 0 or 1
            The label of the example.
        """
        example_pos_seq = self.pos_seqs[idx]
        example_text = self.embedded_text[idx]
        example_label_seq = self.labels[idx]
        # Truncate the sequence if necessary
        example_length = example_text.shape[0]
        assert (example_length == len(example_pos_seq))
        assert (example_length == len(example_label_seq))
        return example_pos_seq, example_text, example_length, example_label_seq

    def __len__(self):
        """
        Return the number of examples in the Dataset.
        """
        return len(self.labels)

    @staticmethod
    def collate_fn(batch):
        """
        Given a list of examples (each from __getitem__),
        combine them to form a single batch by padding.

        Returns:
        -------
        batch_pos_seqs: list
          A list of list: each inner list is a variable-length list of indexed pos tags
        batch_padded_example_text: LongTensor
          LongTensor of shape (batch_size, longest_sequence_length) with the
          padded text for each example in the batch.
        length: LongTensor
          LongTensor of shape (batch_size,) with the unpadded length of the example.
        example_label: LongTensor
          LongTensor of shape (batch_size,) with the label of the example.
        """
        batch_padded_example_text = []
        batch_lengths = []
        batch_padded_labels = []
        batch_pos_seqs = []

        # Get the length of the longest sequence in the batch
        max_length = -1
        for pos, __, __, __ in batch:
            if len(pos) > max_length:
                max_length = len(pos)

        # Iterate over each example in the batch
        for pos, text, length, label in batch:
            # Unpack the example (returned from __getitem__)
            # append the pos_sequence to the batch_pos_seqs
            batch_pos_seqs.append(pos)

            # Amount to pad is length of longest example - length of this example.
            amount_to_pad = max_length - length
            # Tensor of shape (amount_to_pad,), converted to LongTensor
            pad_tensor = torch.zeros(amount_to_pad, text.shape[1])

            # Append the pad_tensor to the example_text tensor.
            # Shape of padded_example_text: (padded_length, embeding_dim)
            # top part is the original text numpy,
            # and the bottom part is the 0 padded tensors

            # text from the batch is a np array, but cat requires the argument to be the same type
            # turn the text into a torch.FloatTenser, which is the same type as pad_tensor
            text = torch.Tensor(text)
            padded_example_text = torch.cat((text, pad_tensor), dim=0)

            # pad the labels with zero.
            padded_example_label = label + [0] * amount_to_pad

            # Add the padded example to our batch
            batch_padded_example_text.append(padded_example_text)
            batch_lengths.append(length)
            batch_padded_labels.append(padded_example_label)

        # Stack the list of LongTensors into a single LongTensor
        return (batch_pos_seqs,
                torch.stack(batch_padded_example_text),
                torch.LongTensor(batch_lengths),
                torch.LongTensor(batch_padded_labels))
