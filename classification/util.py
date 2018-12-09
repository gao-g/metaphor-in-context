from tqdm import tqdm
import torch
import numpy as np
import mmap
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
                a index: int: idx of the focus verb
                a label: int 1 or 0
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


def embed_sequence(sequence, verb_idx, word2idx, glove_embeddings, elmo_embeddings, suffix_embeddings):
    """
    Assume that word2idx has 1 mapped to UNK
    Assume that word2idx maps well implicitly with glove_embeddings
    i.e. the idx for each word is the row number for its corresponding embedding

    :param sequence: a single string: a sentence with space
    :param word2idx: a dictionary: string --> int
    :param glove_embeddings: a nn.Embedding with padding idx 0
    :param elmo_embeddings: a h5py file
                    each group_key is a string: a sentence
                    each inside group is an np array (seq_len, 1024 elmo)
    :param suffix_embeddings: a nn.Embedding without padding idx
    :return: a np.array (seq_len, embed_dim=glove+elmo+suffix)
    """
    words = sequence.split()

    # 1. embed the sequence by glove vector
    # Replace words with tokens, and 1 (UNK index) if words not indexed.
    indexed_sequence = [word2idx.get(x, 1) for x in words]
    # glove_part has shape: (seq_len, glove_dim)
    glove_part = glove_embeddings(Variable(torch.LongTensor(indexed_sequence)))

    # 2. embed the sequence by elmo vectors
    if elmo_embeddings != None:
        elmo_part = elmo_embeddings[sequence]
        assert (elmo_part.shape == (len(words), 1024))

    # 3. embed the sequence by suffix indicators i.e. wether it is a verb or not
    indicated_sequence = [0] * len(words)
    indicated_sequence[verb_idx] = 1
    suffix_part = suffix_embeddings(Variable(torch.LongTensor(indicated_sequence)))

    # concatenate three parts: glove+elmo+suffix along axis 1
    assert(glove_part.shape == (len(words), 300))
    assert(suffix_part.shape == (len(words), 50))
    # glove_part and suffix_part are Variables, so we need to use .data
    # otherwise, throws weird ValueError: incorrect dimension, zero-dimension, etc..
    if elmo_embeddings != None:
        result = np.concatenate((glove_part.data, elmo_part), axis=1)
        result = np.concatenate((result, suffix_part.data), axis=1)
    else:
        result = np.concatenate((glove_part.data, suffix_part.data), axis=1)
    return result


def evaluate(evaluation_dataloader, model, criterion, using_GPU):
    """
    Evaluate the model on the given evaluation_dataloader

    :param evaluation_dataloader:
    :param model:
    :param criterion: loss criterion
    :param using_GPU: a boolean
    :return:
    """
    # Set model to eval mode, which turns off dropout.
    model.eval()

    num_correct = 0
    total_examples = 0
    total_eval_loss = 0
    confusion_matrix = np.zeros((2, 2))
    for (eval_text, eval_lengths, eval_labels) in evaluation_dataloader:
        eval_text = Variable(eval_text, volatile=True)
        eval_lengths = Variable(eval_lengths, volatile=True)
        eval_labels = Variable(eval_labels, volatile=True)
        if using_GPU:
            eval_text = eval_text.cuda()
            eval_lengths = eval_lengths.cuda()
            eval_labels = eval_labels.cuda()

        predicted = model(eval_text, eval_lengths)
        # Calculate loss for this test batch. This is averaged, so multiply
        # by the number of examples in batch to get a total.
        total_eval_loss += criterion(predicted, eval_labels).data[0] * eval_labels.size(0)
        _, predicted_labels = torch.max(predicted.data, 1)
        total_examples += eval_labels.size(0)
        num_correct += torch.sum(predicted_labels == eval_labels.data)
        for i in range(eval_labels.size(0)):
            confusion_matrix[int(predicted_labels[i]), eval_labels.data[i]] += 1

    accuracy = 100 * num_correct / total_examples
    average_eval_loss = total_eval_loss / total_examples

    precision = 100 * confusion_matrix[0, 0] / np.sum(confusion_matrix[0])
    recall = 100 * confusion_matrix[0, 0] / np.sum(confusion_matrix[:, 0])
    lit_f1 = 2 * precision * recall / (precision + recall)

    precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
    recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
    met_f1 = 2 * precision * recall / (precision + recall)
    class_wise_f1 = (met_f1 + lit_f1) / 2

    # Set the model back to train mode, which activates dropout again.
    model.train()
    print(confusion_matrix)
    return average_eval_loss, accuracy, precision, recall, met_f1, class_wise_f1

# Make sure to subclass torch.utils.data.Dataset
class TextDatasetWithGloveElmoSuffix(Dataset):
    def __init__(self, embedded_text, labels, max_sequence_length=100):
        """

        :param embedded_text:
        :param labels: a list of ints
        :param max_sequence_length: an int
        """
        if len(embedded_text) != len(labels):
            raise ValueError("Differing number of sentences and labels!")
        # A list of numpy arrays, where each inner numpy arrays is sequence_length * embed_dim
        # embedding for each word is : glove + elmo + suffix
        self.embedded_text = embedded_text
        # A list of ints, where each int is a label of the sentence at the corresponding index.
        self.labels = labels
        # Truncate examples that are longer than max_sequence_length.
        # Long sequences are expensive and might blow up GPU memory usage.
        self.max_sequence_length = max_sequence_length


    def __getitem__(self, idx):
        """
        Return the Dataset example at index `idx`.

        Returns
        -------
        example_text: numpy array
        length: int
            The length of the (possibly truncated) example_text.
        example_label: int 0 or 1
            The label of the example.
        """
        example_text = self.embedded_text[idx]
        example_label = self.labels[idx]
        # Truncate the sequence if necessary
        example_text = example_text[:self.max_sequence_length]
        example_length = example_text.shape[0]

        return example_text, example_length, example_label

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
        batch_labels = []

        # Get the length of the longest sequence in the batch
        max_length = max(batch, key=lambda example: example[1])[1]

        # Iterate over each example in the batch
        for text, length, label in batch:
            # Unpack the example (returned from __getitem__)

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

            # Add the padded example to our batch
            batch_padded_example_text.append(padded_example_text)
            batch_lengths.append(length)
            batch_labels.append(label)

        # Stack the list of LongTensors into a single LongTensor
        return (torch.stack(batch_padded_example_text),
                torch.LongTensor(batch_lengths),
                torch.LongTensor(batch_labels))
