import torch.nn as nn
import torch.nn.functional as F
from allennlp.nn.util import sort_batch_by_length, last_dim_softmax
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch


class RNNSequenceClassifier(nn.Module):
    # num_classes: The number of classes in the classification problem.
    # embedding_dim: The input dimension
    # hidden_size: The size of the RNN hidden state.
    # num_layers: Number of layers to use in RNN
    # bidir: boolean of wether to use bidirectional or not in RNN
    # dropout1: dropout on input to RNN
    # dropout2: dropout in RNN
    # dropout3: dropout on hidden state of RNN to linear layer
    def __init__(self, num_classes, embedding_dim, hidden_size, num_layers, bidir=True,
                 dropout1=0.2, dropout2=0.2, dropout3=0.2):
        # Always call the superclass (nn.Module) constructor first
        super(RNNSequenceClassifier, self).__init__()

        """
        # Create Embedding object, which will handle efficiently
        # converting numerical word indices to embedding vectors.
        # We set padding_idx=0, so words with index 0 will get zero vectors.
        vocab_size = embedding_matrix.size(0)
        embedding_dim = embedding_matrix.size(1)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Load our embedding matrix weights into the Embedding object.
        # nn.Parameter is basically a Variable that is part of a Module.
        self.embedding.weight = nn.Parameter(embedding_matrix)
        """
        # Set up the RNN: use an LSTM here.
        # We set batch_first=True because our data is of shape (batch_size, seq_len, num_features)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout2, batch_first=True, bidirectional=bidir)

        # Affine transform for attention.
        direc = 2 if bidir else 1
        self.attention_weights = nn.Linear(hidden_size * direc, 1)
        # Set up the final transform to a distribution over classes.
        self.output_projection = nn.Linear(hidden_size * direc, num_classes)

        # Dropout layer
        self.dropout_on_input_to_LSTM = nn.Dropout(dropout1)
        self.dropout_on_input_to_linear_layer = nn.Dropout(dropout3)



    # forward takes an batch of inputs and a
    # List of int with the unpadded length of each example in the batch.
    # input is of shape (batch_size, sequence_length)
    def forward(self, inputs, lengths):
        # 1. run LSTM
        # apply dropout to the input
        # Shape of inputs: (batch_size, sequence_length, embedding_dim)
        embedded_input = self.dropout_on_input_to_LSTM(inputs)
        # Sort the embedded inputs by decreasing order of input length.
        # sorted_input shape: (batch_size, sequence_length, embedding_dim)
        (sorted_input, sorted_lengths, input_unsort_indices, _) = sort_batch_by_length(embedded_input, lengths)
        # Pack the sorted inputs with pack_padded_sequence.
        packed_input = pack_padded_sequence(sorted_input, sorted_lengths.data.tolist(), batch_first=True)
        # Run the input through the RNN.
        packed_sorted_output, _ = self.rnn(packed_input)
        # Unpack (pad) the input with pad_packed_sequence
        # Shape: (batch_size, sequence_length, hidden_size)
        sorted_output, _ = pad_packed_sequence(packed_sorted_output, batch_first=True)
        # Re-sort the packed sequence to restore the initial ordering
        # Shape: (batch_size, sequence_length, hidden_size)
        output = sorted_output[input_unsort_indices]

        # 2. use attention
        # Shape: (batch_size, sequence_length, 1)
        # Shape: (batch_size, sequence_length) after squeeze
        attention_logits = self.attention_weights(output).squeeze(dim=-1)
        mask_attention_logits = (attention_logits != 0).type(
            torch.cuda.FloatTensor if inputs.is_cuda else torch.FloatTensor)
        # Shape: (batch_size, sequence_length)
        softmax_attention_logits = last_dim_softmax(attention_logits, mask_attention_logits)
        # Shape: (batch_size, 1, sequence_length)
        softmax_attention_logits = softmax_attention_logits.unsqueeze(dim=1)
        # Shape of input_encoding: (batch_size, 1, hidden_size )
        #    output: (batch_size, sequence_length, hidden_size)
        #    softmax_attention_logits: (batch_size, 1, sequence_length)
        input_encoding = torch.bmm(softmax_attention_logits, output)
        # Shape: (batch_size, hidden_size)
        input_encoding = input_encoding.squeeze(dim=1)

        # 3. run linear layer
        # apply dropout to input to the linear layer
        input_encoding = self.dropout_on_input_to_linear_layer(input_encoding)
        # Run the RNN encoding of the input through the output projection
        # to get scores for each of the classes.
        unnormalized_output = self.output_projection(input_encoding)
        # Normalize with log softmax
        output_distribution = F.log_softmax(unnormalized_output, dim=-1)
        return output_distribution
