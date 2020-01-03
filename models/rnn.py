import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class RNN(nn.Module):
    """ vanilla-RNN (text) network implementation """

    def __init__(self,
                 n_classes=10,
                 vocabulary_size=26,
                 hidden_dim=256,
                 device="cpu"):
        super(RNN, self).__init__()

        self.lstm_num_hidden = hidden_dim
        self.num_classes = n_classes
        self.device = device
        self.embedding = nn.Embedding(vocabulary_size, hidden_dim)
        self.model = nn.RNN(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.output_layer_toClass = nn.Linear(hidden_dim, n_classes, bias=False)

    def forward(self, x, lengths, **kwargs):
        # get embedding
        x_embedded = self.embedding(x.long())

        # to mask padding as ignorable
        x_packed = pack_padded_sequence(x_embedded, lengths.long(), batch_first=True, enforce_sorted=False)

        # forward through network
        output, _ = self.model(x_packed.float())

        # to mask padding as ignorable, recover
        output, _ = pad_packed_sequence(output, batch_first=True)

        # predict class
        output = self.output_layer_toClass(output)
        return output.sum(dim=1)


if __name__ == '__main__':

    """ run to test, output should be shaped [5 x 10] = (batch x classes) """

    batch_size = 5
    embedding_dim = 10
    min_line_len = 3
    max_line_len = 50
    vocabulary_size = 26
    # simulate lines of different length
    line_lengths = torch.tensor([random.choice(list(range(min_line_len, max_line_len))) for _ in range(batch_size)])
    max_length = max(line_lengths)

    # simulate zero-padding
    lines = torch.zeros((batch_size, max_length))
    for j, line in enumerate(lines):
        # sample random letters in lines whose index is not supposed to be padding
        desired_length = line_lengths[j]
        line[:desired_length] = torch.tensor(
            [random.choice(list(range(vocabulary_size))) for _ in range(desired_length.item())])

    model = RNN(n_classes=10, hidden_dim=embedding_dim, vocabulary_size=vocabulary_size)
    output = model.forward(lines, line_lengths)
    print(output.shape)
