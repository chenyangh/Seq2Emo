import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

import pickle as pkl
import os
from tqdm import tqdm

NUM_EMO = 4


class LSTMEncoder(nn.Module):
    """
    A Hierarchical LSTM with for 3 turns dialogue
    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encoder_dropout=0):
        super(LSTMEncoder, self).__init__()
        self.SENT_LSTM_DIM = hidden_dim
        self.bidirectional = True

        self.sent_lstm_directions = 2 if self.bidirectional else 1

        self.elmo_dim = 1024

        self.num_layers = 2

        self.a_lstm = nn.LSTM(embedding_dim + self.elmo_dim, hidden_dim, num_layers=self.num_layers, batch_first=True,
                              bidirectional=self.bidirectional, dropout=encoder_dropout)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.drop_out = nn.Dropout(encoder_dropout)

    def init_hidden(self, x):
        batch_size = x.size(0)
        if self.bidirectional:
            h0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.SENT_LSTM_DIM), requires_grad=False).cuda()
            c0 = Variable(torch.zeros(2*self.num_layers, batch_size, self.SENT_LSTM_DIM), requires_grad=False).cuda()
        else:
            h0 = Variable(torch.zeros(1*self.num_layers, batch_size, self.SENT_LSTM_DIM), requires_grad=False).cuda()
            c0 = Variable(torch.zeros(1*self.num_layers, batch_size, self.SENT_LSTM_DIM), requires_grad=False).cuda()
        return (h0, c0)

    @staticmethod
    def sort_batch(batch, lengths):
        seq_lengths, perm_idx = lengths.sort(0, descending=True)
        seq_tensor = batch[perm_idx]
        rever_sort = np.zeros(len(seq_lengths))
        for i, l in enumerate(perm_idx):
            rever_sort[l] = i
        return seq_tensor, seq_lengths, rever_sort.astype(int), perm_idx

    def lstm_forward(self, x, x_len, elmo_x, lstm, hidden=None):
        x, x_len_sorted, reverse_idx, perm_idx = self.sort_batch(x, x_len.view(-1))
        max_len = int(x_len_sorted[0])

        emb_x = self.embeddings(x)
        emb_x = self.drop_out(emb_x)
        emb_x = emb_x[:, :max_len, :]

        elmo_x = elmo_x[perm_idx]
        elmo_x = self.drop_out(elmo_x)
        emb_x = torch.cat((emb_x, elmo_x), dim=2)

        packed_input = nn.utils.rnn.pack_padded_sequence(emb_x, x_len_sorted.cpu().numpy(), batch_first=True)
        if hidden is None:
            hidden = self.init_hidden(x)
        packed_output, hidden = lstm(packed_input, hidden)
        output, unpacked_len = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        return output[reverse_idx], (hidden[0][:, reverse_idx, :], hidden[1][:, reverse_idx, :])

    def forward(self, a, a_len, elmo_a):
        # Sentence LSTM A
        a_out, a_hidden = self.lstm_forward(a, a_len, elmo_a, self.a_lstm)

        return a_out, a_hidden

    def load_embedding(self, emb):
        self.embeddings.weight = nn.Parameter(torch.FloatTensor(emb))
