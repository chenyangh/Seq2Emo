import torch
import torch.nn as nn
from modules.luong_attention import Attention


class Seq2SeqDecoder(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, emb_dim, hidden_size, num_class, batch_first=True,
                 dropout=0.2, args=None):
        """Initialize params."""
        super(Seq2SeqDecoder, self).__init__()
        self.args = args
        self.num_class = num_class
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        self.input_feed = self.args.input_feeding
        self.concat_signal = self.args.concat_signal
        # forward decoder parameters
        if self.args.input_feeding:
            lstm_input_size = emb_dim + hidden_size
        else:
            lstm_input_size = emb_dim

        self.forward_signal_embedding = nn.Embedding(num_class, emb_dim)
        self.forward_attention_layer = Attention(hidden_size, args=args)
        self.forward_decoder_lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout)
        self.backward_signal_embedding = nn.Embedding(num_class, emb_dim)
        self.backward_signal_embedding = self.forward_signal_embedding
        # backward decoder parameters
        if self.args.single_direction is False:
            if not self.args.unify_decoder:
                self.backward_attention_layer = Attention(hidden_size, args=args)
                self.backward_decoder_lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout)
            else:
                self.backward_attention_layer = self.forward_attention_layer
                self.backward_decoder_lstm = self.forward_decoder_lstm

        # input feeding option

        # self.decoder2emo_W = nn.Embedding(num_class, hidden_size * 2 * 2)
        # self.decoder2emo_bias = nn.Embedding(num_class, 2)
            if not self.concat_signal:
                self.binary_hidden2label_list = nn.ModuleList([nn.Linear(hidden_size*2, 2) for _ in range(num_class)])
            else:
                self.binary_hidden2label_list = nn.ModuleList([nn.Linear(hidden_size * 2 + emb_dim, 2) for _ in range(num_class)])
        else:
            if not self.concat_signal:
                self.binary_hidden2label_list = nn.ModuleList([nn.Linear(hidden_size, 2) for _ in range(num_class)])
            else:
                self.binary_hidden2label_list = nn.ModuleList(
                    [nn.Linear(hidden_size + emb_dim, 2) for _ in range(num_class)])

    def forward(self, hidden, ctx, src_len):
        def recurrence(_trg_emb_i, _hidden, _h_tilde, _decoder_lstm, _attention_layer):
            if self.input_feed:
                if len(_h_tilde.size()) > 2:
                    _h_tilde = _h_tilde.squeeze(0)
                _lstm_input = torch.cat((_trg_emb_i, _h_tilde), dim=1)
            else:
                _lstm_input = _trg_emb_i
            lstm_out, _hidden = _decoder_lstm(_lstm_input.unsqueeze(1), _hidden)
            _h_tilde, alpha = _attention_layer(lstm_out, ctx, src_len.view(-1))

            return _h_tilde.squeeze(0), _hidden  # squeeze out the trg_len dimension

        b_size = src_len.size(0)
        src_len = src_len.view(-1)
        # hidden_copy = (hidden[0].clone(), hidden[1].clone())

        init_hidden = hidden

        # Note forward
        hs_forward = []
        h_tilde = init_hidden[0]
        hidden = init_hidden
        if len(hidden[0].size()) == 2:
            hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
        for i in range(self.num_class):
            emo_signal = torch.LongTensor([i] * b_size).cuda()
            emo_signal_input = self.forward_signal_embedding(emo_signal)
            emo_signal_input = self.dropout(emo_signal_input)
            h_tilde, hidden = recurrence(emo_signal_input, hidden, h_tilde, self.forward_decoder_lstm,
                                         self.forward_attention_layer)
            if not self.concat_signal:
                hs_forward.append(h_tilde)
            else:
                hs_forward.append(torch.cat((emo_signal_input, h_tilde), dim=1))

        if self.args.single_direction is False:
            # Note backward
            hs_backward = []
            h_tilde = init_hidden[0]
            hidden = init_hidden
            if len(hidden[0].size()) == 2:
                hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
            for i in range(self.num_class - 1, -1, -1):
                emo_signal = torch.LongTensor([i] * b_size).cuda()
                emo_signal_input = self.backward_signal_embedding(emo_signal)
                emo_signal_input = self.dropout(emo_signal_input)
                h_tilde, hidden = recurrence(emo_signal_input, hidden, h_tilde, self.backward_decoder_lstm, self.backward_attention_layer)
                hs_backward.append(h_tilde)

            decoder_output = []
            h_list = []
            for i in range(self.num_class):
                h_bidirection = torch.cat((hs_forward[i], hs_backward[self.num_class - i - 1]), dim=1)
                # h_bidirection = self.dropout(h_bidirection)
                # h_bidirection = torch.relu(h_bidirection)
                h_list.append(h_bidirection)

                # emo_signal = torch.LongTensor([i] * b_size).cuda()
                # emo_out = torch.bmm(h_bidirection.unsqueeze(1),
                #                     self.decoder2emo_W(emo_signal).view(-1, self.hidden_size * 2, 2)).squeeze()
                # emo_out = torch.add(emo_out, self.decoder2emo_bias(emo_signal))
                # decoder_output.append(emo_out)

            pred_list = [self.binary_hidden2label_list[i](h_list[i]) for i in range(self.num_class)]
        else:
            pred_list = [self.binary_hidden2label_list[i](hs_forward[i]) for i in range(self.num_class)]

        return torch.stack(pred_list, dim=1)
