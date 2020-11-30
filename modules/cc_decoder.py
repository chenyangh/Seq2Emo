import torch
import torch.nn as nn
from modules.luong_attention import Attention


class CCDecoder(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, emb_dim, hidden_size, num_class, batch_first=True,
                 dropout=0.2, args=None):
        """Initialize params."""
        super(CCDecoder, self).__init__()
        self.args = args
        self.num_class = num_class
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        self.input_feed = self.args.input_feeding
        self.concat_hidden = self.args.concat_hidden
        # forward decoder parameters
        if self.args.input_feeding:
            lstm_input_size = emb_dim + hidden_size
        else:
            lstm_input_size = emb_dim

        self.attention_layer = Attention(hidden_size, args=args)
        self.decoder_lstm = nn.LSTM(lstm_input_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout)

        self.pad_len = num_class + 3
        self.sos_idx = num_class
        self.eos_idx = num_class + 1
        self.pad_idx = num_class + 2

        self.embedding = nn.Embedding(self.pad_len, emb_dim)
        self.hidden2label = nn.Linear(hidden_size, self.pad_len)  # plus one for </s> and <pad>

    def forward(self, trg_emb, hidden, ctx, src_len):
        def recurrence(_trg_emb_i, _hidden, _h_tilde, _decoder_lstm, _attention_layer):
            if self.input_feed:
                _lstm_input = torch.cat((_trg_emb_i, _h_tilde.squeeze(0)), dim=1)
            else:
                _lstm_input = _trg_emb_i
            lstm_out, _hidden = _decoder_lstm(_lstm_input.unsqueeze(1), _hidden)
            _h_tilde, _ = _attention_layer(lstm_out, ctx, src_len.view(-1))

            return _h_tilde.squeeze(0), _hidden  # squeeze out the trg_len dimension

        max_step = trg_emb.size()[1]
        src_len = src_len.view(-1)

        init_hidden = hidden

        # Note forward
        hs_list = []
        h_tilde = init_hidden[0]
        hidden = init_hidden
        if len(hidden[0].size()) == 2:
            hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
        for i in range(max_step):
            emb_t = trg_emb[:, i, :]
            h_tilde, hidden = recurrence(emb_t, hidden, h_tilde, self.decoder_lstm, self.attention_layer)
            if not self.concat_hidden:
                hs_list.append(h_tilde)
            else:
                hs_list.append(torch.cat((emb_t, h_tilde), dim=1))

        output = torch.stack(hs_list, dim=0).transpose(0, 1)

        return output, hidden

