import torch.nn as nn
import torch
from modules.lstm_encoder import LSTMEncoder
from modules.self_attention import SelfAttention
from modules.binary_decoder import BinaryDecoder


class BinaryLSTMClassifier(nn.Module):
    def __init__(self, emb_dim, hidden_dim, vocab_size, num_label, attention_mode, args):
        super(BinaryLSTMClassifier, self).__init__()
        self.num_label = num_label
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.args = args

        # Encoder
        self.encoder = LSTMEncoder(emb_dim, hidden_dim, vocab_size, encoder_dropout=args.encoder_dropout)
        if self.encoder.bidirectional:
            hidden_dim = hidden_dim * 2

        # Init Attention
        if attention_mode == 'self':
            self.att = SelfAttention
        elif attention_mode == 'None':
            self.att = None
        if self.att is not None:
            self.attention_layer = self.att(hidden_dim)

        # Decoder
        self.decoder = BinaryDecoder(hidden_dim, num_label)

    def load_encoder_embedding(self, emb, fix_emb=False):
        self.encoder.embeddings.weight = nn.Parameter(torch.FloatTensor(emb))
        if fix_emb:
            self.encoder.embeddings.weight.requires_grad = False

    def forward(self, x, seq_len, elmo):
        out, hidden = self.encoder(x, seq_len, elmo)

        if self.att is not None:
            out, alpha = self.attention_layer(out, seq_len.view(-1))
        else:
            seq_len_expand = seq_len.view(-1, 1, 1).expand(out.size(0), 1, out.size(2)) - 1
            out = torch.gather(out, 1, seq_len_expand).squeeze(1)

        pred = self.decoder(out)
        return pred
