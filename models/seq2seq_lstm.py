import torch
import torch.nn as nn
from modules.lstm_encoder import LSTMEncoder
from modules.seq2seq_decoder import Seq2SeqDecoder


class LSTMSeq2Seq(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
        self,
        emb_dim,
        vocab_size,
        trg_vocab_size,
        src_hidden_dim,
        trg_hidden_dim,
        attention_mode,
        batch_size,
        nlayers=2,
        nlayers_trg=2,
        dropout=0.2,
        encoder_dropout=0.2,
        decoder_dropout=0.2,
        attention_dropout=0.2,
        args=None

    ):
        """Initialize model."""
        super(LSTMSeq2Seq, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.attention_mode = attention_mode
        self.bidirectional = True
        self.trg_vocab_size = trg_vocab_size

        self.encoder = LSTMEncoder(emb_dim, src_hidden_dim, vocab_size, encoder_dropout=encoder_dropout)

        self.decoder = Seq2SeqDecoder(
            emb_dim,
            trg_hidden_dim,
            self.trg_vocab_size,
            batch_first=True,
            dropout=decoder_dropout,
            args=self.args
        )
        self.encoder2decoder_scr_hm = nn.Linear(src_hidden_dim * 2, trg_hidden_dim, bias=False)
        self.encoder2decoder_ctx = nn.Linear(src_hidden_dim * 2, trg_hidden_dim, bias=False)

    def load_encoder_embedding(self, emb, fix_emb=False):
        self.encoder.embeddings.weight = nn.Parameter(torch.FloatTensor(emb))
        if fix_emb:
            self.encoder.embeddings.weight.requires_grad = False

    def load_emotion_embedding(self, emb):
        if self.args.load_emo_emb:
            self.decoder.forward_signal_embedding.weight = nn.Parameter(torch.FloatTensor(emb))
        if self.args.fix_emo_emb:
            self.decoder.forward_signal_embedding.weight.requires_grad = False

    def forward(self, src, src_len, src_elmo, moji_id=None, moji_len=None):
        """Propogate input through the network."""
        # trg_emb = self.embedding(trg)

        src_h, (_, _) = self.encoder(src, src_len, src_elmo)
        cur_batch_size = src_h.size()[0]
        # src_h_m = src_h_m.view(self.encoder.num_layers, 2, cur_batch_size, self.src_hidden_dim)[-1]
        # src_h_m = torch.cat((src_h_m[0], src_h_m[1]), dim=1)
        src_h_m = torch.cat([src_h[idx][one_len - 1] for idx, one_len in enumerate(src_len)], dim=0)

        decoder_h_0 = torch.tanh(self.encoder2decoder_scr_hm(src_h_m))  # torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        decoder_c_0 = torch.zeros((cur_batch_size, self.decoder.hidden_size)).cuda()

        ctx = self.encoder2decoder_ctx(src_h)

        decoder_logit= self.decoder(
            (decoder_h_0, decoder_c_0),
            ctx,
            src_len
        )

        return decoder_logit

