import torch.nn as nn
import torch
from modules.lstm_encoder import LSTMEncoder
from modules.cc_decoder import CCDecoder


class CCLSTMClassifier(nn.Module):
    def __init__(self, emb_dim, hidden_dim, vocab_size, num_label, args):
        super(CCLSTMClassifier, self).__init__()
        self.num_label = num_label
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.args = args

        # Encoder
        self.encoder = LSTMEncoder(emb_dim, hidden_dim, vocab_size, encoder_dropout=args.encoder_dropout)

        self.encoder2decoder_scr_hm = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.encoder2decoder_ctx = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

        self.decoder = CCDecoder(emb_dim, hidden_dim, num_label, args=args)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.decoder.pad_idx)


    def load_encoder_embedding(self, emb, fix_emb=False):
        self.encoder.embeddings.weight = nn.Parameter(torch.FloatTensor(emb))
        if fix_emb:
            self.encoder.embeddings.weight.requires_grad = False

    def binary_trg_to_seq(self, binary_trg):
        batch_size = len(binary_trg)
        idx_map = torch.arange(self.num_label).repeat(batch_size, 1)
        assert binary_trg.size() == idx_map.size()
        seq_list = [idx_map[idx][binary_trg[idx]==1] for idx in range(batch_size)]
        max_len = max([len(seq) for seq in seq_list])

        seq_list_pad = torch.stack([torch.cat((torch.tensor([self.decoder.sos_idx]),
                                seq,
                                torch.tensor([self.decoder.eos_idx]),
                                torch.tensor([self.decoder.pad_idx] * (max_len - len(seq)))))
                     for seq in seq_list], dim=0).to(dtype=torch.long).cuda()

        return seq_list_pad

    def seq_trg_to_binary(self, seq_trg, cur_batch_size):
        binary_pred = torch.zeros(cur_batch_size, self.num_label)
        for i, instance in enumerate(seq_trg):
            for label_idx in instance:
                if label_idx < self.num_label:
                    binary_pred[i][label_idx] = 1
        return binary_pred.cuda()

    def loss(self, x, seq_len, elmo, binary_trg):
        cur_batch_size = len(x)
        src_h, hidden = self.encoder(x, seq_len, elmo)
        src_h_m = torch.cat([src_h[idx][one_len - 1] for idx, one_len in enumerate(seq_len)], dim=0)

        decoder_h_0 = torch.tanh(self.encoder2decoder_scr_hm(src_h_m))  # torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        decoder_c_0 = torch.zeros((cur_batch_size, self.decoder.hidden_size)).cuda()

        ctx = self.encoder2decoder_ctx(src_h)

        seq_trg = self.binary_trg_to_seq(binary_trg)
        trg_emb = self.decoder.embedding(seq_trg)

        trg_h, (_, _) = self.decoder(
            trg_emb,
            (decoder_h_0, decoder_c_0),
            ctx,
            seq_len.view(-1)
        )

        decoder_logit = self.decoder.hidden2label(trg_h)
        decoder_logit = decoder_logit[:, :-1, :]
        seq_trg_shift = seq_trg[:, 1:]
        loss = self.loss_fn(
            decoder_logit.reshape(-1, self.decoder.pad_len),
            seq_trg_shift.reshape(-1).cuda()
        )
        return loss

    def greedy_decode_batch(self, x, seq_len, elmo):
        """Decode a minibatch."""
        cur_batch_size = len(x)
        src_h, hidden = self.encoder(x, seq_len, elmo)
        src_h_m = torch.cat([src_h[idx][one_len - 1] for idx, one_len in enumerate(seq_len)], dim=0)

        decoder_h_0 = torch.tanh(self.encoder2decoder_scr_hm(src_h_m))  # torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        decoder_c_0 = torch.zeros((cur_batch_size, self.decoder.hidden_size)).cuda()
        dec_states = (decoder_h_0, decoder_c_0)
        ctx = self.encoder2decoder_ctx(src_h)

        next_input = self.decoder.sos_idx
        next_input_tensor = torch.tensor([next_input] * cur_batch_size).cuda()

        batched_ouput = []
        for step in range(self.decoder.pad_len):
            trg_emb = self.decoder.embedding(next_input_tensor)

            trg_h, dec_states = self.decoder(
                trg_emb.unsqueeze(1),
                dec_states,
                ctx,
                seq_len.view(-1)
            )

            decoder_logit = self.decoder.hidden2label(trg_h)

            greedy_next = torch.argmax(torch.softmax(decoder_logit, dim=-1), dim=-1)
            next_input_tensor = torch.tensor(greedy_next).cuda().view(-1)
            batched_ouput.append(greedy_next)
        seq_pred = torch.stack(batched_ouput, dim=1).squeeze(2)
        binary_pred = self.seq_trg_to_binary(seq_pred, cur_batch_size)

        return binary_pred
