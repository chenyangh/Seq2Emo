import os
import pickle as pkl
from tqdm import tqdm
import numpy as np


class Tokenizer(object):
    """
    A abstract class that needs the implementation of the following functions:
    to_ids() : function that converts list of tokens/ a string to list of tokens
    to_tokens() : function that converts list of ids to list of tokens
    get_embedding() : get the initial embedding matrix

    """

    def __init__(self):
        pass

    def get_embeddings(self):
        pass

    def encode_ids(self, text):
        pass

    def decode_ids(self, ids):
        pass

    def tokenize(self, text):
        pass


class GloveTokenizer(Tokenizer):
    """
    A tokenizer that is able to read any pre-trained word embedding files that in the general .vec format
    i.e. each line is in the form: token[space]emb_0[space]emb_1[space]...emb_dim\n, where emb_i is the ith value of
    the R^{dim} embedding.
    """
    def __init__(self, pad_len=None):
        super(GloveTokenizer, self).__init__()
        # init variables
        self.word2id = {}
        self.id2word = {}
        self.embeddings = {}
        self.emb_dim = None
        self.vocab_size = None
        self.pad_len = pad_len

        # use nltk tokenizer

    def build_tokenizer(self,  sent_list, vocab_size=50000):
        # build vocab
        from utils.build_vocab import build_vocab
        word2id, id2word, vocab_size = build_vocab(sent_list, vocab_size, fill_vocab=False)
        self.word2id = word2id
        self.id2word = id2word
        self.vocab_size = vocab_size

    def get_emb_by_words(self, fname, words):
        def load_vectors(fname):
            print("Loading Glove Model")
            f = open(fname, 'r', encoding='utf8')
            model = {}
            for line in tqdm(f.readlines(), total=2196017):
                values = line.split(' ')
                word = values[0]
                try:
                    embedding = np.array(values[1:], dtype=np.float32)
                    model[word] = embedding
                except ValueError:
                    print(len(values), values[0])

            print("Done.", len(model), " words loaded!")
            f.close()
            return model

        pkl_path = fname + '.pkl'
        if not os.path.isfile(pkl_path):
            print('creating pkl file for the emb text file')
            emb_dict = load_vectors(fname)
            with open(pkl_path, 'wb') as f:
                pkl.dump(emb_dict, f)
        else:
            print('loading pkl file')
            with open(pkl_path, 'rb') as f:
                emb_dict = pkl.load(f)
            print('loading finished')

        emb_list = []
        for word in words:
            if word not in emb_dict:
                raise ValueError
            else:
                emb_list.append(emb_dict[word])
        return emb_list


    def build_embedding(self, fname, emb_dim=300, save_pkl=True, dataset_name=''):
        # get embedding
        def load_vectors(fname):
            print("Loading Glove Model")
            f = open(fname, 'r', encoding='utf8')
            model = {}
            for line in tqdm(f.readlines(), total=2196017):
                values = line.split(' ')
                word = values[0]
                try:
                    embedding = np.array(values[1:], dtype=np.float32)
                    model[word] = embedding
                except ValueError:
                    print(len(values), values[0])

            print("Done.", len(model), " words loaded!")
            f.close()
            return model

        def get_emb(emb_dict, vocab_size, embedding_dim):
            # emb_dict = load_vectors(fname)
            all_embs = np.stack(emb_dict.values())
            emb_mean, emb_std = all_embs.mean(), all_embs.std()

            emb = np.random.normal(emb_mean, emb_std, (vocab_size, embedding_dim))

            # emb = np.zeros((vocab_size, embedding_dim))
            num_found = 0
            print('loading glove')
            for idx in tqdm(range(vocab_size)):
                word = self.id2word[idx]
                if word == '<pad>' or word == '<unk>':
                    emb[idx] = np.zeros([embedding_dim])
                elif word in emb_dict:
                    emb[idx] = emb_dict[word]
                    num_found += 1
                # else:
                #     emb[idx] = np.random.uniform(-0.5 / embedding_dim, 0.5 / embedding_dim,
                #                                  embedding_dim)
            print('vocab size:', vocab_size)
            print('vocab coverage:', num_found/vocab_size)
            return emb, num_found

        emb_path = 'emb_' + dataset_name + '.pkl'
        if not os.path.isfile(emb_path):
            print('creating pkl file for the emb numpy')
            pkl_path = fname + '.pkl'
            if not os.path.isfile(pkl_path):
                print('creating pkl file for the emb text file')
                emb_dict = load_vectors(fname)
                with open(pkl_path, 'wb') as f:
                    pkl.dump(emb_dict, f)
            else:
                print('loading pkl file')
                with open(pkl_path, 'rb') as f:
                    emb_dict = pkl.load(f)
                print('loading finished')

            emb, num_found = get_emb(emb_dict, self.vocab_size, emb_dim)
            with open(emb_path, 'wb') as f:
                pkl.dump((emb, num_found), f)
        else:
            print('loading pkl file')
            with open(emb_path, 'rb') as f:
                emb, num_found = pkl.load(f)
            print('loading finished')
        self.embeddings = emb

    def get_vocab_size(self):
        return len(self.word2id)

    def get_embeddings(self):
        return self.embeddings

    def encode_ids(self, text):
        return [self.word2id[x] for x in self.tokenize(text)]

    def encode_ids_pad(self, text):
        text = self.encode_ids(text)
        if len(text) < self.pad_len:
            id_len = len(text)
            ids = text + [0] * (self.pad_len - len(text))
        else:
            ids = text[:self.pad_len]
            id_len = self.pad_len
        return ids, id_len

    def decode_ids(self, ids):
        return [self.id2word[int(x)] for x in ids if x != 0]

    def tokenize(self, text):
        tokens = [x for x in text.split() if x in self.word2id]
        if len(tokens) == 0:
            tokens = ['<empty>']
        return tokens

