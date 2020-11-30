import os
import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from models.cc_lstm import CCLSTMClassifier
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
import pickle as pkl
from utils.seq2emo_metric import get_metrics, get_multi_metrics, jaccard_score, report_all, get_single_metrics
from utils.tokenizer import GloveTokenizer
from copy import deepcopy
from allennlp.modules.elmo import Elmo, batch_to_ids
import argparse
from data.data_loader import load_sem18_data, load_goemotions_data
from utils.scheduler import get_cosine_schedule_with_warmup
import utils.nn_utils as nn_utils
from utils.others import find_majority
from utils.file_logger import get_file_logger

# Argument parser
parser = argparse.ArgumentParser(description='Options')
parser.add_argument('--batch_size', default=32, type=int, help="batch size")
parser.add_argument('--pad_len', default=50, type=int, help="batch size")
parser.add_argument('--postname', default='', type=str, help="post name")
parser.add_argument('--gamma', default=0.2, type=float, help="post name")
parser.add_argument('--folds', default=5, type=int, help="num of folds")
parser.add_argument('--en_lr', default=5e-4, type=float, help="encoder learning rate")
parser.add_argument('--de_lr', default=1e-4, type=float, help="decoder learning rate")
parser.add_argument('--loss', default='ce', type=str, help="loss function ce/focal")
parser.add_argument('--dataset', default='sem18', type=str, choices=['sem18', 'goemotions'])
parser.add_argument('--en_dim', default=1200, type=int, help="dimension")
parser.add_argument('--de_dim', default=400, type=int, help="dimension")
parser.add_argument('--criterion', default='jaccard', type=str, choices=['jaccard', 'macro', 'micro', 'h_loss'])
parser.add_argument('--glove_path', default='data/glove.840B.300d.txt', type=str)
parser.add_argument('--attention', default='dot', type=str, help='general/mlp/dot')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
parser.add_argument('--encoder_dropout', default=0.2, type=float, help='dropout rate')
parser.add_argument('--decoder_dropout', default=0, type=float, help='dropout rate')
parser.add_argument('--attention_dropout', default=0.2, type=float, help='dropout rate')
parser.add_argument('--patience', default=13, type=int, help='dropout rate')
parser.add_argument('--download_elmo', action='store_true')
parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--glorot_init', action='store_true')
parser.add_argument('--warmup_epoch', default=0, type=float, help='')
parser.add_argument('--stop_epoch', default=10, type=float, help='')
parser.add_argument('--max_epoch', default=20, type=float, help='')
parser.add_argument('--min_lr_ratio', default=0.1, type=float, help='')
parser.add_argument('--fix_emb', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--input_feeding', action='store_true')
parser.add_argument('--dev_split_seed', type=int, default=0)
parser.add_argument('--normal_init', action='store_true')
parser.add_argument('--unify_decoder', action='store_true')
parser.add_argument('--eval_every', type=bool, default=True)
parser.add_argument('--log_path', type=str, default=None)
parser.add_argument('--attention_heads', type=int, default=1)
parser.add_argument('--concat_hidden', action='store_true')
parser.add_argument('--no_cross', action='store_true')
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--shuffle_emo', type=str, default=None)

args = parser.parse_args()

if args.log_path is not None:
    dir_path = os.path.dirname(args.log_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

logger = get_file_logger(args.log_path)  # Note: this is ugly, but I am lazy

SRC_EMB_DIM = 300
MAX_LEN_DATA = args.pad_len
PAD_LEN = MAX_LEN_DATA
MIN_LEN_DATA = 3
BATCH_SIZE = args.batch_size
CLIPS = 0.666
GAMMA = 0.5
SRC_HIDDEN_DIM = args.en_dim
TGT_HIDDEN_DIM = args.de_dim
VOCAB_SIZE = 60000
ENCODER_LEARNING_RATE = args.en_lr
DECODER_LEARNING_RATE = args.de_lr
ATTENTION = args.attention
PATIENCE = args.patience
WARMUP_EPOCH = args.warmup_epoch
STOP_EPOCH = args.stop_epoch
MAX_EPOCH = args.max_epoch
RANDOM_SEED = args.seed
# Seed
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# Init Elmo model
if args.download_elmo:
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
else:
    options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0).cuda()
elmo.eval()

GLOVE_EMB_PATH = args.glove_path
glove_tokenizer = GloveTokenizer(PAD_LEN)

data_path_postfix = '_split'
data_pkl_path = 'data/' + args.dataset + data_path_postfix + '_data.pkl'
if not os.path.isfile(data_pkl_path):
    if args.dataset == 'sem18':
        X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name = \
            load_sem18_data()
    elif args.dataset == 'goemotions':
        X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name = \
            load_goemotions_data()
    else:
        raise NotImplementedError

    with open(data_pkl_path, 'wb') as f:
        logger('Writing file')
        pkl.dump((X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name), f)

else:
    with open(data_pkl_path, 'rb') as f:
        logger('loading file')
        X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name = pkl.load(f)

NUM_EMO = len(EMOS)

class TestDataReader(Dataset):
    def __init__(self, X, pad_len, max_size=None):
        self.glove_ids = []
        self.glove_ids_len = []
        self.pad_len = pad_len
        self.build_glove_ids(X)

    def build_glove_ids(self, X):
        for src in X:
            glove_id, glove_id_len = glove_tokenizer.encode_ids_pad(src)
            self.glove_ids.append(glove_id)
            self.glove_ids_len.append(glove_id_len)

    def __len__(self):
        return len(self.glove_ids)

    def __getitem__(self, idx):
        return torch.LongTensor(self.glove_ids[idx]), \
               torch.LongTensor([self.glove_ids_len[idx]])


class TrainDataReader(TestDataReader):
    def __init__(self, X, y, pad_len, max_size=None):
        super(TrainDataReader, self).__init__(X, pad_len, max_size)
        self.y = []
        self.read_target(y)

    def read_target(self, y):
        self.y = y

    def __getitem__(self, idx):
        return torch.LongTensor(self.glove_ids[idx]), \
               torch.LongTensor([self.glove_ids_len[idx]]), \
               torch.LongTensor(self.y[idx])


def elmo_encode(ids):
    data_text = [glove_tokenizer.decode_ids(x) for x in ids]
    with torch.no_grad():
        character_ids = batch_to_ids(data_text).cuda()
        elmo_emb = elmo(character_ids)['elmo_representations']
        elmo_emb = (elmo_emb[0] + elmo_emb[1]) / 2  # avg of two layers
    return elmo_emb


def show_classification_report(gold, pred):
    from sklearn.metrics import classification_report
    logger(classification_report(gold, pred, target_names=EMOS, digits=4))


def eval(model, best_model, loss_criterion, es, dev_loader, dev_set):
    # Evaluate
    exit_training = False
    model.eval()
    test_loss_sum = 0
    preds = []
    gold = []
    logger("Evaluating:")
    for i, (src, src_len, trg) in tqdm(enumerate(dev_loader), total=int(len(dev_set) / BATCH_SIZE), disable=True):
        with torch.no_grad():
            elmo_src = elmo_encode(src)

            pred = model.greedy_decode_batch(src.cuda(), src_len.cuda(), elmo_src.cuda())

            gold.append(trg.data.numpy())
            preds.append(pred.cpu().numpy())
            del pred

    preds = np.concatenate(preds, axis=0)
    gold = np.concatenate(gold, axis=0)
    # binary_gold = conver_to_binary(gold)
    # binary_preds = conver_to_binary(preds)
    metric = get_metrics(gold, preds)
    jaccard = jaccard_score(gold, preds)
    logger("Evaluation results:")
    # show_classification_report(binary_gold, binary_preds)
    logger("Evaluation Loss", test_loss_sum / len(dev_set))

    logger('Normal: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4], 'micro P', metric[5],
          'micro R', metric[6])
    metric_2 = get_multi_metrics(gold, preds)
    logger('Multi only: h_loss:', metric_2[0], 'macro F', metric_2[1], 'micro F', metric_2[4])
    logger('Jaccard:', jaccard)

    if args.criterion == 'loss':
        criterion = test_loss_sum
    elif args.criterion == 'macro':
        criterion = 1 - metric[1]
    elif args.criterion == 'micro':
        criterion = 1 - metric[4]
    elif args.criterion == 'h_loss':
        criterion = metric[0]
    elif args.criterion == 'jaccard':
        criterion = 1 - jaccard
    else:
        raise ValueError

    if es.step(criterion):  # overfitting
        del model
        logger('overfitting, loading best model ...')
        model = best_model
        exit_training = True
    else:
        if es.is_best():
            if best_model is not None:
                del best_model
            logger('saving best model ...')
            best_model = deepcopy(model)
        else:
            logger(f'patience {es.cur_patience} not best model , ignoring ...')
            if best_model is None:
                best_model = deepcopy(model)

    return model, best_model, exit_training


def train(X_train, y_train, X_dev, y_dev, X_test, y_test):
        train_set = TrainDataReader(X_train, y_train, MAX_LEN_DATA)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

        dev_set = TrainDataReader(X_dev, y_dev, MAX_LEN_DATA)
        dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE*3, shuffle=False)

        test_set = TestDataReader(X_test, MAX_LEN_DATA)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE*3, shuffle=False)

        # Model initialize
        model = CCLSTMClassifier(
            emb_dim=SRC_EMB_DIM,
            hidden_dim=SRC_HIDDEN_DIM,
            num_label=NUM_EMO,
            vocab_size=glove_tokenizer.get_vocab_size(),
            args=args
        )

        if args.fix_emb:
            para_group = [
                {'params': [p for n, p in model.named_parameters() if n.startswith("encoder") and
                            not 'encoder.embeddings' in n], 'lr': args.en_lr},
                {'params': [p for n, p in model.named_parameters() if n.startswith("decoder")], 'lr': args.de_lr}]
        else:
            para_group = [
                {'params': [p for n, p in model.named_parameters() if n.startswith("encoder")], 'lr': args.en_lr},
                {'params': [p for n, p in model.named_parameters() if n.startswith("decoder")], 'lr': args.de_lr}]
        loss_criterion = nn.CrossEntropyLoss() # reduction='sum'
        optimizer = optim.Adam(para_group)
        if args.scheduler:
            epoch_to_step = len(train_set) / BATCH_SIZE
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(WARMUP_EPOCH * epoch_to_step),
                num_training_steps=int(STOP_EPOCH * epoch_to_step),
                min_lr_ratio=args.min_lr_ratio
            )

        if args.glorot_init:
            logger('use glorot initialization')
            for group in para_group:
                nn_utils.glorot_init(group['params'])

        model.load_encoder_embedding(glove_tokenizer.get_embeddings(), fix_emb=args.fix_emb)
        model.cuda()

        # Start training
        EVAL_EVERY = int(len(train_set) / BATCH_SIZE / 4)
        best_model = None
        es = EarlyStopping(patience=PATIENCE)
        update_step = 0
        exit_training = False
        for epoch in range(1, MAX_EPOCH+1):
            logger('Training on epoch=%d -------------------------' % (epoch))
            train_loss_sum = 0
                # print('Current encoder learning rate', scheduler.get_lr())
                # print('Current decoder learning rate', scheduler.get_lr())


            for i, (src, src_len, trg) in tqdm(enumerate(train_loader), total=int(len(train_set) / BATCH_SIZE)):
                model.train()
                update_step += 1
                # print('i=%d: ' % (i))
                # trg = torch.index_select(trg, 1, torch.LongTensor(list(range(1, len(EMOS)+1))))


                optimizer.zero_grad()

                elmo_src = elmo_encode(src)

                loss = model.loss(src.cuda(), src_len.cuda(), elmo_src.cuda(), trg.cuda())

                loss.backward()
                train_loss_sum += loss.data.cpu().numpy() * src.shape[0]

                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPS)
                optimizer.step()
                if args.scheduler:
                    scheduler.step()

                if update_step % EVAL_EVERY == 0: #
                    model, best_model, exit_training = eval(model, best_model, loss_criterion, es, dev_loader, dev_set)
                    if exit_training:
                        break

            logger(f"Training Loss for epoch {epoch}:", train_loss_sum / len(train_set))
            # model, best_model, exit_training = eval(model, best_model, loss_criterion, es, dev_loader, dev_set)
            if exit_training:
                break

        # final_testing
        model.eval()
        preds = []
        logger("Testing:")
        for i, (src, src_len) in tqdm(enumerate(test_loader), total=int(len(test_set) / BATCH_SIZE)):
            with torch.no_grad():
                elmo_src = elmo_encode(src)
                pred = model.greedy_decode_batch(src.cuda(), src_len.cuda(), elmo_src.cuda())
                preds.append(pred.cpu().numpy())
                del pred

        preds = np.concatenate(preds, axis=0)
        gold = np.asarray(y_test)
        binary_gold = gold
        binary_preds = preds
        logger("NOTE, this is on the test set")
        metric = get_metrics(binary_gold, binary_preds)
        logger('Normal: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
        metric = get_multi_metrics(binary_gold, binary_preds)
        logger('Multi only: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
        # show_classification_report(binary_gold, binary_preds)
        logger('Jaccard:', jaccard_score(gold, preds))
        return binary_gold, binary_preds


def main():

    global X_train_dev, X_test, y_train_dev, y_test
    if args.shuffle_emo is not None:
        new_order = np.asarray([int(tmp) for tmp in args.shuffle_emo.split()])
        y_train_dev = np.asarray(y_train_dev).T[new_order].T
        y_test = np.asarray(y_test).T[new_order].T

    glove_tokenizer.build_tokenizer(X_train_dev + X_test, vocab_size=VOCAB_SIZE)
    glove_tokenizer.build_embedding(GLOVE_EMB_PATH, dataset_name=data_set_name)

    from sklearn.model_selection import ShuffleSplit, KFold

    kf = KFold(n_splits=args.folds, random_state=args.dev_split_seed)
    # kf.get_n_splits(X_train_dev)

    all_preds = []
    gold_list = None

    for i, (train_index, dev_index) in enumerate(kf.split(y_train_dev)):
        logger('STARTING Fold -----------', i + 1)
        X_train, X_dev = [X_train_dev[i] for i in train_index], [X_train_dev[i] for i in dev_index]
        y_train, y_dev = [y_train_dev[i] for i in train_index], [y_train_dev[i] for i in dev_index]

        gold_list, pred_list = train(X_train, y_train, X_dev, y_dev, X_test, y_test)
        all_preds.append(pred_list)
        if args.no_cross:
            break

    all_preds = np.stack(all_preds, axis=0)

    shape = all_preds[0].shape
    mj = np.zeros(shape)
    for m in range(shape[0]):
        for n in range(shape[1]):
            mj[m, n] = find_majority(np.asarray(all_preds[:, m, n]).reshape((-1)))[0]

    final_pred = mj

    show_classification_report(gold_list, final_pred)
    metric = get_metrics(gold_list, final_pred)
    logger('Normal: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
    metric = get_multi_metrics(gold_list, final_pred)
    logger('Multi only: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])
    metric = get_single_metrics(gold_list, final_pred)
    logger('Single only: h_loss:', metric[0], 'macro F', metric[1], 'micro F', metric[4])


    logger('Final Jaccard:', jaccard_score(gold_list, final_pred))
    logger(os.path.basename(__file__))
    logger(args)

    if args.output_path is not None:
        with open(args.output_path, 'bw') as _f:
            pkl.dump(final_pred, _f)


if __name__ == '__main__':
    main()


