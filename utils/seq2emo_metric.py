from sklearn import metrics
import numpy as np

def report_all(y, y_pre):
    print(metrics.classification_report(y, y_pre))

def get_metrics(y, y_pre):
    hamming_loss = metrics.hamming_loss(y, y_pre)
    macro_f1 = metrics.f1_score(y, y_pre, average='macro')
    macro_precision = metrics.precision_score(y, y_pre, average='macro')
    macro_recall = metrics.recall_score(y, y_pre, average='macro')
    micro_f1 = metrics.f1_score(y, y_pre, average='micro')
    micro_precision = metrics.precision_score(y, y_pre, average='micro')
    micro_recall = metrics.recall_score(y, y_pre, average='micro')
    return hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall


def get_multi_metrics(y, y_pre):
    multi_condition = np.sum(y, axis=1) > 1
    y = y[multi_condition]
    y_pre = y_pre[multi_condition]

    hamming_loss = metrics.hamming_loss(y, y_pre)
    macro_f1 = metrics.f1_score(y, y_pre, average='macro')
    macro_precision = metrics.precision_score(y, y_pre, average='macro')
    macro_recall = metrics.recall_score(y, y_pre, average='macro')
    micro_f1 = metrics.f1_score(y, y_pre, average='micro')
    micro_precision = metrics.precision_score(y, y_pre, average='micro')
    micro_recall = metrics.recall_score(y, y_pre, average='micro')
    return hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall


def get_single_metrics(y, y_pre):
    multi_condition = np.sum(y, axis=1) == 1
    y = y[multi_condition]
    y_pre = y_pre[multi_condition]
    hamming_loss = metrics.hamming_loss(y, y_pre)
    macro_f1 = metrics.f1_score(y, y_pre, average='macro')
    macro_precision = metrics.precision_score(y, y_pre, average='macro')
    macro_recall = metrics.recall_score(y, y_pre, average='macro')
    micro_f1 = metrics.f1_score(y, y_pre, average='micro')
    micro_precision = metrics.precision_score(y, y_pre, average='micro')
    micro_recall = metrics.recall_score(y, y_pre, average='micro')
    return hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall


# conver_to_binary = conver_to_binary_only
def jaccard_score(y_gold, y_pred):
    y_gold = np.asarray(y_gold).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    assert len(y_gold) == len(y_pred)
    tmp_sum = 0
    num_sample = len(y_gold)
    for i in range(num_sample):
        if sum(np.logical_or(y_gold[i], y_pred[i])) == 0:
            tmp_sum += 1
        else:
            tmp_sum += sum(y_gold[i] & y_pred[i]) / sum(np.logical_or(y_gold[i], y_pred[i]))
    return tmp_sum / num_sample


# def print_report(y_gold, y_pred):
