from sklearn.metrics import *
import numpy as np

def find_majority(k):
    myMap = {}
    maximum = ('', 0)  # (occurring element, occurrences)
    for n in k:
        if n in myMap:
            myMap[n] += 1
        else:
            myMap[n] = 1
        # Keep track of maximum on the go
        if myMap[n] > maximum[1]:
            maximum = (n, myMap[n])
    return maximum

def test_threshold(gold_list, pred_list):
    thres_dict = {}
    for threshold in [0.025 * x for x in range(1, 40)]:
        # print('Threshold:', threshold, end=' ')
        tmp_pred_list = np.asarray([1 & (v > threshold) for v in pred_list])

        f1 = f1_score(gold_list, tmp_pred_list, average='macro')
        f1_micro = f1_score(gold_list, tmp_pred_list, average='micro')
        thres_dict[threshold] = f1
        # print('macro F1:', f1,
        #       'micro F1:', f1_micro
        #       )

    arg_max = max(thres_dict, key=thres_dict.get)
    print('Best Dev Macro F1', thres_dict[arg_max])
    return arg_max


def use_threshold(pred_list, threshold):
    return np.asarray([1 & (v > threshold) for v in pred_list])
