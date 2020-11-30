import pandas as pd
import os

train_file = 'train.tsv'
dev_file = 'dev.tsv'
test_file = 'test.tsv'

emotion_file = 'emotions.txt'


def get_emo(file_name):
    emo_list = [line.strip() for idx, line in enumerate(open(file_name, 'r').readlines())]
    return emo_list


def get_emotion(file, emo_list):
    text_list = []
    label_list = []
    with open(file, 'r') as f:
        for line in f.readlines():
            text, emotions, _ = line.split('\t')
            text_list.append(text)
            one_label = [0] * len(emo_list)
            for emo in emotions.split(','):
                one_label[int(emo)] = 1
            label_list.append(one_label)
    return text_list, label_list


def goemotion_data(file_path='', remove_stop_words=True, get_text=True):
    emo_list = get_emo(os.path.join(file_path, emotion_file))

    X_train, y_train = get_emotion(os.path.join(file_path, train_file), emo_list)
    X_dev, y_dev = get_emotion(os.path.join(file_path, dev_file), emo_list)
    X_test, y_test = get_emotion(os.path.join(file_path, test_file), emo_list)

    return X_train, y_train, X_dev, y_dev, X_test, y_test, emo_list
