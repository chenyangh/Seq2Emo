import numpy as np
from data.semeval2018t3ec.data_loader_semeval2018 import load_sem18_all
from data.goemotions.goemotion_loader import goemotion_data
from tqdm import tqdm
import pandas as pd
from utils.tweet_processor import TextProcessor
import os

MAX_LEN_DATA = 50

def load_sem18_data():
    text_processor = TextProcessor()
    def __process_data(data, label, EMOS):
        EMOS_DIC = {}
        for idx, emo in enumerate(EMOS):
            EMOS_DIC[emo] = idx
        target = []
        for l in label:
            a_target = [0] * len(EMOS)
            pos_position = np.where(np.asarray(l) == 1)[0].tolist()
            for pos in pos_position:
                a_target[pos] = 1
            target.append(a_target)

        source = []
        for text in tqdm(data):
            text = text_processor.processing_pipeline(text)
            source.append(text)
        return source, target, EMOS, EMOS_DIC

    data, label, EMOS = load_sem18_all(is_test=False, load_all=False)
    X_train_dev, y_train_dev, _, _ = __process_data(data, label, EMOS)
    data, label, EMOS = load_sem18_all(is_test=True, load_all=False)
    X_test, y_test, EMOS, EMOS_DIC = __process_data(data, label, EMOS)
    return X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, 'sem18_split'


def load_goemotions_data():
    X_train_raw, y_train, X_dev_raw, y_dev, X_test_raw, y_test, emo_list = goemotion_data(file_path='data/goemotions')
    X_train_dev_raw = X_train_raw + X_dev_raw
    y_train_dev = y_train + y_dev
    # preprocess
    text_processor = TextProcessor()

    X_train_dev = []
    for text in tqdm(X_train_dev_raw):
        text = text_processor.processing_pipeline(text)
        X_train_dev.append(text)

    X_test = []
    for text in tqdm(X_test_raw):
        text = text_processor.processing_pipeline(text)
        X_test.append(text)

    EMOS = emo_list
    EMOS_DIC = {}
    for idx, emo in enumerate(EMOS):
        EMOS_DIC[emo] = idx
    data_set_name = ''
    return X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name

