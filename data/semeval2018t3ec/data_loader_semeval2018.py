import numpy as np


def read_sem18(file_name):
    emotion_list = ["anger", "anticipation", "disgust", "fear", "joy", "love",
                    "optimism", "pessimism", "sadness", "surprise", "trust"]
    label_list = []
    text_list = []
    with open(file_name, 'r', encoding="utf-8") as f:
        num_tokens_per_line = None
        for idx, line in enumerate(f.readlines()):
            line = line.strip()
            if idx == 0:
                column_names = line.split('\t')
                num_tokens_per_line = len(column_names)
            else:
                tokens = line.split('\t')
                assert len(tokens) == num_tokens_per_line

                one_label = [int(x) for x in tokens[2:]]
                assert len(one_label) == len(emotion_list)
                label_list.append(one_label)
                text_list.append(tokens[1])
    return text_list, label_list, emotion_list


def stats(file_name):
    text_list, label_list, emotion_list = read_sem18(file_name)

    from collections import Counter

    label_count_list = []
    for label in label_list:
        label_count_list.append(sum(label))
    print(Counter(label_count_list))

    emo_count_list = []
    for label in label_list:
        label = np.asarray(label)
        pos_emo = np.where(label == 1)[0].tolist()
        emo_count_list.extend(pos_emo)

    emo_count_dict = Counter(emo_count_list)
    for emo in emotion_list:
        print(emo, emo_count_dict[emotion_list.index(emo)]/len(text_list))


def load_sem18_all(is_test=False, load_all=True):
    if load_all:
        train_file = 'SemEval18_train.txt'
        dev_file = 'SemEval18_dev.txt'
        test_file = 'SemEval18_test.txt'
        files = [train_file, dev_file, test_file]
    else:
        if not is_test:
            train_file = 'SemEval18_train.txt'
            dev_file = 'SemEval18_dev.txt'
            files = [train_file, dev_file]
        else:
            test_file = 'SemEval18_test.txt'
            files = [test_file]

    all_data = []
    all_label = []
    path_prefix = 'data/semeval2018t3ec/'
    emo_list = None
    for file in files:
        data, label, emo_list = read_sem18(path_prefix + file)
        all_data.extend(data)
        all_label.extend(label)

    return all_data, all_label, emo_list


"""
Train
Counter({2: 2773, 3: 2114, 1: 982, 4: 658, 0: 204, 5: 96, 6: 11})
anger 0.37203860778005265
anticipation 0.14302427610412402
disgust 0.3805206200643463
fear 0.181632056156771
joy 0.36224042117578237
love 0.10236911377595788
optimism 0.29014331675928634
pessimism 0.11626206493126645
sadness 0.29365311494589064
surprise 0.052793214390172566
trust 0.05220824802573852

dev
Counter({2: 351, 3: 290, 1: 117, 4: 99, 5: 14, 0: 14, 6: 1})
anger 0.35553047404063204
anticipation 0.1399548532731377
disgust 0.3600451467268623
fear 0.136568848758465
joy 0.45146726862302483
love 0.1489841986455982
optimism 0.34650112866817157
pessimism 0.11286681715575621
sadness 0.29909706546275394
surprise 0.039503386004514675
trust 0.04853273137697517

test
Counter({2: 1367, 3: 1055, 1: 382, 4: 316, 0: 75, 5: 60, 6: 4})
anger 0.3378336913163547
anticipation 0.13040810064436945
disgust 0.33722000613685177
fear 0.14881865602945687
joy 0.44246701442160175
love 0.15833077631175208
optimism 0.3507210800859159
pessimism 0.11506597115679656
sadness 0.2945688861613992
surprise 0.052163240257747774
trust 0.046946916231972995
"""
