import pandas as pd
import multiprocessing
import numpy as np
from multiprocessing import Pool
import emoji
from emoji import UNICODE_EMOJI
import string

class TextProcessor:
    def __init__(self):
        self.text_processor = self.get_text_processor()
        self.emotion_tags = None
        self.printable = set(string.printable)

    def get_emotion_tags(self):
        emotion_hashtags = {'anger': ['anger', 'rage', 'pain', 'angry'],
                            'fear': ['fear', 'anxiety', 'horror', 'horrific'],
                            'joy': ['joy', 'happy', 'like', 'happiness', 'smile', 'peace', 'pleased', 'satisfied',
                                    'satisfying'],
                            'love': ['love', 'beautiful'],
                            'sadness': ['sadness', 'sad', 'sadness', 'depression', 'depressed', 'alone',
                                        'loneliness', 'lonely'],
                            'surprise': ['surprise', 'amazing', 'awesome', 'fascinate', 'fascinating', 'incredible',
                                         'marvelous', 'prodigious', 'shocking', 'stunning', 'surprising',
                                         'unbelievable'],
                            'thankfulness': ['thankfulness', 'thankful', 'gratitude', 'kindness', 'thanks',
                                             'gratefulness',
                                             'grateful'],
                            'disgust': ['disgust', 'disgusting', 'dislike', 'antipathy', 'distaste', 'distasteful',
                                        'hatred',
                                        'loathing'],
                            'guilt': ['guilt', 'guilty', 'culpability', 'disgrace', 'indiscretion', 'liability',
                                      'regret',
                                      'remorse', 'responsibility', 'shame', 'shameful', 'sin']
                            }
        self.emotion_hashtags = emotion_hashtags
        emotion_tags = []
        for emo in emotion_hashtags:
            emotion_tags.extend(['#' + x for x in emotion_hashtags[emo]])
        self.emotion_tags = emotion_tags

    def get_text_processor(self):
        from ekphrasis.classes.preprocessor import TextPreProcessor
        from ekphrasis.classes.tokenizer import SocialTokenizer
        from ekphrasis.dicts.emoticons import emoticons

        text_processor = TextPreProcessor(
            # terms that will be normalized
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                       'time', 'url', 'date', 'number'],
            # terms that will be annotated
            # annotate={"hashtag", "allcaps", "elongated", "repeated",
            #           'emphasis', 'censored'},

            # annotate={"repeated", "emphasis", "elongated"},

            annotate={"allcaps", "elongated", #  "repeated",
                          'emphasis', 'censored'},

            fix_html=True,  # fix HTML tokens

            # corpus from which the word statistics are going to be used
            # for word segmentation
            segmenter="twitter",

            # corpus from which the word statistics are going to be used
            # for spell correction
            corrector="twitter",

            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=True,  # spell correction for elongated words

            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=True).tokenize,

            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=[emoticons]
        )
        return text_processor

    def remove_tags(self, s):
        return ' '.join([x for x in s.split() if x not in self.emotion_tags])

    def emotion_detector(self, tweet):
        tokens = tweet.split()
        emo_found = []
        for token in tokens:
            if token.startswith('#'):
                for emo, tag_list in self.emotion_hashtags.items():
                    for word in tag_list:
                        emo_hashtag = '#' + word
                        if emo_hashtag == token:
                            if emo not in emo_found:
                                emo_found.append(emo)
        return emo_found

    def parallelize_dataframe(self, df, func):
        num_partitions = 12
        num_cores = multiprocessing.cpu_count()
        part_list = np.array_split(df, num_partitions)
        pool = Pool(num_cores)
        df = pd.concat(pool.map(func, part_list))
        pool.close()
        pool.join()
        return df

    def process_tweet(self, s):
        return ' '.join(self.text_processor.pre_process_doc(self.remove_tags(s)))

    def tweet_process(self, text):
        text = ' '.join(self.text_processor.pre_process_doc(self.remove_tags(text)))
        text = emoji.demojize(text, delimiters=(' ', ' '))
        tokens = text.split()
        ret_list = []
        for token in tokens:
            if len(token) > 3 and '_' in token:
                token = token.replace('_', ' ')

            if token[0] == '<' and token[-1] == '>':
                token = token[1:-1]

            ret_list.append(token)
        text = ' '.join(ret_list)
        return text

    def remove_dup_emoji(self, sent):
        ret = []
        for word in sent.split():
            emo_found = [char for char in word if char in UNICODE_EMOJI]
            if len(emo_found) > 1:
                word = emo_found[0]
            ret.append(word)
        return ' '.join(ret)

    def remove_underscope_for_emoji(self, text):
        tokens = text.split()
        ret_list = []
        for token in tokens:
            if len(token) > 3 and '_' in token:
                token = token.replace('_', ' ')

            if token[0] == '<' and token[-1] == '>':
                token = token[1:-1]

            ret_list.append(token)
        return ' '.join(ret_list)

    def only_printable(self, text):
        """
        Usage Warning, for the sake of efficient, this method did not rejoin the string with space
        Therefore, in the 'processing_pipeline', I put it before 'remove_underscope_for_emoji'
        """

        text = ''.join([x for x in text if x in self.printable])
        return text

    def processing_pipeline(self, text, remove_hashtag=False):
        text = text.lower().strip()
        if remove_hashtag:
            text = self.remove_tags(text)
        text = ' '.join(self.text_processor.pre_process_doc(text))
        # text = only_printable(text)
        text = emoji.demojize(text, delimiters=(' ', ' '))
        text = self.only_printable(text)
        text = self.remove_underscope_for_emoji(text)
        return text

# print(processing_pipelie('e day હત ા શ ા ર ો ગ મ ા ટ ે હ ો મ ી ય ો પ ે થ ી homeop'))