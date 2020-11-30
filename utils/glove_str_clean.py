"""
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

import regex as re
FLAGS = re.MULTILINE | re.DOTALL


def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join([""] + [re.sub(r"([A-Z])", r" \1", hashtag_body, flags=FLAGS)])
    return result


def allcaps(text):
    text = text.group()
    return text.lower() + " "


def glove_str_clean(text):
    """
    A python version of glove twitter pretrain embedding data preprocessing
    :param text: text
    :return: cleaned text
    """

    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`-]?"

    # function so code less repetitive
    def re_sub(pattern, repl, _text):
        return re.sub(pattern, repl, _text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ", text)
    text = re_sub(r"@\w+", " <user> ", text)
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " <smile> ", text)
    text = re_sub(r"{}{}p+".format(eyes, nose), " <lolface> ", text)
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " <sadface> ", text)
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " <neutralface> ", text)
    text = re_sub(r"/", " / ", text)
    text = re_sub(r"<3", " <heart> ", text)
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ", text)
    text = re_sub(r"#\S+", hashtag, text)
    text = re_sub(r"([!?.]){2,}", r"\1  <repeat> ", text)
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2  <elong> ", text)

    # -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps, text)

    return ' '.join(text.lower().split())





