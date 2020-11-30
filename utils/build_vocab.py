def build_vocab(data_list, vocab_size, fill_vocab=True):
    """
    data_list the list of sentences
    vocab_size is the num of vocabulary you wanted it to have
    if vocab_size < actual num of vocab, and fill_vocab = False, only vocab_size will be created
                                            if fill_vocab = True, it will return as many vocab as possible
    if vocab_size > actual num of vocab, an error is raised
    """

    word_count = {}
    word2id = {}
    id2word = {}
    for tokens in data_list:
        for word in tokens.split():
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

    word_list = [x for x, _ in sorted(word_count.items(), key=lambda v: v[1], reverse=True)]
    print('found ' + str(len(word_count)) + ' words')
    if len(word_count) < vocab_size:
        print('Vocab less than requested, will fill the vocab as many as possible')

    # vocab_size = word_count

    # add <pad> first
    word2id['<pad>'] = 0
    id2word[0] = '<pad>'

    word2id['<unk>'] = 1
    id2word[1] = '<unk>'
    word2id['<empty>'] = 2
    id2word[2] = '<empty>'

    n = len(word2id)
    if not fill_vocab:
        if len(word_count) > vocab_size:
            word_list = word_list[:vocab_size - n]


    for word in word_list:
        word2id[word] = n
        id2word[n] = word
        n += 1

    if fill_vocab:
        print('filling vocab to ' + str(len(id2word)))
        return word2id, id2word, len(id2word)
    return word2id, id2word, len(word2id)
