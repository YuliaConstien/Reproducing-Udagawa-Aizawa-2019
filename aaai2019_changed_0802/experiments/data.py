import os
import random
import sys
import pdb
import copy
import re
from collections import OrderedDict, defaultdict

#import en_vectors_web_lg

import torch
import numpy as np



# special tokens
SPECIAL = [
    '<eos>',
    '<unk>',
    '<selection>',
    '<pad>',
]

# tokens that stops either a sentence or a conversation
STOP_TOKENS = [
    '<eos>',
    '<selection>',
]


def get_tag(tokens, tag):
    """Extracts the value inside the given tag."""
    return tokens[tokens.index('<' + tag + '>') + 1:tokens.index('</' + tag + '>')]


def to_float(tokens):
    return [float(token) for token in tokens.split()]


def read_lines(file_name):
    """Reads all the lines from the file."""
    assert os.path.exists(file_name), 'file does not exists %s' % file_name
    lines = []
    with open(file_name, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines


class Dictionary(object):
    """Maps words into indeces.

    It has forward and backward indexing.
    """

    def __init__(self, init=True):
        self.word2idx = OrderedDict()
        # there is a OrderedDict where the keys are tokens and the values are indices
        # if you call get on the dict and the key (token) you get the index
        self.idx2word = []
        if init:
            # add special tokens if asked
            for i, k in enumerate(SPECIAL):
                self.word2idx[k] = i
                self.idx2word.append(k)
                # here: add special characters to word embedding or later?

    def add_word(self, word):
        """Adds a new word, if the word is in the dictionary, just returns its index."""
        if word not in self.word2idx:
            new_word_idx = len(self.idx2word)
            self.word2idx[word] = new_word_idx
            self.idx2word.append(word)
            # here: add to word embedding or all in once later?
        return self.word2idx[word]

    def i2w(self, idx):
        """Converts a list of indeces into words."""
        return [self.idx2word[i] for i in idx]

    def w2i(self, words):
        """Converts a list of words into indeces. Uses <unk> for the unknown words."""
        unk = self.word2idx.get('<unk>', None)
        return [self.word2idx.get(w, unk) for w in words]

    def get_idx(self, word):
        """Gets index for the word."""
        unk = self.word2idx.get('<unk>', None) # get the index of unk token
        return self.word2idx.get(word, unk) # get the index of word. if word is not in dict, get index of unk

    def get_word(self, idx):
        """Gets word by its index."""
        return self.idx2word[idx]

    def __len__(self):
        return len(self.idx2word)

    def read_tag(file_name, tag, freq_cutoff=-1, init_dict=True):
        """Extracts all the values inside the given tag.

        Applies frequency cuttoff if asked.
        """
        token_freqs = OrderedDict()
        with open(file_name, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                tokens = get_tag(tokens, tag)
                for token in tokens:
                    token_freqs[token] = token_freqs.get(token, 0) + 1
        dictionary = Dictionary(init=init_dict)
        token_freqs = sorted(token_freqs.items(),
                             key=lambda x: x[1], reverse=True)
        for token, freq in token_freqs:
            if freq > freq_cutoff:
                dictionary.add_word(token)
        return dictionary

    def from_file(file_name, freq_cutoff, args):
        """Constructs a dictionary from the given file."""
        assert os.path.exists(file_name)
        word_dict = Dictionary.read_tag(
            file_name, 'dialogue', freq_cutoff=freq_cutoff)
        pretrained_embeds = create_embeddings(word_dict.word2idx, args=args, special_characters=SPECIAL)
        return word_dict, pretrained_embeds


class WordCorpus(object):
    """An utility that stores the entire dataset.

    It has the train, valid and test datasets and corresponding dictionaries.
    """

    def __init__(self, path, args, freq_cutoff=2, train='train.txt',
                 valid='valid.txt', test='test.txt', verbose=False, word_dict=None):
        self.verbose = verbose
        self.args = args
        if word_dict is None:
            # this reads the words from the training data and adds them to a dictionary of tokens-idx
            # (called word2idx in Dictionary class, here it is now called word_dict)
            # also creates a frequency dict
            self.word_dict, self.pretrained_embeds = Dictionary.from_file(
                os.path.join(path, train), freq_cutoff=freq_cutoff, args=self.args)
        else:
            self.word_dict = word_dict
            print("Warning: no pretrained_embeds have been created in WordCorpus __init__")

        # construct all 3 datasets
        self.train = self.tokenize(os.path.join(path, train)) if train else []
        self.valid = self.tokenize(os.path.join(path, valid)) if valid else []
        self.test = self.tokenize(os.path.join(path, test)) if test else []

        # find out the output length from the train dataset
        self.output_length = max([len(x) for x in self.train])

    def tokenize(self, file_name, test=False):
        """Tokenizes the file and produces a dataset.
        This function is only called within WordCorpus itself (in the __init__) and
        the WordCorpus class is instantiated in train.py.

        :arg file_name is defined by the parser argument args.data, which is by default data/onecommon and
        whether this function tokenizes train/val/test data is determined in __init__ of WordCorpus"""
        lines = read_lines(file_name) # lines also has the context info (blob representation etc.)
        random.shuffle(lines)

        unk = self.word_dict.get_idx('<unk>') # get index of "unknown" token
        dataset, total, unks = [], 0, 0
        for line in lines:
            tokens = line.split() # split at whitespace

            # get all the parts of the data where the blobs are described (found between input tags)
            input_vals = [float(val) for val in get_tag(tokens, 'input')]

            # get all the parts of the data that is dialogue and
            # convert all the words from dialogue to indices of the dict
            # where do new words get added to dict? -> in WordCorpus init (read comments)
            word_idxs = self.word_dict.w2i(get_tag(tokens, 'dialogue'))

            output_idx = int(get_tag(tokens, 'output')[0]) # get which blob the player chose
            dataset.append((input_vals, word_idxs, output_idx)) # adds these three above into dataset as tuples
            # compute statistics
            total += len(word_idxs)
            unks += np.count_nonzero([idx == unk for idx in word_idxs])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%' % (
                file_name, total, unks, 100. * unks / total))
        return dataset

    def train_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.train), bsz,
                                        shuffle=shuffle, device=device)

    def valid_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.valid), bsz,
                                        shuffle=shuffle, device=device)

    def test_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.test), bsz, shuffle=shuffle,
                                        device=device)

    def _split_into_batches(self, dataset, bsz, shuffle=True, device=None):
        """Splits given dataset into batches."""
        if shuffle:
            random.shuffle(dataset)

        # sort by dialog length and pad
        dataset.sort(key=lambda x: len(x[1]))
        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0,
        }

        for i in range(0, len(dataset), bsz):
            inputs, words, output = [], [], []
            for j in range(i, min(i + bsz, len(dataset))):
                inputs.append(dataset[j][0])
                words.append(dataset[j][1])
                output.append(dataset[j][2])

            # the longest dialogue in the batch
            max_len = len(words[-1])

            # pad all the dialogues to match the longest dialogue
            for j in range(len(words)):
                stats['n'] += max_len
                stats['nonpadn'] += len(words[j])
                # one additional pad
                words[j] += [pad] * (max_len - len(words[j]) + 1)

            # construct tensor for context
            ctx = torch.FloatTensor(inputs)
            data = torch.LongTensor(words).transpose(0, 1).contiguous()
            # construct tensor for selection target
            sel_tgt = torch.LongTensor(output)
            if device is not None:
                ctx = ctx.to(device)
                data = data.to(device)
                sel_tgt = sel_tgt.to(device)

            # construct tensor for input and target
            inpt = data.narrow(0, 0, data.size(0) - 1)
            tgt = data.narrow(0, 1, data.size(0) - 1).view(-1)

            batches.append((ctx, inpt, tgt, sel_tgt))

        if shuffle:
            random.shuffle(batches)

        return batches, stats


def create_embeddings(word_idx_dict, args, special_characters=None):
    """
    a function that loads the GloVe pretrained embeddings for the given dictionary of words

    :param word_idx_dict: the original word and index dictionary created in Dictionary class
    :param args: parser arguments
    :param special_characters: list of special characters
    :return: tensor or array of embeddings
    """
    #  special chars are already in embedding (see init of Dictionary), first specials then words

    if args.embed_source == "en_vectors_web_lg":
        import en_vectors_web_lg
        embed_loader = en_vectors_web_lg.load()
    else:
        if args.debug_vectors_option == "four":
            import spacy.cli
            spacy.cli.download("en_core_web_lg")
            embed_loader = spacy.load("en_core_web_lg")
        elif args.debug_vectors_option == "three":
            import spacy
            spacy.prefer_gpu()
            import en_core_web_lg
            embed_loader = en_core_web_lg.load()

    embeds_list = []

    # add special chars to embedding by checking how many of word_dict are special chars
    # only need this if problems with unknown tokens. What happens if spacy doesnt recognize
    # no_special_chars = len(special_characters)
    # assert word_dict[special_characters[-1]] == no_special_chars -1, "problem in create_embeddings special chars"

    # add words in word_dict to embedding
    for word in word_idx_dict.keys():
        embeds_list.append(embed_loader(word).vector)

    pretrained_embeds = np.array(embeds_list)

    if torch.cuda.is_available():
        pretrained_embeds = torch.from_numpy(pretrained_embeds)

    return pretrained_embeds

