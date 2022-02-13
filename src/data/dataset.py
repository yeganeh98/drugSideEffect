import re
import logging
import collections

import torch
from torch.utils.data import Dataset
from torchtext.data import utils
from torchtext import vocab

import pandas as pd


class DrugSideEffect(Dataset):
    def __init__(self, root):
        self.root = root

        # read data
        data = pd.read_xml(root)
        negative_samples = remove_none_type(list(data["sentence"]))
        positive_samples = remove_none_type(list(data["text"]))

        # pre process and data cleaning
        negative_samples = clean_sentences(negative_samples)
        positive_samples = clean_sentences(positive_samples)
        # log number of data
        logging.info("number of positive samples: %d" % len(positive_samples))
        logging.info("number of negative samples: %d" % len(negative_samples))

        # create tokenizer
        tokenizer = utils.get_tokenizer("basic_english")
        self.vocab = get_vocab(negative_samples + positive_samples, tokenizer)
        self.index2word = self.vocab.get_itos()
        self.word2index = self.vocab.get_stoi()

        # create final data
        self.data, self.target = [], []
        # iterate on negative samples
        for sample in negative_samples:
            tokenize = []
            for token in tokenizer(sample):
                if token in self.word2index.keys():
                    tokenize.append(self.word2index[token])
                else:
                    tokenize.append(self.word2index["<UNK>"])
            # add sentence to data
            self.data.append(tokenize)
            self.target.append(0)

        # iterate on positive samples
        for sample in positive_samples:
            tokenize = []
            for token in tokenizer(sample):
                if token in self.word2index.keys():
                    tokenize.append(self.word2index[token])
                else:
                    tokenize.append(self.word2index["<UNK>"])
            # add sentence to data
            self.data.append(tokenize)
            self.target.append(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        target = self.target[idx]

        # covert to torch tensor
        sentence_tensor = torch.tensor(sentence, dtype=torch.long)
        return sentence_tensor, target


def get_vocab(sentence, tokenizer):
    counter = collections.Counter()
    for sentence in sentence:
        counter.update(tokenizer(sentence))
    vocabulary = vocab.vocab(counter, min_freq=5)
    vocabulary.insert_token("<UNK>", 1)
    vocabulary.insert_token("<PAD>", 0)
    return vocabulary


def remove_none_type(sentences):
    new_sentences = []
    for sentence in sentences:
        if isinstance(sentence, str):
            new_sentences.append(sentence)
    return new_sentences


def clean_sentences(sentences):
    clean_sen = []
    for sentence in sentences:
        clean_sen.append(texts_cleaner(sentence))
    return clean_sen


def texts_cleaner(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[0-9]*", "", text)
    text = re.sub(r"(”|“|-|\+|`|#|,|;|\|\[|\])*", "", text)
    text = re.sub(r"&amp", "", text)
    text = text.lower()

    return text


logging.basicConfig(level=logging.INFO)
dataset = DrugSideEffect("/home/kave/PycharmProjects/yugi-drug/data/myresult1 (1).xml")
