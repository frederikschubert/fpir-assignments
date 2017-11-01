#! /usr/bin/python
# -*- coding: utf-8 -*-


"""Rank sentences based on cosine similarity and a query."""


from argparse import ArgumentParser
from collections import Counter
import math
import numpy as np


def get_sentences(file_path):
    """Return a list of sentences from a file."""
    with open(file_path, encoding='utf-8') as hfile:
        return hfile.read().splitlines()


def get_top_k_words(sentences, k):
    """Return the k most frequent words as a list."""
    words = list()
    for sentence in sentences:
        words.extend(sentence.split(' '))
    c = Counter(words)
    most_common_words = list()
    for word in list(c.most_common(k)):
        most_common_words.append(word[0])
    return most_common_words


def encode(sentence, vocabulary):
    """Return a vector encoding the sentence."""
    encoded_sentence = np.zeros(len(vocabulary))
    for word in sentence.split(' '):
        try:
            index = vocabulary.index(word)
            encoded_sentence[index] = encoded_sentence[index] + 1
        except ValueError:
            pass
    return np.asarray(encoded_sentence)


def get_top_l_sentences(sentences, query, vocabulary, l):
    """
    For every sentence in "sentences", calculate the similarity to the query.
    Sort the sentences by their similarities to the query.

    Return the top-l most similar sentences as a list of tuples of the form
    (similarity, sentence).
    """
    encoded_query = encode(query, vocabulary)
    print(encoded_query)
    similarities = list()
    for sentence in sentences:
        encoded_sentence = encode(sentence, vocabulary)
        similarities.append((cosine_sim(encoded_query, encoded_sentence), sentence))
    similarities.sort(reverse=True, key=lambda entry: entry[0])
    return similarities[:l]


def cosine_sim(u, v):
    """Return the cosine similarity of u and v."""
    return dot_product(u, v) / (l2_norm(u) * l2_norm(v))

def dot_product(u, v):
    return sum(map(lambda i: u[i] * v[i], range(len(u))))

def l2_norm(v):
    return math.sqrt(sum(map(lambda x: x**2, v)))

def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('INPUT_FILE', help='An input file containing sentences, one per line')
    arg_parser.add_argument('QUERY', help='The query sentence')
    arg_parser.add_argument('-k', type=int, default=1000,
                            help='How many of the most frequent words to consider')
    arg_parser.add_argument('-l', type=int, default=10, help='How many sentences to return')
    args = arg_parser.parse_args()

    sentences = get_sentences(args.INPUT_FILE)
    top_k_words = get_top_k_words(sentences, args.k)
    query = args.QUERY.lower()

    print('using vocabulary: {}\n'.format(top_k_words))
    print('using query: {}\n'.format(query))

    # suppress numpy's "divide by 0" warning.
    # this is fine since we consider a zero-vector to be dissimilar to other vectors
    with np.errstate(invalid='ignore'):
        result = get_top_l_sentences(sentences, query, top_k_words, args.l)

    print('result:')
    for sim, sentence in result:
        print('{:.5f}\t{}'.format(sim, sentence))


if __name__ == '__main__':
    main()
