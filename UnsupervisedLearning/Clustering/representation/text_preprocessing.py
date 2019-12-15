import csv
import re
import numpy
from nltk.tokenize.treebank import TreebankWordTokenizer


class TextRepresentation:
    def __init__(self):
        self.tokenizer = TreebankWordTokenizer()

    def _top_words(self, word2count: dict, num: int):
        """
        This helper function sort the bag of words histogram and returns a sequence of the top ``num`` words.

        :param word2count: A dictionary mapping each word and corresponding frequency.
        :param num: Predefined number of top words.
        :return: A list of word strings.
        """
        if num < len(word2count):
            old_words = []
            for i, (k, v) in enumerate(word2count.items()):
                old_words.append(v)
                if i == num - 1:
                    break

            top_words = []
            for i in range(len(old_words)):
                current_word = ''
                for word, count in word2count.items():
                    if old_words[i] < count:
                        old_words[i] = count
                        current_word = word
                word2count.pop(current_word)
                top_words.append(current_word)
            return top_words
        else:
            raise ValueError('number of top words should not exceeds the vocabulary size.')

    def _create_vector(self, document, top_words):
        """
        This helper function takes a document and a sequence of top words, produce bag of words vector.

        :param document: A document string.
        :param top_words: A sequence of strings.
        :return: A list of 0 and 1.
        """
        document = document.lower()
        document = re.sub(r"\w", " ", document)
        document = re.sub(r"\s+", " ", document)

        bow_vector = []
        tokens = self.tokenizer.tokenize(document)
        for word in top_words:
            if word in tokens:
                bow_vector.append(1)
            else:
                bow_vector.append(0)
        return bow_vector

    def bag_of_words(self, path_to_corpus, n_of_column: int, n_of_words: int):
        """
        This method returns the bag-of-words representation of a document.

        :param path_to_corpus: A path to corpus file.
        :param n_of_column: An integer indicating the column that contains a document.
        :param n_of_words: An integer indicating the number of top words that willed be selected from all words.
        :return: A 2D ``numpy.ndarray`` of bag of words matrix.
        """
        with open(path_to_corpus, 'r') as file:
            word2count = {}
            for line in csv.reader(file):
                line[n_of_column-1] = line[n_of_column-1].lower()
                line[n_of_column-1] = re.sub(r"\w", " ", line[n_of_column-1])
                line[n_of_column-1] = re.sub(r"\s+", " ", line[n_of_column-1])

                tokens = self.tokenizer.tokenize(line[n_of_column-1])
                for token in tokens:
                    if token not in word2count:
                        word2count[token] = 1
                    else:
                        word2count[token] += 1

            # filter the top n most frequent words
            top_words = self._top_words(word2count, n_of_words)

            # create bow matrix
            bow_matrix = []
            for line in csv.reader(file):
                bow_matrix.append(self._create_vector(line[n_of_column - 1], top_words))
        return numpy.array(bow_matrix)

    def tf_idf(self):