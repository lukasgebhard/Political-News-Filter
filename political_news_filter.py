__author__ = 'Lukas Gebhard <freerunningapps@gmail.com>'

import doctest

import pandas as pd
import numpy as np
from keras.engine.saving import load_model
from keras.utils import np_utils
from keras_preprocessing.text import tokenizer_from_json
from keras_preprocessing import sequence


_POLITICAL_ARTICLE = '''White House declares war against terror. The US government officially announced a ''' \
                     '''large-scale military offensive against terrorism. Today, the Senate agreed to spend an ''' \
                     '''additional 300 billion dollars on the advancement of combat drones to be used against ''' \
                     '''global terrorism. Opposition members sharply criticize the government. ''' \
                     '''"War leads to fear and suffering. ''' \
                     '''Fear and suffering is the ideal breeding ground for terrorism. So talking about a ''' \
                     '''war against terror is cynical. It's actually a war supporting terror."'''
_NONPOLITICAL_ARTICLE = '''Table tennis world cup 2025 takes place in South Korea. ''' \
                        '''The 2025 world cup in table tennis will be hosted by South Korea, ''' \
                        '''the Table Tennis World Commitee announced yesterday. ''' \
                        '''Three-time world champion, Hu Ho Han, did not pass the qualification round, ''' \
                        '''to the advantage of underdog Bob Bobby who has been playing outstanding matches ''' \
                        '''in the National Table Tennis League this year.'''


def filter_news(news_articles, threshold=0.5):
    """
    Filter out all news articles that do not cover policy topics.

    # Arguments
        news_articles: A 1D NumPy array of news articles. A news article is the string concatenation of title,
            lead paragraph, and body.
        threshold: A value in [0, 1]. The higher the threshold, the more aggressive is the filter.
            The evaluation statistics (see `README.md`) are based on a threshold of 0.5.

    # Returns
        The filtered list of news articles.

    >>> assert _POLITICAL_ARTICLE == filter_news([_POLITICAL_ARTICLE, _NONPOLITICAL_ARTICLE])[0]
    """

    classifier = Classifier()
    estimations = classifier.estimate(news_articles)
    return [a for a, p in zip(news_articles, estimations) if p >= threshold]


class Classifier:
    """
    A machine learning classifier that estimates if an English news article covers policy topics.

    The classifier is based on Heng Zheng's convolutional neural network, published at
    <https://www.kaggle.com/hengzheng/news-category-classifier-val-acc-0-65?scriptVersionId=4623537>
    under the Apache 2.0 license <http://www.apache.org/licenses/LICENSE-2.0>.
    """

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._load()

    def _load(self):
        with open('./pon_classifier/tokenizer.json', 'r') as tokenizer_file:
            json = tokenizer_file.read()

        self._tokenizer = tokenizer_from_json(json)
        self._model = load_model('./pon_classifier/model.h5')

    @staticmethod
    def _as_array(tokens):
        return np.array(tokens.values.tolist())

    @staticmethod
    def _one_hot_encode(labels):
        return np_utils.to_categorical(labels.values, num_classes=2)

    def estimate(self, news_articles):
        """
        For each given news article, estimate if it covers policy topics.

        # Arguments
            news_articles: A 1D NumPy array of news articles. A news article is the string concatenation of title,
                lead paragraph, and body.

        # Returns
            The estimated probabilities as a list of length `len(news_articles)`.

        >>> classifier = Classifier()
        >>> estimations = classifier.estimate([_POLITICAL_ARTICLE, _NONPOLITICAL_ARTICLE])
        >>> estimations[0] > 0.99
        True
        >>> estimations[1] < 0.01
        True
        """

        to_estimate = EstimationSet(data=news_articles, tokenizer=self._tokenizer).get_data()
        tokens = to_estimate[EstimationSet.COL_TOKENS]
        estimations = self._model.predict(Classifier._as_array(tokens), batch_size=256)

        return [float(p) for p in list(estimations[:, 1])]


class EstimationSet:
    COL_TOKENS = 'TOKENS'
    _COL_TEXT = 'TEXT'

    def __init__(self, data, tokenizer):
        self._data = pd.DataFrame({EstimationSet._COL_TEXT: data})
        self._tokenizer = tokenizer
        self._preprocess()

    def get_data(self):
        return self._data

    def _preprocess(self):
        self._data[EstimationSet.COL_TOKENS] = self._tokenizer.texts_to_sequences(self._data[EstimationSet._COL_TEXT])
        self._data[EstimationSet.COL_TOKENS] = EstimationSet._pad_tokens(self._data[EstimationSet.COL_TOKENS], 1500)

    @staticmethod
    def _pad_tokens(tokens, padding_size):
        return pd.Series(list(sequence.pad_sequences(tokens, maxlen=padding_size)), index=tokens.index)


if __name__ == '__main__':
    doctest.testmod(raise_on_error=True)
