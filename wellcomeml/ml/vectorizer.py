#!/usr/bin/env python3
# coding: utf-8

"""
A generic vectorizer that can fallback to tdidf or bag of words from sklearn
or embed using bert, doc2vec etc
"""
import os
from collections import defaultdict
from hashlib import sha256

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from wellcomeml.ml.frequency_vectorizer import WellcomeTfidf
from wellcomeml.ml.bert_vectorizer import BertVectorizer
from wellcomeml.ml.constants import CACHE_DIR


class Vectorizer(BaseEstimator, TransformerMixin):
    """
    Abstract class, sklearn-compatible, that can vectorize texts using
    various models.

    """
    def __init__(self, embedding='tf-idf', **kwargs):
        """
        Args:
            embedding(str): One of `['bert', 'tf-idf']`
        """
        self.embedding = embedding

        vectorizer_dispatcher = {
            'tf-idf': WellcomeTfidf,
            'bert': BertVectorizer
        }

        if not vectorizer_dispatcher.get(embedding):
            raise ValueError(f'Model {embedding} not available')

        self.vectorizer = vectorizer_dispatcher.get(embedding)(**kwargs)

    def __hash__(self):
        """Hashes attributes of a vectorizer"""
        bytes_attributes = ' '.join(sorted(
            [attr + '_' + str(value)
             for attr, value in self.vectorizer.__dict__.items()]
        )).encode()

        return sha256(bytes_attributes)

    def fit(self, X=None, *_):
        return self.vectorizer.fit(X)

    def transform(self, X, use_cache=False, *_):
        # Load cache

        X_transformed = self.vectorizer.transform(X)

        # Save cache
        if use_cache:
            self._update_cache(X, X_transformed)

    def save_transformed(self, path, X_transformed):
        """
        Saves transformed vector X_transformed vector, using the corresponding
        save_transformed method for the specific vectorizer.

        Args:
            path: A path to the embedding file
            X_transformed: A transformed vector (as output by using the
            .transform method)

        """
        save_method = getattr(self.vectorizer.__class__, 'save_transformed',
                              None)
        if not save_method:
            raise NotImplementedError(f'Method save_transformed not implemented'
                                      f' for class '
                                      f'{self.vectorizer.__class__.__name__}')

        return save_method(path=path, X_transformed=X_transformed)

    def load_transformed(self, path):
        """
        Loads transformed vector X_transformed vector, using the corresponding
        load method for the specific vectorizer.

        Args:
            path: A path to the file containing embedded vectors

        Returns:
            X_transformed (array), like the one returned by the the
            fit_transform function.
        """
        load_method = getattr(self.vectorizer.__class__, 'load_transformed',
                              None)
        if not load_method:
            raise NotImplementedError(f'Method load_transformed not implemented'
                                      f' for class '
                                      f'{self.vectorizer.__class__.__name__}')

        return load_method(path=path)

    def _update_cache(self, X, X_transformed):

        model_cache_folder = os.path.join(CACHE_DIR, self.__hash__.hexdigest())

        os.makedirs(
            model_cache_folder,
            exist_ok=True
        )

        X_hashed = defaultdict(list)

        for i, text in enumerate(X):
            x_sha = sha256(text).hex_digest()
            X_hashed[x_sha[:3]] += [(x_sha, i)]

        for file_start, relationships in X_hashed.items():
            idx_file = os.path.join(model_cache_folder, f"{file_start}_idx")
            embedding_file = os.path.join(model_cache_folder, f"{file_start}_embedding")

            if os.path.exists(idx_file):
                X_cached = self.load_transformed(embedding_file)
                self.save_transformed(
                    path=embedding_file,
                    X_transformed=np.concatenate([X_cached, X_transformed])
                )

            else:
                with open(idx_file, 'w') as f:
                    f.write("\n".join(
                        [f"{x_sha},{i}" for x_sha, i in relationships]
                    ))

                X_subset = X_transformed[[i for _, i in relationships]]

                self.save_transformed(path=embedding_file,
                                      X_transformed=X_subset)


