from wellcomeml.ml import CNNClassifier, KerasVectorizer
from sklearn.pipeline import Pipeline
import numpy as np


def test_vanilla():
    X = [
        "One",
        "One only",
        "Two nothing else",
        "Two and three"
    ]
    Y = np.array([0,0,1,1])

    model = Pipeline([
        ('vec', KerasVectorizer()),
        ('clf', CNNClassifier())
    ])
    model.fit(X, Y)
    assert model.score(X, Y) > 0.6

def test_multilabel():
    X = [
        "One and two",
        "One only",
        "Three and four, nothing else",
        "Two nothing else",
        "Two and three"
    ]
    Y = np.array([
        [1,1,0,0],
        [1,0,0,0],
        [0,0,1,1],
        [0,1,0,0],
        [0,1,1,0]
    ])
    model = Pipeline([
        ('vec', KerasVectorizer()),
        ('clf', CNNClassifier(multilabel=True))
    ])
    model.fit(X, Y)
    assert model.score(X, Y) > 0.4
    assert model.predict(X).shape == (5,4)
