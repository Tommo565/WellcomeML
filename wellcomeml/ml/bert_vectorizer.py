# encoding: utf-8
"""
BERT Vectorizer that embeds text using a prertained BERT model
"""

import os
import tarfile

import boto3
from botocore import UNSIGNED
from botocore.client import Config

from transformers import BertModel, BertTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import torch

from wellcomeml.logger import logger


MODELS_DIR = os.path.expanduser("~/.cache/wellcomeml/models")
MODEL_DISPATCH = {
    'scibert_scivocab_uncased': {
        "bucket": "ai2-s2-research",
        "path": "scibert/huggingface_pytorch/scibert_scivocab_uncased.tar",
        "file_name": "scibert_scivocab_uncased.tar"
        }
}

class BertVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, pretrained='bert', sentence_embedding='mean_second_to_last'):
        self.pretrained = pretrained
        self.sentence_embedding = sentence_embedding

    def bert_embedding(self, x):
        # Max sequence length is 512 for BERT
        if len(x) > 512:
            embedded_a = self.bert_embedding(x[:512])
            embedded_b = self.bert_embedding(x[512:])
            return embedded_a + embedded_b

        tokenized_x = self.tokenizer.tokenize("[CLS] " + x + " [SEP]")
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_x)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensor = torch.zeros(tokens_tensor.shape, dtype=torch.long)
        with torch.no_grad():
            output = self.model(tokens_tensor, token_type_ids=segments_tensor)
        last_layer = output[2][-1]
        second_to_last_layer = output[2][-2]

        if self.sentence_embedding == 'mean_second_to_last':
            embedded_x = second_to_last_layer.mean(dim=1)
        elif self.sentence_embedding == 'mean_last':
            embedded_x = last_layer.mean(dim=1)
        elif self.sentence_embedding == 'sum_last':
            embedded_x = last_layer.sum(dim=1)
        else:
            # 'last_cls'
            embedded_x = last_layer[0,:]

        return embedded_x.cpu().numpy().flatten()

    def transform(self, X, *_):
        return np.array([self.bert_embedding(x) for x in X])

    def fit(self, *_):
        model_name = 'bert-base-uncased' if self.pretrained == 'bert' else 'scibert_scivocab_uncased'

        # If model_name doesn't exist checks cache and change name to
        # full path
        if model_name == 'scibert_scivocab_uncased':
            model_name = _check_cache_and_download(model_name)

        logger.info("Using {} embedding".format(model_name))
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model.eval()
        return self


def _check_cache_and_download(model_name):
    """ Checks if model_name is cached and return complete path"""
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        logger.info(f"Could not find model {model_name}. Downloading from S3")

        # The following allows to download from S3 without AWS credentials
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        tmp_file = os.path.join(MODELS_DIR, MODEL_DISPATCH[model_name]['file_name'])

        s3.download_file(MODEL_DISPATCH[model_name]['bucket'], MODEL_DISPATCH[model_name]['path'], tmp_file)

        tar = tarfile.open(tmp_file)
        tar.extractall(path=MODELS_DIR)
        tar.close()

        os.remove(tmp_file)

    return model_path