import os

MODELS_DIR = os.path.expanduser("~/.cache/wellcomeml/models")
CACHE_DIR = os.path.expanduser("~/.cache/wellcomeml/cache")

MODEL_DISPATCH = {
    'scibert_scivocab_uncased': {
        "bucket": "ai2-s2-research",
        "path": "scibert/huggingface_pytorch/scibert_scivocab_uncased.tar",
        "file_name": "scibert_scivocab_uncased.tar"
    }
}
