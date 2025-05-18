import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv("../.env")

VOCAB_SIZE = 5000


class SentencePieceConfig(BaseModel):
    model_prefix: str = "../resources/tfidf/sp"
    vocab_size: int = VOCAB_SIZE
    character_coverage: float = 0.9995


class TrainingConfig(BaseModel):
    """Configuration for training."""
    dataset_name: str = os.environ['TRAIN_DATASET_NAME']
    vocab_size: int = 5000
    sp: SentencePieceConfig = SentencePieceConfig()

