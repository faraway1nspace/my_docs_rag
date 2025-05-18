import os

from dotenv import load_dotenv
from pydantic import BaseModel


VOCAB_SIZE = 5000

PATH_TFIDF = "../resources/tfidf"

if not os.path.isdir(PATH_TFIDF):
    os.makedirs(PATH_TFIDF, exist_ok=True)

class SentencePieceConfig(BaseModel):
    model_prefix: str = f"{PATH_TFIDF}/sp"
    vocab_size: int = VOCAB_SIZE
    character_coverage: float = 0.9995


class TrainingConfig(BaseModel):
    """Configuration for training."""
    dataset_name: str = os.environ['TRAIN_DATASET_NAME']
    vocab_size: int = 5000
    sp: SentencePieceConfig = SentencePieceConfig()
    sp_train_file: str = f"{PATH_TFIDF}/corpus_for_sp.txt"

