from dotenv import load_dotenv
print(load_dotenv(".env"))

import os
assert os.environ.get("TRAIN_DATASET_NAME",None) is not None

from pydantic import BaseModel
from typing import Literal, Tuple


VOCAB_SIZE = 5000

PATH_TFIDF = "resources/tfidf"

if not os.path.isdir(PATH_TFIDF):
    os.makedirs(PATH_TFIDF, exist_ok=True)

class SentencePieceConfig(BaseModel):
    model_prefix: str = f"{PATH_TFIDF}/sp"
    vocab_size: int = VOCAB_SIZE
    character_coverage: float = 0.9995
    sp_train_file: str = f"{PATH_TFIDF}/corpus_for_sp.txt"


class TFIDFConfig(BaseModel):
    model_path: str = f"{PATH_TFIDF}/tfidf_model.pkl"
    ngram_range: Tuple[int,int] = (1, 2)
    max_df: float | int = 0.9 #When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float in range [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
    min_df: float | int = 0.005 # When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float in range of [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
    max_features: int | None = VOCAB_SIZE # If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used.
    norm: Literal["l1", "l2", None] = "l2"
    use_idf: bool = True
    smooth_idf: bool = True #Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.
    sublinear_tf: bool = False 


class TrainingConfig(BaseModel):
    """Configuration for training."""
    dataset_name: str = os.environ['TRAIN_DATASET_NAME']
    vocab_size: int = VOCAB_SIZE
    sp: SentencePieceConfig = SentencePieceConfig()
    tfidf: TFIDFConfig = TFIDFConfig()
   

