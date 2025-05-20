from dotenv import load_dotenv
print(load_dotenv(".env"))

import os
assert os.environ.get("TRAIN_DATASET_NAME",None) is not None

from pydantic import BaseModel
from typing import Literal, Tuple


VOCAB_SIZE = 5000

PATH_TFIDF = "resources/tfidf"
DOCS_PATH = "./docs"
PATH_DATABASES = "./database"
PATH_DATABASE_SBERT = f"{PATH_DATABASES}/sbert_vectors.pkl"
PATH_DATABASE_TFIDF = f"{PATH_DATABASES}/tfidf_vectors.pkl"


for _dir in [PATH_TFIDF, DOCS_PATH, PATH_DATABASES]:
    if not os.path.isdir(_dir):
        os.makedirs(_dir, exist_ok=True)


class SentencePieceConfig(BaseModel):
    """Configuration for the SentencePiece tokenizer (for TFIDF)"""
    model_prefix: str = f"{PATH_TFIDF}/sp"
    vocab_size: int = VOCAB_SIZE
    character_coverage: float = 0.9995
    sp_train_file: str = f"{PATH_TFIDF}/corpus_for_sp.txt"


class TFIDFConfig(BaseModel):
    """Configuration for the TFIDF retriever."""
    do_tfidf: bool = True # whether or not to set-up TFIDF retriever
    model_path: str = f"{PATH_TFIDF}/tfidf_model.pkl"
    ngram_range: Tuple[int,int] = (1, 2)
    max_df: float | int = 0.9 
    min_df: float | int = 0.005 
    max_features: int | None = VOCAB_SIZE 
    norm: Literal["l1", "l2", None] = "l2"
    use_idf: bool = True
    smooth_idf: bool = True #Smooth idf weights by adding one to document frequencies
    sublinear_tf: bool = False 


class BertConfig(BaseModel):
    """Configuration for the SBERT retriever."""
    do_sbert: bool = True # whether or not to set-up sBERT retriever
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    prefix_doc: str = "search_document: "
    prefix_query: str = "search_query: "
    batch_size: int = 16


class TrainingConfig(BaseModel):
    """Configuration for training."""
    dataset_name: str = os.environ['TRAIN_DATASET_NAME']
    vocab_size: int = VOCAB_SIZE
    sp: SentencePieceConfig = SentencePieceConfig()
    tfidf: TFIDFConfig = TFIDFConfig()
   

class RunConfig(BaseModel):
    """Configuration at run-time"""
    docs_path: str = DOCS_PATH
    path_databases: str = PATH_DATABASES
    path_database_sbert: str = PATH_DATABASE_SBERT
    path_database_tfidf: str = PATH_DATABASE_TFIDF
    max_similarity: float = 0.95
    sbert: BertConfig = BertConfig()
    tfidf: TFIDFConfig = TFIDFConfig()
    training_config: TrainingConfig = TrainingConfig()
