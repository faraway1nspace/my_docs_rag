import os
import pickle
import re

import sentencepiece as spm

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict

from src.config import BertConfig, TFIDFConfig, TrainingConfig, SentencePieceConfig


class SentencePieceTokenizer:
    """Tokenizes text by BPE for TFIDF."""
    
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        if model_path is not None:
            self.model_path = model_path
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)

    def tokenize(self, text: str) -> List[str]:
        return self.sp.encode_as_pieces(self._preprocess(text))

    @staticmethod
    def _preprocess(text: str) -> str:
        # lowercase
        text = text.lower()
        # remove special char
        text = re.sub("[^\w]+", " ", text)
        return text

    @classmethod
    def train(
            cls, 
            path_sp_training: str | None = None,  
            config: SentencePieceConfig = SentencePieceConfig()
    ) -> 'SentencePieceTokenizer':
        """Train SentencePiece tokenizer (and save locally)."""

        if path_sp_training is None:
            path_sp_training = config.sp_train_file
        assert os.path.isfile(path_sp_training)

        # run the spm trainer
        spm.SentencePieceTrainer.train(
            input=path_sp_training,
            model_prefix=config.model_prefix,
            vocab_size=config.vocab_size,
            character_coverage=config.character_coverage,
            model_type='bpe'
        )

        # reload the sp model as a tokenizer
        tokenizer = cls.load(config=config)    

        return tokenizer

    @classmethod
    def load(cls, config: SentencePieceConfig = SentencePieceConfig()) -> 'SentencePieceTokenizer':
        return SentencePieceTokenizer(
            model_path=config.model_prefix + ".model"
        )


class TFIDF:
    """TFIDF vectorizer with SentencePiece tokenizer for retrieval."""
    def __init__(
            self, 
            vectorizer: TfidfVectorizer | None = None, 
            tokenizer: SentencePieceTokenizer | None = None,
            config:TFIDFConfig = TFIDFConfig()
    ):
        self.config = config
        # attach tfidf vectorizer (or re-initialize blank)
        self.vectorizer = (
            vectorizer 
            if (vectorizer is not None) 
            else TfidfVectorizer(
                ngram_range = config.ngram_range,
                max_df = config.max_df,
                min_df = config.min_df,
                max_features = config.max_features,
                norm = config.norm,
                use_idf = config.use_idf,
                smooth_idf = config.smooth_idf,
                sublinear_tf = config.sublinear_tf
            )
        )
        # attach the tokenizer (or reload if None)
        self.tokenizer = (
            tokenizer 
            if tokenizer is not None
            else SentencePieceTokenizer.load()
        )


    def tokenize(self, texts: List[str]) -> List[List[str]]:
        return [
            self.tokenizer.tokenize(text)
            for text in texts
        ] 

    def vectorize(self, docs: List[str]):
        doc_tokenized = self.tokenize(docs)
        doc_strings = [' '.join(tokens) for tokens in doc_tokenized]
        return self.vectorizer.transform(doc_strings).toarray()

    def train(
        self,
        docs:List[str],
        tokenizer: SentencePieceTokenizer | None = None,
    ) -> None:
        if tokenizer is not None:
            self.tokenizer = tokenizer
        
        if self.tokenizer is None:
            raise NotImplementedError('Need to supply tokenizer')

        # tokenize the corpus
        docs_tokenized = self.tokenize(docs)
        docs_tokenized = [d for d in docs_tokenized if d] # remove empty

        # reconstitute the docs into astrings
        doc_strings = [' '.join(tokens) for tokens in docs_tokenized]

        # train the tfidf vectorizer
        self.vectorizer.fit(doc_strings)

    def save(self, vectorizer_path: str|None=None):
        if vectorizer_path is None:
            vectorizer_path = self.config.model_path
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    @classmethod
    def load(cls, config:TrainingConfig = TrainingConfig()) -> 'TFIDF':
        vectorizer_path = config.tfidf.model_path
        assert os.path.isfile(vectorizer_path)

        # reload the sentencepiece tokenizer
        tokenizer = SentencePieceTokenizer.load(config.sp)

        # reload the tfidf vectorizer
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        return cls(
            vectorizer=vectorizer, 
            tokenizer=tokenizer,
            config=config.tfidf
        )


class SBERT:

    def __init__(
            self, 
            vectorizer: SentenceTransformer | None = None, 
            config:BertConfig = BertConfig()
    ):
        self.config = config
        # attach tfidf vectorizer (or re-initialize blank)
        self.vectorizer = (
            vectorizer 
            if (vectorizer is not None) 
            else self._download_sbert(config)
        )

    def _download_sbert(config:BertConfig = BertConfig()) -> SentenceTransformer:
        """Fetch model from Huggingface."""
        self.vectorizer = SentenceTransformer(config.model_name)

        