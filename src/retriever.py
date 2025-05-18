import os
import pickle
import re

import sentencepiece as spm

from pydantic import BaseModel
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TFIDFConfig, TrainingConfig, SentencePieceConfig


class SentencePieceTokenizer:
    
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
    def load(cls, config: SentencePieceConfig = SentencePieceConfig()) -> 'SentencePieceTokenizer':
        return SentencePieceTokenizer(
            model_path=config.model_prefix + ".model"
        )


class TFIDFRetriever:

    def __init__(
            self, 
            vectorizer: TfidfVectorizer | None = None, 
            tokenizer: SentencePieceTokenizer | None = None,
            config:TFIDFConfig = TFIDFConfig()
    ):
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
        self.config = config

    def tokenize(self, texts: List[str]) -> List[List[str]]:
        return [
            self.tokenizer.tokenize(text)
            for text in texts
        ] 

    def vectorize(self, docs: List[str]):
        doc_tokenized = self.tokenize(docs)
        doc_strings = [' '.join(tokens) for tokens in docs]
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
    def load(cls, config:TrainingConfig = TrainingConfig()) -> 'TFIDFRetriever':
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
