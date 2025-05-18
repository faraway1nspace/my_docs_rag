import os
import pickle
import re

import sentencepiece as spm

from pydantic import BaseModel
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer


class SentencePieceTokenizer(BaseModel):
    model_path: str

    def __init__(self, model_path: str | None):
        if model_path is not None:
            self.model_path = model_path
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)

    def tokenize(self, text: str) -> List[str]:
        return self.sp.encode_as_pieces(self._preprocess(text))

    @classmethod
    def _preprocess(text: str) -> str:
        # lowercase
        text = text.lower()
        # remove special char
        text = re.sub("[^\w]+", " ", text)
        return text


class TFIDFRetriever(BaseModel):
    vectorizer: TfidfVectorizer
    tokenizer: SentencePieceTokenizer

    def __init__(
            self, vectorizer: TfidfVectorizer, tokenizer: SentencePieceTokenizer
    ):
        self.vectorizer = vectorizer
        self.tokenizer = tokenizer

    def vectorize(self, text: str) -> List[float]:
        tokens = self.tokenizer.tokenize(text)
        return self.vectorizer.transform([' '.join(tokens)]).toarray()[0]

    def save(self, vectorizer_path: str, tokenizer_path: str):
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        self.tokenizer.sp.save(tokenizer_path)

    @classmethod
    def load(cls, vectorizer_path: str, tokenizer_path: str):
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        tokenizer = SentencePieceTokenizer(model_path=tokenizer_path)
        return cls(vectorizer=vectorizer, tokenizer=tokenizer)
