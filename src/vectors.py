import os
import pickle

import numpy as np

from pandas import Series as pd_Series
from typing import Dict, List, Literal, Tuple, Union

from sklearn.metrics.pairwise import cosine_similarity


class DocVector:
    """A vector representation of a document."""
    filename: str | None = ""
    path:str | None = ""    
    text:str | None = None
    vector: np.ndarray | None = None
    score: int | None = None

    def __init__(
            self,
            filename: str,
            vector: np.ndarray,
            path: str,
            text: str | None = None,
            score: int | None = None
    ):
        self.filename = filename
        self.vector = vector
        self.path = path
        self.text = text
        self.score = score

    def to_dict(self):
        return {
            "filename": self.filename,
            "path": self.path,     
            "text": self.text,                 
            "vector": self.vector.tolist(),
            "score": self.score
        }

    @classmethod
    def load(cls, a:dict)->"DocVector":
        return DocVector(
            filename=a["filename"],
            path=a["path"],   
            text=a.get("text",""),
            vector=np.array(a["vector"]),
            score=a.get('score',None)
        )


class VectorDataset:
    """A collection of document vectors to facilitate query retrieval."""
    docs: List[DocVector] = []

    def __init__(
            self, docs: List[DocVector] = []
    ):
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx: Union[str, int]) -> DocVector:
        """Get item key can be an int or filename."""
        if isinstance(idx, int):
            return self.docs[idx]
        elif isinstance(idx, str):
            idx_doc = [doc for doc in self.docs if doc.filename == idx]
            if len(idx_doc)==1:
                return idx_doc[0]
            raise ValueError(f"{len(idx_doc)} returned for key {idx}")
        raise NotImplementedError(f"No items for {idx}")

    def append(self, doc: DocVector):
        """Add a doc vector to the vector dataset"""
        self.docs.append(doc)

    def array(self) -> np.ndarray:
        """Combine all document vectors into a single matrix."""
        return np.array([doc.vector for doc in self.docs])

    def score(
            self,
            query_vector: np.ndarray,
            return_type: Literal['doc','dict','list','pandas'] = 'dict'
    ) -> Union[List[DocVector], Dict[str, float], List[float], pd_Series]:
        """Given a query vector, compute cosine similarity of docs."""
        # Combine all document vectors into a matrix
        m = self.array()

        # compute the cosine similarity between query vector and m
        similarities = cosine_similarity(query_vector, m).flatten()

        # set the scores on the DocVector objects
        for doc, score in zip(self.docs, similarities):
            doc.score = score
        
        if return_type in ["doc","docs"]:
            return self.docs
        
        # for other return types, collect scores as dictionary, then convert
        scores = {
            doc.filename: float(score) for doc, score in zip(self.docs, similarities)
        }        
        if return_type=='list':
            return list(scores.values())
        if return_type=='pandas':
            return pd_Series(scores)
        return scores

    def save(self, path:str):
        """Save the vector database to a pickle file."""
        with open(path, 'wb') as pcon:
            for doc in self.docs:
                pickle.dump(doc.to_dict(), pcon)

    @property
    def text_to_embed(self) -> Tuple[List[str], List[int]]:
        """Which docs have a null vector and need embedding."""
        indices_to_embed = [i for i,doc in enumerate(self.docs) if doc.vector is None]
        text_to_embed = [self.docs[i].text for i in indices_to_embed]
        return text_to_embed, indices_to_embed

    def insert_embeddings(self, embeddings: np.ndarray, indices: List[int]) -> None:
        """Insert embeddings into doc's vectors at indicies."""
        for i in indices:
            self.docs[i].vector = embeddings[i]

    @property
    def filenames(self) -> List[str]:
        """Return a list of filenames in the vector database."""
        return [doc.filename for doc in self.docs]

    @classmethod
    def load(cls, path:str) -> 'VectorDataset':
        """Load the vector database from a pickle file."""
        docs:List[DocVector] = []
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                while True:
                    try:
                        docs.append(DocVector.load(pickle.load(f)))
                    except EOFError:
                        break
        return cls(docs=docs)

