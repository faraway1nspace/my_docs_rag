import os
import pickle

import numpy as np
import pandas as pd
from typing import List, Literal

from sklearn.metrics.pairwise import cosine_similarity


class DocVector:
    """A vector representation of a document."""
    filename:str
    vector:np.ndarray
    path:str
    text:str | None = None

    def __init__(
            self, 
            filename: str, 
            vector: np.ndarray,
            path: str,
            text: str | None = None
    ):
        self.filename = filename
        self.vector = vector
        self.path = path
        self.text = text

    def to_dict(self):
        return {
            "filename": self.filename,
            "vector": self.vector.tolist(),
            "path": self.path,
            "text": self.text
        }
    
    @classmethod
    def load(cls, a:dict)->"DocVector":
        return DocVector(
            filename=a["filename"],
            vector=np.array(a["vector"]),
            path=a["path"],
            text=a["text"]
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
    
    def __getitem__(self, idx):
        return self.docs[idx]

    def append(self, doc: DocVector):
        """Add a doc vector to the vector dataset"""
        self.docs.append(doc)

    def array(self) -> np.ndarray:
        """Combine all document vectors into a single matrix."""
        return np.array([doc.vector for doc in self.docs])

    def score(
            self, 
            query_vector: np.ndarray,
            return_type: Literal['dict','list','pandas'] = 'dict'
        ) -> Dict[str, float]:
        # Combine all document vectors into a matrix
        m = self.array()

        # compute the cosine similarity between query vector and m
        similarities = cosine_similarity(query_vector, m).flatten()
        scores = {
            doc.filename: float(score) for doc, score in zip(self.docs, similarities)
        }
        if return_type=='list':
            return list(scores.values())
        elif return_type=='pandas':
            return pd.Series(scores)
        return scores

    def save(self, path:str):
        """Save the vector database to a pickle file."""
        with open(path, 'wb') as pcon:
            for doc in self.docs:
                pickle.dump(doc.to_dict(), pcon)

    @property
    def filenames(self) -> List[str]:
        """Return a list of filenames in the vector database."""
        return [doc.filename for doc in self.docs]
    
    @classmethod
    def load(cls, path:str) -> 'VectorDataset':
        """Load the vector database from a pickle file."""
        vector_database = VectorDataset()
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                while True:
                    try:
                        vector_database.append(DocVector.load(pickle.load(f)))
                    except EOFError:
                        break
        return vector_database
    
