import os
import pickle

import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Tuple, Union

from sklearn.metrics.pairwise import cosine_similarity


class DocVector:
    """A vector representation of a document."""
    filename: str | None = ""
    path:str | None = ""    
    text:str | None = None
    vector: np.ndarray | None = None

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
            "path": self.path,     
            "text": self.text,                 
            "vector": self.vector.tolist(),
        }

    @classmethod
    def load(cls, a:dict)->"DocVector":
        return DocVector(
            filename=a["filename"],
            path=a["path"],   
            text=a["text"],
            vector=np.array(a["vector"]),
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
        ) -> Union[Dict[str, float], List[float], pd.Series]:
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
        vector_database = VectorDataset()
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                while True:
                    try:
                        vector_database.append(DocVector.load(pickle.load(f)))
                    except EOFError:
                        break
        return vector_database

