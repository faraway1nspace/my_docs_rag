import os
import pickle
import hashlib
import glob
from typing import Dict, Tuple

import docx
import PyPDF2

import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.retriever import TFIDFRetriever
from src.config import TrainingConfig
from src.vectors import DocVector, VectorDataset

# Base path for documents
DOCS_PATH = "./docs"

PATH_DATABASES = "./database"
PATH_DATABASE_SBERT = f"{PATH_DATABASES}/sbert_vectors.pkl"
PATH_DATABASE_TFIDF = f"{PATH_DATABASES}/tfidf_vectors.pkl"

class BaseCorpusVectorizer:
    def __init__(self, database_path: str):
        if not os.path.isdir(PATH_DATABASES):
            os.makedirs(PATH_DATABASES,exist_ok=True)        
        self.database_path = database_path
        # load the vector dataset
        self.database: VectorDataset = self.load_database(database_path)

    def load_database(self, path: str | None) -> VectorDataset:
        """Load a vectorized corpus of documents."""
        if path is None:
            path = self.database_path
        vector_database = VectorDataset.load(path)
        return vector_database

    def save_database(self, path: str | None = None):
        """Save the vector database to a pickle file."""
        if path is None:
            path = self.database_path        
        self.database.save(path)

    def hash_file(self, filepath: str) -> str:
        """Generate a hash for a file based on its content."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def extract_text(self, filepath: str) -> str:
        """Extract text from a file, supporting docx, pdf, and txt formats."""
        if filepath.endswith(".docx"):
            return self.extract_text_from_docx(filepath)
        elif filepath.endswith(".pdf"):
            return self.extract_text_from_pdf(filepath)
        elif filepath.endswith(".txt"):
            return self.extract_text_from_txt(filepath)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")

    def extract_text_from_docx(self, filepath: str) -> str:
        """Extract text from a DOCX file."""
        doc = docx.Document(filepath)
        return '\n'.join([para.text for para in doc.paragraphs])

    def extract_text_from_pdf(self, filepath: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def extract_text_from_txt(self, filepath: str) -> str:
        """Extract text from a TXT file."""
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()

    def vectorize_corpus(self):
        """Abstract method for vectorizing the corpus, to be implemented by child classes."""
        raise NotImplementedError("This method should be implemented by subclasses.")



class TFIDFCorpusVectorizer(BaseCorpusVectorizer):
    """Wrapper to vectorize a local directory of files by TFIDF."""

    def __init__(self, retriever: TFIDFRetriever, path_database:str = PATH_DATABASE_TFIDF):
        super().__init__(database_path=path_database)
        self.retriever = retriever

    def vectorize_corpus(self, docs_path: str = DOCS_PATH):
        """Vectorize the documents using TFIDF and update the database."""

        files_to_vectorize = glob.glob(os.path.join(docs_path, "*"))
        new_files:List[str] = [] # new files
        files_in_corpus = self.database.filenames # files already in corpus
        for filepath in files_to_vectorize:
            filename = filepath.split('/')[-1]
            # only vectorize new files
            if filename not in files_in_corpus:
                print(f"Vectorizing new file: {filepath}")
                text = self.extract_text(filepath)
                vector = self.retriever.vectorize([text])[0]
                self.database.append(
                    DocVector(
                        filename=filepath.split('/')[-1],
                        vector=vector,
                        path=filepath,
                        text=text
                    )
                )
                new_files.append(filepath)

        if new_files:
            self.save_database()
            print(f"Updated TFIDF database with {len(new_files)} new files.")
        else:
            print("No new files to vectorize.")        


class SBERTCorpusVectorizer(BaseCorpusVectorizer):
    def __init__(self, model_name: str, path_database:str = PATH_DATABASE_SBERT):
        super().__init__(database_path=path_database)
        self.model = SentenceTransformer(model_name)

    def vectorize_corpus(self):
        """Vectorize the documents using SBERT and update the database."""
        files = glob.glob(os.path.join(DOCS_PATH, "*"))
        new_files = []

        for filepath in files:
            file_hash = self.hash_file(filepath)
            if file_hash not in self.database:
                print(f"Vectorizing new file: {filepath}")
                text = self.extract_text(filepath)
                vector = self.model.encode(text, convert_to_numpy=True).tolist()
                self.database[file_hash] = (filepath, vector)
                new_files.append(filepath)

        if new_files:
            self.save_database()
            print(f"Updated SBERT database with {len(new_files)} new files.")
        else:
            print("No new files to vectorize.")

# Usage
if __name__ == "__main__":
    config = TrainingConfig()
    
    # TFIDF Vectorization
    tfidf_retriever = TFIDFRetriever.load(config)
    tfidf_vectorizer = TFIDFCorpusVectorizer(tfidf_retriever)
    tfidf_vectorizer.vectorize_corpus()

    # SBERT Vectorization
    sbert_vectorizer = SBERTCorpusVectorizer(config.sbert_model_string)
    sbert_vectorizer.vectorize_corpus()
