"""Methods to vectorize all docx and pdf files in local directory"""
# src/base_corpus_vectorizer.py
import os
import pickle
import hashlib
import glob
from typing import Dict, Tuple

import docx
import PyPDF2

from sentence_transformers import SentenceTransformer

from src.retriever import TFIDFRetriever
from src.config import TrainingConfig


# Base path for documents
DOCS_PATH = "./docs"

PATH_DATABASE_SBERT = "./database/sbert.pkl"
PATH_DATABASE_TFIDF = "./database/tfidf.pkl"

class BaseCorpusVectorizer:
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.database: Dict[str, Tuple[str, list]] = self.load_database()

    def load_database(self) -> Dict[str, Tuple[str, list]]:
        """Load the vector database from a pickle file."""
        if os.path.isfile(self.database_path):
            with open(self.database_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_database(self):
        """Save the vector database to a pickle file."""
        with open(self.database_path, 'wb') as f:
            pickle.dump(self.database, f)

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
    def __init__(self, retriever: TFIDFRetriever):
        super().__init__(database_path=PATH_DATABASE_TFIDF)
        self.retriever = retriever

    def vectorize_corpus(self):
        """Vectorize the documents using TFIDF and update the database."""
        files = glob.glob(os.path.join(DOCS_PATH, "*"))
        new_files = []

        for filepath in files:
            file_hash = self.hash_file(filepath)
            if file_hash not in self.database:
                print(f"Vectorizing new file: {filepath}")
                text = self.extract_text(filepath)
                vector = self.retriever.vectorize([text])[0]
                self.database[file_hash] = (filepath, vector)
                new_files.append(filepath)

        if new_files:
            self.save_database()
            print(f"Updated TFIDF database with {len(new_files)} new files.")
        else:
            print("No new files to vectorize.")


class SBERTCorpusVectorizer(BaseCorpusVectorizer):
    def __init__(self, model_name: str):
        super().__init__(database_path=PATH_DATABASE_SBERT)
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
