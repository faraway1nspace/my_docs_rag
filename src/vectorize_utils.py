"""Methods to vectorize all docx and pdf files in local directory"""

import os
import pickle
import hashlib
import glob
from typing import List, Dict, Tuple

import docx
import PyPDF2

from src.retriever import TFIDFRetriever
from src.config import TrainingConfig

# Directory paths
DOCS_PATH = "./docs"
DATABASE_PATH = "./database/tfidf.pkl"

# Ensure database directory exists
if not os.path.isdir(os.path.dirname(DATABASE_PATH)):
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)


class CorpusVectorizer:

    def __init__(self, retriever: TFIDFRetriever):
        self.retriever = retriever
        self.database: Dict[str, Tuple[str, List[float]]] = self.load_database()

    def load_database(self) -> Dict[str, Tuple[str, List[float]]]:
        """Load the vector database from a pickle file."""
        if os.path.isfile(DATABASE_PATH):
            with open(DATABASE_PATH, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_database(self):
        """Save the vector database to a pickle file."""
        with open(DATABASE_PATH, 'wb') as f:
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
        """Vectorize the documents in the specified directory and update the database."""
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
            print(f"Updated database with {len(new_files)} new files.")
        else:
            print("No new files to vectorize.")

# Usage
if __name__ == "__main__":
    config = TrainingConfig()
    retriever = TFIDFRetriever.load(config)
    corpus_vectorizer = CorpusVectorizer(retriever)
    corpus_vectorizer.vectorize_corpus()
