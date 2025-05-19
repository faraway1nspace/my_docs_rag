import os
import pickle
import hashlib
import glob
from typing import Dict, Tuple

import docx
import PyPDF2

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from src.retriever import TFIDF, SBERT
from src.config import TrainingConfig, RunConfig
from src.vectors import DocVector, VectorDataset


class BaseCorpusProcessor:
    def __init__(self, database_path: str | None = None, run_config: RunConfig = RunConfig()):      
        self.run_config = run_config
        if database_path is None:
            database_path = self.run_config.path_databases
        
        self.database_path = database_path
        # load the vector dataset
        self.database: VectorDataset = self.load_database(database_path)

    def load_database(self, path: str | None) -> VectorDataset:
        """Load a vectorized corpus of documents."""
        if path is None:
            path = self.database_path
        if os.path.isfile(path):
            vector_database = VectorDataset.load(path)
            print(f"Loaded {path} vector database with {len(vector_database)} documents.")
            return vector_database
        print(f'Empty dataset: no file existing {path}')
        return VectorDataset(docs=[])

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

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using a pre-trained model."""
        raise NotImplementedError("Subclasses must implement this method.")

    def vectorize_corpus(self, docs_path: str | None = None):
        """Extract text from a file, supporting docx, pdf, and txt formats."""
        if docs_path is None:
            docs_path = self.run_config.docs_path
        files_to_load = glob.glob(os.path.join(docs_path, "*"))
        found_new_paths:List[str] = [] # new files
        files_in_corpus = self.database.filenames # files already in corpus
        for filepath in files_to_load:
            filename = filepath.split('/')[-1]
            # only extract text from new files
            if filename not in files_in_corpus:
                print(f"Extract text from new file: {filepath}")
                text = self.extract_text(filepath)
                self.database.append(
                    DocVector(
                        filename=filename,
                        vector=None,
                        path=filepath,
                        text=text
                    )
                )
                found_new_paths.append(filepath)

        if found_new_paths:
            print(f"Extracted text from {len(found_new_paths)} new files: {found_new_paths}")

            # for new docs, get their text that require embedding
            text_to_embed, indices_to_embed = self.database.text_to_embed

            # embed new text (child method)
            embeddings = self.embed_texts(text_to_embed)
            
            # insert embeddings into docs
            self.database.insert_embeddings(embeddings, indices_to_embed)

            # pickle the database
            self.save_database()
            
        else:
            print("No new files to vectorize.")       



class TFIDFCorpusProcessor(BaseCorpusProcessor):
    """Wrapper to vectorize a local directory of files by TFIDF."""

    def __init__(
            self, 
            retriever: TFIDF, 
            path_database: str,
            run_config: RunConfig = RunConfig()
        ):
        super().__init__(database_path=path_database, run_config=run_config)
        self.retriever = retriever       
        
    def embed_texts(self, texts:List[str]) -> np.ndarray:
        """Vectorize the list of texts using TFIDF."""
        embeddings = self.retriever.vectorize(texts)
        return embeddings


class SBERTCorpusProcessor(BaseCorpusProcessor):
    """Wrapper to vectorize a local directory of files by TFIDF."""

    def __init__(
            self, 
            retriever: SBERT, 
            path_database: str,
            run_config: RunConfig = RunConfig()
        ):
        super().__init__(database_path=path_database, run_config=run_config)
        self.retriever = retriever  

    def embed_texts(self, texts:List[str]) -> np.ndarray:
        """Vectorize the list of texts using TFIDF."""
        embeddings = self.retriever.vectorize(texts)
        return embeddings

