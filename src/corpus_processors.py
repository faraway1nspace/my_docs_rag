
import hashlib
import docx
import glob

import logging
import numpy as np
import os
import pandas as pd
import pickle
import PyPDF2

from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Literal, Tuple, Union
from pathlib import Path

from src.retriever import TFIDF, SBERT
from src.config import TrainingConfig, RunConfig
from src.vectors import DocVector, VectorDataset

logging.getLogger().setLevel(logging.INFO)

class BaseCorpusProcessor:
    """Preprocesses local files, extract texts, and vectorize for search."""
    def __init__(
        self, 
        database_path: str | None = None, 
        run_config: RunConfig = RunConfig(),
        do_reload:bool = True
    ):      
        self.run_config = run_config
        if database_path is None:
            database_path = self.run_config.path_databases
        
        self.database_path = database_path
        if do_reload:
            # reload the vector dataset
            self.database: VectorDataset = self.load_database(database_path)
        else:
            # instantiate an empty vector database
            self.database: VectorDataset = self.load_database("")

    def __len__(self) -> int:
        return len(self.database.docs)

    def __getitem__(self, idx: Union[str, int]) -> DocVector:
        """Get item key can be an int or filename."""
        return self.database[idx]

    def load_database(self, path: str | None) -> VectorDataset:
        """Load a vectorized corpus of documents."""
        if path is None:
            path = self.database_path
        if os.path.isfile(path):
            vector_database = VectorDataset.load(path)
            print(f"Loaded {path} vector database with {len(vector_database)} documents.")
            return vector_database
        logging.info(f'Empty dataset: no file existing {path}')
        return VectorDataset(docs=[])

    def save_database(self, path: str | None = None):
        """Save the vector database to a pickle file."""
        if path is None:
            path = self.database_path        
        self.database.save(path)
        logging.info(f"=== saved vector database {path} ==")

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
        doc_text = '\n'.join([para.text for para in doc.paragraphs])
        if not doc_text.strip():
            return ""
        return doc_text

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

    def score(
            self, 
            query: str, 
            return_type: Literal['dict','list','pandas'] = 'dict'
        ) -> Union[Dict[str, float], List[float], pd.Series]:
        """Calc cosine_similarity score for query-text across corpus-database."""
        raise NotImplementedError("Subclasses must implement this method.")

    def vectorize_corpus(
        self, 
        input: Union[str, Path, List[str], None] = None, 
        do_save:bool = True
    ) -> None:
        """Vectorizes the inputs, which can be local files or a list of text."""
        if input is None:
            # vectorize the corpus according to self.config directory of files
            self._vectorize_directory(
                directory=self.run_config.docs_path, do_save=do_save
            )
            return
        
        if (isinstance(input, str) or isinstance(input, Path)) and os.path.isdir(input):
            # vectorize the corpus in directory as specified by Input
            self._vectorize_directory(directory=input, do_save=do_save)
            return

        if isinstance(input, list):
            if all([os.path.isfile(x) for x in input]):
                # inputs are paths to files
                self._vectorize_files(doc_paths=input, do_save=do_save) 
                return
            
            elif all([isinstance(x,str) for x in input]):
                self._vectorize_texts(texts=input, do_save=False)
                return
        
        raise NotImplementedError(
            (
                f"No method for input type {type(input)}."
                " Check whether files exist, directory exists "
                " or pass in a list of document-texts"
            )
        )

    def _vectorize_directory(self, directory: str | None = None, do_save:bool = True):
        """Extract text from a file, supporting docx, pdf, and txt formats."""
        if directory is None:
            directory = self.run_config.docs_path
        files_to_load = sorted(glob.glob(os.path.join(directory, "*")))
        files_to_load = [
            f for f in files_to_load
            if (
                f.lower().endswith(".pdf") or
                f.lower().endswith(".docx") or
                f.lower().endswith(".txt")
            )
        ]
        if not files_to_load:
            raise NotImplementedError(f"{directory} has no files of type pdf, docx, txt")
        logging.info(f"Grabbed files in directory {directory}")
        self._vectorize_files(docs_paths = files_to_load)

    def _vectorize_texts(self, texts: List[str], do_save:bool = True):
        """Extract text from a file, supporting docx, pdf, and txt formats."""
        for i,text in enumerate(texts):
            # filename acts as doc id
            filename_fake = "file_{i}.txt"
            self.database.append(
                DocVector(
                    filename=filename_fake,
                    vector=None, # will be given vector below
                    path="",
                    text=text
                )
            )
        text_to_embed, indices_to_embed = self.database.text_to_embed
        # embed new text (child method)
        embeddings = self.embed_texts(text_to_embed)
        # insert embeddings into docs
        self.database.insert_embeddings(embeddings, indices_to_embed) 
        if do_save and self.database_path:
            self.save_database()    
        logging.info(f"Added {len(texts)} to database on-the-fly from inputs texts")   

    def _vectorize_files(self, docs_paths: List[Union[str,Path]], do_save:bool = True):
        """Extract text from a file, supporting docx, pdf, and txt formats."""
        files_to_load = sorted(docs_paths)
        found_new_paths:List[str] = [] # new files
        files_in_corpus = self.database.filenames # files already in corpus
        for filepath in files_to_load:
            filename = filepath.split('/')[-1]
            # only extract text from new files
            if filename not in files_in_corpus:
                logging.info(f"Extract text from new file: {filepath}")
                text = self.extract_text(filepath)
                if text:
                    self.database.append(
                        DocVector(
                            filename=filename,
                            vector=None, # will be given vector below
                            path=filepath,
                            text=text
                        )
                    )
                    found_new_paths.append(filepath)
                else:
                    logging.warning(f"!!! No text extracted from {filepath}")

        if found_new_paths:
            logging.info(f"Extracted text from {len(found_new_paths)} new files: {found_new_paths}")

            # for new docs, get their text that require embedding
            text_to_embed, indices_to_embed = self.database.text_to_embed

            # embed new text (child method)
            embeddings = self.embed_texts(text_to_embed)
            
            # insert embeddings into docs
            self.database.insert_embeddings(embeddings, indices_to_embed)

            # pickle the database
            if do_save and self.database_path:
                self.save_database()
            
        else:
            logging.info("No new files to vectorize.")               


class TFIDFCorpusProcessor(BaseCorpusProcessor):
    """Wrapper to vectorize a local directory of files by TFIDF."""

    def __init__(
            self, 
            retriever: TFIDF, 
            path_database: str,
            run_config: RunConfig = RunConfig(),
            do_reload: bool = True
        ):
        super().__init__(database_path=path_database, run_config=run_config)
        self.retriever = retriever

    def score(
            self, 
            query: str, 
            return_type: Literal['dict','list','pandas'] = 'dict'
    ) -> Union[List[DocVector], Dict[str, float], List[float], pd.Series]:
        """Calc cosine_similarity score for query-text across corpus-database."""
        query_vector = self.retriever.vectorize_query(query)
        return self.database.score(query_vector, return_type)     
        
    def embed_texts(self, texts:List[str]) -> np.ndarray:
        """Vectorize the list of texts using TFIDF."""
        return self.retriever.vectorize(texts)


class SBERTCorpusProcessor(BaseCorpusProcessor):
    """Wrapper to vectorize a local directory of files by TFIDF."""

    def __init__(
            self, 
            retriever: SBERT, 
            path_database: str,
            run_config: RunConfig = RunConfig(),
            do_reload: bool = True
        ):
        super().__init__(database_path=path_database, run_config=run_config)
        self.retriever = retriever  
    
    def score(
            self, 
            query: str, 
            return_type: Literal['dict','list','pandas'] = 'dict'
    ) -> Union[List[DocVector], Dict[str, float], List[float], pd.Series]:
        """Calc cosine_similarity score for query-text across corpus-database."""
        query_vector = self.retriever.vectorize_query(query)
        return self.database.score(query_vector, return_type)   

    def embed_texts(self, texts:List[str]) -> np.ndarray:
        """Vectorize the list of texts using TFIDF."""
        return self.retriever.vectorize(texts = texts)

