## Search class
import os
import logging
from typing import List, Literal
from sklearn.metrics.pairwise import cosine_similarity

from src.config import RunConfig
from src.corpus_processors import TFIDFCorpusProcessor, SBERTCorpusProcessor
from src.retriever import TFIDF, SBERT
from src.train_tfidf import train_tfidf
from src.vectors import DocVector

logging.getLogger().setLevel(logging.INFO)

class Search:
    """Wrapper for TFIDF and SBERT retrievers to search vectorized corpus by two methods."""

    def __init__(self, config: RunConfig = RunConfig()):
        self.config = config

        # initialize the TFIDF and SBERT retrievers
        self._initialize_retrievers()

        # initialize the document vectors for search
        self._initialize_document_database()

        assert 'tfidf_corpus_processor' in self.__dict__, "TFIDF corpus processor not initialized"
        assert 'sbert_corpus_processor' in self.__dict__, "SBERT corpus processor not initialized"
        assert len(self.tfidf_corpus_processor.database)>0, "TFIDF corpus processor has no documents"
        assert len(self.sbert_corpus_processor.database)>0, "SBERT corpus processor has no documents"

    def _initialize_retrievers(self) -> None:
        """Sets up the TFIDF and sBERT retrievers."""
        if self.config.tfidf.do_tfidf:

            # check if TFIDF vectorizer exists
            tfidf_exists = os.path.isfile(self.config.tfidf.model_path)
            sp_exists = os.path.isfile(self.config.training_config.sp.model_prefix + ".model")

            if (not tfidf_exists) or (not sp_exists):

                # do TFIDF training if neither exist already
                logging.info("=== Training TFIDF and SentencePiece vectorizer ===")
                train_tfidf(self.config.training_config)

            # load the TFIDF
            tfidf_retriever = TFIDF.load(self.config.training_config)
            logging.info("=== Loaded TFIDF retriever")
            self.tfidf_corpus_processor = TFIDFCorpusProcessor(
                retriever=tfidf_retriever,
                path_database=self.config.path_database_tfidf,
                run_config=self.config
            )
            logging.info("=== Loaded TFIDF corpus processor")


        if self.config.sbert.do_sbert:
            # load the SBERT
            sbert_retriever = SBERT.load(self.config.sbert)
            logging.info("=== Loaded SBERT retriever")
            self.sbert_corpus_processor = SBERTCorpusProcessor(
                retriever=sbert_retriever,
                path_database=self.config.path_database_sbert,
                run_config=self.config
            )
            logging.info("=== Loaded SBERT corpus procesor")

    def _initialize_document_database(self) -> None:
        """Vectorizes the local documents (reloading from cache if possible)"""
        if self.config.tfidf.do_tfidf:
            try:
                assert 'tfidf_corpus_processor' in self.__dict__, "TFIDF corpus processor not initialized"
            except:
                self._initialize_retrievers()

            self.tfidf_corpus_processor.vectorize_corpus()
            logging.info("=== Done getting TFIDF vectors from local documents.===")

        if self.config.sbert.do_sbert:
            try:
                assert 'sbert_corpus_processor' in self.__dict__, "SBERT corpus processor not initialized"
            except:
                self._initialize_retrievers()
            self.sbert_corpus_processor.vectorize_corpus()
            logging.info("=== Done getting SBERT vectors from local documents.===")

    def _filter_topk(self, sorted_docs: List[DocVector], k: int = 3, max_similarity: float = 0.95) -> List[DocVector]:
        """Filter out redundant results based on cosine similarity thresholds."""
        if len(sorted_docs) <= k:
            return sorted_docs
        top_k_results: List[DocVector] = []
        for candidate_doc in sorted_docs:
            if len(top_k_results) >= k:
                break
            is_redundant = False
            for prev_doc in top_k_results:
                similarity = cosine_similarity([prev_doc.vector], [candidate_doc.vector])[0][0]
                if similarity > max_similarity:
                    is_redundant = True
                    break
            if not is_redundant:
                top_k_results.append(candidate_doc)
        return top_k_results

    def _search_sparse(self, query: str, k: int = 3) -> List[str]:
        """Search the corpus using TFIDF and return top k non-similar results."""
        docs_scored = self.tfidf_corpus_processor.score(query, return_type='doc')
        docs_sorted = sorted(docs_scored, key = lambda x: x.score, reverse=True)
        # get top k docs and ensure they are not too similar
        top_k_docs = self._filter_topk(docs_sorted, k, self.config.max_similarity)
        return [doc.text for doc in top_k_docs]

    def _search_dense(self, query: str, k: int = 3) -> List[str]:
        """Search the corpus using SBERT and return top k non-similar results."""
        docs_scored = self.sbert_corpus_processor.score(query, return_type='doc')
        docs_sorted = sorted(docs_scored, key = lambda x: x.score, reverse=True)
        # get top k docs and ensure they are not too similar
        top_k_docs = self._filter_topk(docs_sorted, k, self.config.max_similarity)
        return [doc.text for doc in top_k_docs]

    def search(self, query: str, k: int = 3, method: Literal['sparse','dense','both']="sparse") -> List[str]:
        """Search the corpus using TFIDF and/or SBERT and return top k diverse non-similar results."""
        if method == 'sparse':
            return self._search_sparse(query, k)
        elif method == 'dense':
            return self._search_dense(query, k)
        raise NotImplementedError(f"Method '{method}' not implemented yet.")
