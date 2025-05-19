"""Search via tfidf and dense-retrieval of local documents (pdf, word)."""

import os
import logging

from src.train_tfidf import train_tfidf
from src.corpus_processors import TFIDFCorpusProcessor, SBERTCorpusProcessor
from src.config import RunConfig
from src.retriever import TFIDF, SBERT

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
            tfidf_exists = os.path.isfile(self.config.tfidf_config.model_path)
            sp_exists = os.path.isfile(self.config.training_config.sp.model_prefix + ".model")

            if (not tfidf_exists) or (not sp_exists):

                # do TFIDF training if neither exist already
                logging.info("=== Training TFIDF and SentencePiece vectorizer ===")
                train_tfidf(self.config)

            # load the TFIDF
            tfidf_retriever = TFIDF.load(self.config.tfidf_config)
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
            
    def _search_sparse(self, query: str, k: int) -> List[str]:
        """Search the corpus using TFIDF and return top k diverse results."""
        scores = self.tfidf_corpus_processor.score(query, return_type='dict')
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_k_docs = self._filter_redundant_results(sorted_docs, k, self.config.tfidf.max_similarity_threshold)
        return [self.tfidf_corpus_processor[doc_idx].text for doc_idx in top_k_docs]

    def _search_dense(self, query: str, k: int) -> List[str]:
        """Search the corpus using SBERT and return top k diverse results."""
        scores = self.sbert_corpus_processor.score(query, return_type='dict')
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_k_docs = self._filter_redundant_results(sorted_docs, k, self.config.sbert.max_similarity_threshold)
        return [self.sbert_corpus_processor[doc_idx].text for doc_idx in top_k_docs]

    def _filter_redundant_results(self, sorted_docs: List[Tuple[str, float]], k: int, max_similarity_threshold: float) -> List[int]:
        """Filter out redundant results based on cosine similarity thresholds."""
        heterogenous_results: List[int] = []
        for doc_idx, (filename, score) in enumerate(sorted_docs):
            if len(heterogenous_results) >= k:
                break
            is_redundant = False
            for prev_doc_idx in heterogenous_results:
                prev_vector = self.tfidf_corpus_processor[prev_doc_idx].vector
                current_vector = self.tfidf_corpus_processor[doc_idx].vector
                similarity = cosine_similarity([prev_vector], [current_vector])[0][0]
                if similarity > max_similarity_threshold:
                    is_redundant = True
                    break
            if not is_redundant:
                heterogenous_results.append(doc_idx)
        return heterogenous_results       
        

