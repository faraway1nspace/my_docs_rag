"""Main interface to do retrieval on local (pickled) vector database."""

import os
import logging

from typing import List, Literal, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity

from src.config import RunConfig
from src.corpus_processors import TFIDFCorpusProcessor, SBERTCorpusProcessor
from src.retriever import TFIDF, SBERT
from src.train_tfidf import train_tfidf
from src.vectors import DocVector

logging.getLogger().setLevel(logging.INFO)


class Search:
    """Wrapper for TFIDF and SBERT retrievers to search vectorized corpus by three methods."""

    def __init__(self, config: RunConfig = RunConfig()):
        self.config = config

        # initialize the TFIDF and SBERT retrievers
        self._initialize_retrievers()

        # initialize the document vectors for search
        self._initialize_local_database()

        assert 'tfidf_corpus_processor' in self.__dict__, "TFIDF corpus processor not initialized"
        assert 'sbert_corpus_processor' in self.__dict__, "SBERT corpus processor not initialized"
        assert len(self.tfidf_corpus_processor.database)>0, "TFIDF corpus processor has no documents"
        assert len(self.sbert_corpus_processor.database)>0, "SBERT corpus processor has no documents"
        logging.info("=== Initialized Search API ===")

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

    def _initialize_local_database(self) -> None:
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

    def _initialize_database_from_texts(
            self, texts:List[str], method: Literal["sparse","dense"]
        )->Union[TFIDFCorpusProcessor, SBERTCorpusProcessor]:
        """Creates a vector database on the fly from input texts."""
        if method == "sparse":
            processor = TFIDFCorpusProcessor(
                retriever=self.tfidf_corpus_processor.retriever, 
                path_database="", 
                run_config=self.config
            )
        elif method == "dense":
            processor = SBERTCorpusProcessor(
                retriever=self.sbert_corpus_processor.retriever, 
                path_database="", 
                run_config=self.config
            )
        else:
            raise NotImplementedError(f"_initialize_database_from_texts no method: {method}")
        processor.vectorize_corpus(texts)
        logging.info(f"Created on the fly vector-corpus with {len(processor)} docs")            
        return processor         

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

    def _search_sparse(self, query: str, k: int = 3, corpus: Optional[List[str]]=[]) -> List[str]:
        """Search the corpus using TFIDF and return top k non-similar results."""
        if not corpus:
            logging.info("=== sparse search using attached databases === ")
            processor = self.tfidf_corpus_processor
        else:
            logging.info("=== making vector databases on the fly === ")
            processor = self._initialize_database_from_texts(corpus, method="sparse")
        
        docs_scored = processor.score(query, return_type='doc')
        docs_sorted = sorted(docs_scored, key = lambda x: x.score, reverse=True)
        # get top k docs and ensure they are not too similar
        top_k_docs = self._filter_topk(docs_sorted, k, self.config.max_similarity)
        return [doc.text for doc in top_k_docs]

    def _search_dense(self, query: str, k: int = 3, corpus: Optional[List[str]]=[]) -> List[str]:
        """Search the corpus using SBERT and return top k non-similar results."""
        if not corpus:
            logging.info("=== dense search using attached databases === ")
            processor = self.tfidf_corpus_processor
        else:
            logging.info("=== making vector databases on the fly === ")
            processor = self._initialize_database_from_texts(corpus, method="dense")
                
        docs_scored = processor.score(query, return_type='doc')
        docs_sorted = sorted(docs_scored, key = lambda x: x.score, reverse=True)
        # get top k docs and ensure they are not too similar
        top_k_docs = self._filter_topk(docs_sorted, k, self.config.max_similarity)
        return [doc.text for doc in top_k_docs]

    def _search_combined(self, query: str, k: int = 3, corpus: Optional[List[str]]=[]) -> List[str]:
        """Search the corpus using TFIDF and SBERT and return top k non-similar results."""
        def combine_scores(x:float, y:float, eps:float=0.0001) -> float:
            """Harmonic mean"""
            x+=eps
            y+=eps
            return 2*x*y / (x+y)
        
        if not corpus:
            logging.info("=== combined search using attached databases === ")
            tfidf_corpus_processor = self.tfidf_corpus_processor
            sbert_corpus_processor = self.sbert_corpus_processor
        else:
            logging.info("=== making vector databases on the fly === ")
            tfidf_corpus_processor = self._initialize_database_from_texts(corpus, method="sparse")
            sbert_corpus_processor = self._initialize_database_from_texts(corpus, method="dense")          

        docs_scored_1 = tfidf_corpus_processor.score(query, return_type='doc') # sparse doc vectors
        docs_scored_2 = sbert_corpus_processor.score(query, return_type='doc') # dense doc vectors

        # ensure the names of the files are the same between sparse & dense vectors
        assert [doc.filename for doc in docs_scored_1] == [doc.filename for doc in docs_scored_2], 'name mismatch in combined search'

        # combine the scores (like geometric mean -- but because of ranking we don't really care)
        scores_combined = {
            doc1.filename:combine_scores(doc1.score,doc2.score)
            for doc1,doc2 in zip(docs_scored_1,docs_scored_2)
        }

        # sort filenamesdescending highest scores
        filenames_sorted = sorted(scores_combined, key = lambda x: scores_combined[x],reverse=True)

        # ensure returned results are not redundant
        top_k_results: List[str] = []
        for candidate_filenm in filenames_sorted:
            if len(top_k_results) >= k:
                break
            is_redundant = False
            for prev_filenm in top_k_results:
                doc_a = tfidf_corpus_processor[candidate_filenm] # sparse vector candidate
                doc_b = sbert_corpus_processor[candidate_filenm] # dense vector candidate
                prev_doc_a = tfidf_corpus_processor[prev_filenm] # sparse vector previously selected
                prev_doc_b = sbert_corpus_processor[prev_filenm] # dense vector previously selected
                similarity_a = cosine_similarity([doc_a.vector], [prev_doc_a.vector])[0][0] # sparse similarity
                similarity_b = cosine_similarity([doc_b.vector], [prev_doc_b.vector])[0][0] # dense similarity
                # threshold is max_similarity squared...
                if combine_scores(similarity_a,similarity_b) > combine_scores(*[self.config.max_similarity]*2):
                    is_redundant = True
                    break

            if not is_redundant:
                top_k_results.append(candidate_filenm) # add candidate to top results to return

        # return the documents
        return [tfidf_corpus_processor[filename].text for filename in top_k_results]


    def search(
            self, 
            query: str, 
            k: int = 3, 
            method: Literal['sparse','dense','combined']="combined",
            corpus: Optional[List[str]]=[]
        ) -> List[str]:
        """Search the corpus using TFIDF and/or SBERT and return top k diverse non-similar results.
        
        Arguments:
            query: str, the query to used to search the corpus
            k: int, number of results to retrieve from corpus
            method: sparse==TFIDF, dense==sBERT, and combined
            corpus: optionally, we can vectorze text on the fly
                and create a new database, otherwise, use the local
                database that is attached and already vectorized.
        """
        if method == 'sparse':
            return self._search_sparse(query, k, corpus)
        elif method == 'dense':
            return self._search_dense(query, k, corpus)
        elif method == 'combined':
            return self._search_combined(query, k, corpus)
        raise NotImplementedError(f"Method '{method}' not implemented yet.")
