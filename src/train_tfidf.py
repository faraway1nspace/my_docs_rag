import logging
import os
import pickle
import re
from datasets import load_dataset

from typing import List, Tuple

from src.config import SentencePieceConfig, TrainingConfig
from src.retriever import SentencePieceTokenizer, TFIDF

logging.getLogger().setLevel(logging.INFO)


TEST_STRING = "The ideal candidate will have a background in business, tax, and legal contracts."

def train_tfidf(config: TrainingConfig = TrainingConfig()):
    """Trains the TFIDF Vectorizer and SentencePiece tokenizer."""

    # save locally to reload through SP
    corpus, path_corpus_text_for_sp = make_tfidf_training_set(config)
    
    # Train SentencePiece tokenizer (and save locally)
    tokenizer = SentencePieceTokenizer.train(
        path_sp_training = path_corpus_text_for_sp,  
        config = SentencePieceConfig()
    )

    # test the tokenizer
    logging.info(f"""=== Test tokenizer: {tokenizer.tokenize(TEST_STRING)} ===""")

    # instantiate TFIDF retriever for training
    tfidf = TFIDF(
        tokenizer=tokenizer,
        config=config.tfidf
    )

    # train the tfidf vectorizer
    tfidf.train(corpus)

    # save tfidf artfiact locally
    tfidf.save()

    # try reloading and testing
    tfidf2 = TFIDF.load(config)

    # ensure reloading works
    diff = tfidf.vectorize([TEST_STRING])-tfidf2.vectorize([TEST_STRING])
    assert diff.sum() < 0.00000001

    # demo the retrieval
    logging.info('=== Passed test: training and reloading TFIDF plus tokenizer.===')


def make_tfidf_training_set(config: TrainingConfig) -> Tuple[List[str],str]:
    """Downloads a HF dataset and saves text to a SentencePiece dataset"""
    # preprocessing
    preproces = SentencePieceTokenizer()._preprocess

    # corpus to return
    corpus:List[str] = []

    with open(config.sp.sp_train_file, 'w') as filecon:

        for dataset_pointer in config.dataset_name.split(" "):
            
            # process pointer to get column names and the dataset_name
            columns = re.search(r"\{.*?\}",dataset_pointer)
            if columns:
                coltext = columns.group()
                columns = coltext[1:-1].split(",")
                dataset_name = dataset_pointer.replace(coltext,"")
            else:
                dataset_name = dataset_pointer
                columns = ["title","text"] # defaults
                logging.warning(f'No columns in .env datassets name: using default columns {columns}')

            logging.info(f'=== Downloading HF dataset for training tokenizer {dataset_name} ===')
            data = load_dataset(dataset_name, split='train')
            for row in data:
                # concatenate the title and text
                doc_text = " ".join([str(row[column]) for column in columns])
                corpus.append(doc_text)

                # preprocess the text for sentencepiece
                text_cleaned = preproces(doc_text)

                # write to file for sentencepiece training
                filecon.write(text_cleaned + "\n")

    # return for other processes, and the sentencepiece file
    logging.info(f"=== Wrote SP training dataset to {config.sp.sp_train_file} ===")
    return corpus, config.sp.sp_train_file 
    
            



        

      
    
    

    
    
