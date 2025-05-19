import os
import pickle
from datasets import load_dataset

from typing import List, Tuple

from src.config import SentencePieceConfig, TrainingConfig
from src.retriever import SentencePieceTokenizer, TFIDF

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
    print(tokenizer.tokenize(TEST_STRING))

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
    print('Done training the TFIDF and tokenizer.')


def make_tfidf_training_set(config: TrainingConfig) -> Tuple[List[str],str]:
    """Downloads a HF dataset and saves text to a SentencePiece dataset"""
    
    data = load_dataset(config.dataset_name)

    # preprocessing
    preproces = SentencePieceTokenizer()._preprocess

    # corpus to return
    corpus:List[str] = []

    # convert the HF dataset into a text file for sentencepiece
    with open(config.sp.sp_train_file, 'w') as filecon:

        for title, text in zip(data.data['train']['title'], data.data['train']['description']):

            # concatenate the title and text
            doc_text = str(title) + " " + str(text)
            corpus.append(doc_text)

            # preprocess the text for sentencepiece
            text_cleaned = preproces(doc_text)

            # write to file for sentencepiece training
            filecon.write(text_cleaned + "\n")

    # return for other processes, and the sentencepiece file
    return corpus, config.sp.sp_train_file 
    
            



        

      
    
    

    
    
