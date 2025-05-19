import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
import sentencepiece as spm

from typing import List

from src.config import TrainingConfig
from src.retriever import SentencePieceTokenizer, TFIDFRetriever

TEST_STRING = "The ideal candidate will have a background in business, tax, and legal contracts."

def train_tfidf(config: TrainingConfig = TrainingConfig()):
    """Trains the TFIDF Vectorizer and SentencePiece tokenizer."""

    # save locally to reload through SP
    corpus = make_tfidf_training_set(config)
    
    # Train SentencePiece tokenizer (and save locally)
    spm.SentencePieceTrainer.train(
        input=config.sp.sp_train_file,
        model_prefix=config.sp.model_prefix,
        vocab_size=config.sp.vocab_size,
        character_coverage=config.sp.character_coverage,
        model_type='bpe'
    )

    # reload the sp model as a tokenizer
    tokenizer = SentencePieceTokenizer.load(config=config.sp)

    # test the tokenizer
    print(tokenizer.tokenize(TEST_STRING))

    # instantiate TFIDF retriever for training
    tfidf = TFIDFRetriever(
        tokenizer=tokenizer,
        config=config.tfidf
    )

    # train the tfidf vectorizer
    tfidf.train(corpus)

    # save tfidf artfiact locally
    tfidf.save()

    # try reloading and testing
    tfidf2 = TFIDFRetriever.load(config)

    # ensure reloading works
    diff = tfidf.vectorize([TEST_STRING])-tfidf2.vectorize([TEST_STRING])
    assert diff.sum() < 0.00000001

    # demo the retrieval
    print('Done training the TFIDF and tokenizer.')

    


def make_tfidf_training_set(config: TrainingConfig) -> List[str]:
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

    return corpus # return for other processes
    
            



        

      
    
    

    
    
