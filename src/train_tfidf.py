import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
import sentencepiece as spm
from src.retriever import SentencePieceTokenizer, TFIDFRetriever

from src.config import TrainingConfig


def train_tfidf(config: TrainingConfig = TrainingConfig()):

    # save locally to reload through SP
    make_tfidf_training_set(config)
    
    # Train SentencePiece tokenizer
    spm.SentencePieceTrainer.train(
        input=config.sp_train_file,
        model_prefix=config.sp.model_prefix,
        vocab_size=config.sp.vocab_size,
        character_coverage=config.sp.character_coverage,
        model_type='bpe'
    )

    # reload the sp model as a tokenizer
    tokenizer = SentencePieceTokenizer(model_path=f"{config.sp.model_prefix}.model")

    # test the tokenizer
    print(tokenizer.tokenize("The ideal candidate will have a background in business, tax, and legal contracts."))

    # Tokenize texts
    #tokenized_texts = [' '.join(tokenizer.tokenize(text)) for text in texts]

    # Train TFIDF vectorizer
    #vectorizer = TfidfVectorizer()
    #vectorizer.fit(tokenized_texts)

    # Save artifacts
    #retriever = TFIDFRetriever(vectorizer=vectorizer, tokenizer=tokenizer)
    #retriever.save(config.vectorizer_path, config.tokenizer_path)


def make_tfidf_training_set(config: TrainingConfig) -> None:
    """Downloads a HF dataset and saves text to a SentencePiece dataset"""
    
    data = load_dataset(config.dataset_name)

    # preprocessing
    preproces = SentencePieceTokenizer()._preprocess

    # convert the HF dataset into a text file for sentencepiece
    with open(config.sp_train_file, 'w') as filecon:

        for title, text in zip(data.data['train']['title'], data.data['train']['description']):

            # preprocess the text
            text_cleaned = preproces(str(title) + " " + str(text))
            # write to file
            filecon.write(text_cleaned + "\n")
    
            



        

      
    
    

    
    
