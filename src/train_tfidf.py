import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
import sentencepiece as spm
from retrievers import TFIDFRetriever, SentencePieceTokenizer

from src.config import TrainingConfig

    
def train_tfidf(config: TrainingConfig):

    # Load dataset
    
    train_corpus = dataset['train'][config.text_column]

    # save locally to reload through SP
    fetch_train_corpus(dataset_name = config.dataset_name
    
    # Train SentencePiece tokenizer
    spm.SentencePieceTrainer.train(
        input=path_to_sp_train_file,
        model_prefix=config.sp.model_prefix,
        vocab_size=config.sp.vocab_size,
        character_coverage=config.sp.character_coverage,
        model_type='bpe'
    )
    tokenizer = SentencePieceTokenizer(model_path=f"{config.model_prefix}.model")

    # Tokenize texts
    tokenized_texts = [' '.join(tokenizer.tokenize(text)) for text in texts]

    # Train TFIDF vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(tokenized_texts)

    # Save artifacts
    retriever = TFIDFRetriever(vectorizer=vectorizer, tokenizer=tokenizer)
    retriever.save(config.vectorizer_path, config.tokenizer_path)


def fetch_train_corpus(dataset_name: str) -> str:
    
    dataset = load_dataset(dataset_name)

    # 
    
    

    
    
