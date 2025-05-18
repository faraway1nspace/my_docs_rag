from dotenv import load_dotenv

load_dotenv(".env")

import os
assert os.environ['TRAIN_DATASET_NAME']

from datasets import load_dataset

from src.config import TrainingConfig

config_train = TrainingConfig()
print(config_train)

load_dataset(config_train.dataset_name)
