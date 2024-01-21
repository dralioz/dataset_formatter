from transformers import pipeline
from datasets import load_dataset

REPO_ID = 'lmsys/lmsys-chat-1m'
FILENAME = 'default'


data = load_dataset(REPO_ID, FILENAME)
data.save_to_disk('./data')