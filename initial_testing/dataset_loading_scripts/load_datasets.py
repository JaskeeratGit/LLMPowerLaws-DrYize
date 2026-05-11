from datasets import load_dataset
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
#os.getenv("HF_TOKEN") UNCOMMENT THIS LINE AND ADD YOUR HF TOKEN

def load_and_save_dataset(dataset_url, split, save_path):

    dataset = load_dataset(dataset_url, split=split)
    dataset.save_to_disk(save_path)

    print(f"Dataset '{dataset_url}' saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    dataset_url = "suchievement/rp-chat-persona-sharegpt"  # Replace with your desired dataset URL
    split = None  # Replace with the desired split (e.g., 'train', 'test', 'validation')
    save_path = "datasets/rp-chat-persona-sharegpt_train"  # Replace with your desired save path
    
    load_and_save_dataset(dataset_url, split, save_path)
