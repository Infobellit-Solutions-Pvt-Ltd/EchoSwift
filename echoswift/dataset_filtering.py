import pandas as pd
from transformers import AutoTokenizer
import os
from datasets import load_dataset
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatasetFilterer:
    TOKEN_LENGTH_RANGES = {
        "_32": (30, 40),
        "_64": (60, 70),
        "_128": (120, 135),
        "_256": (240, 260),
        "_512": (490, 530),
        "_1024": (1000, 1100),
        "_2048": (2000, 2100),
    }

    def __init__(self, dataset: str, tokenizer_path_or_repo_id: str, base_folder: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_repo_id)
        self.dataset_name = dataset
        self.base_folder = Path(base_folder)
        self.no_of_prompts = 1000
        self.output_filename = "dataset_info.csv"

    def filter(self):
        logging.info(f"Loading dataset: {self.dataset_name}")
        data = load_dataset(self.dataset_name, split="train")
        filtered_dataset = {key: [] for key in self.TOKEN_LENGTH_RANGES.keys()}
        if not self.base_folder.exists():
            self.base_folder.mkdir(parents=True, exist_ok=True)

            logging.info("Filtering prompts")
            for text in tqdm(data["prompt"], desc="Processing prompts"):
                tokens = self.tokenizer(text).input_ids
                for key, length_range in self.TOKEN_LENGTH_RANGES.items():
                    if len(tokens) in range(*length_range):
                        filtered_dataset[key].append([text, len(tokens)])

            logging.info("Saving filtered datasets")
            for key in tqdm(filtered_dataset.keys(), desc="Saving datasets"):
                df = pd.DataFrame({
                    "Input_Prompt": [row[0] for row in filtered_dataset[key][:self.no_of_prompts]],
                    "No of Tokens": [row[1] for row in filtered_dataset[key][:self.no_of_prompts]],
                })
                output_file = self.base_folder / f'Dataset{key}.csv'
                df.to_csv(output_file, index=False)
                logging.info(f"Saved {len(df)} prompts for token range {key}")
        else:
            logging.info(f"{self.base_folder} already exists")

    def extract_info(self):
        logging.info("Extracting dataset information")
        details = {"filename": [], "min_tokens": [], "max_tokens": [], "Total_queries": [], "Range": []}
        outputfile = self.base_folder / self.output_filename

        for filename in tqdm(list(self.base_folder.glob('*.csv')), desc="Processing files"):
            if filename.name == self.output_filename:
                continue
            try:
                df = pd.read_csv(filename)
                if not df.empty:
                    tokens = df["No of Tokens"].tolist()
                    details["Total_queries"].append(len(tokens))
                    details["min_tokens"].append(min(tokens))
                    details["max_tokens"].append(max(tokens))
                    details["Range"].append((min(tokens), max(tokens)))
                    details["filename"].append(filename.name)
            except pd.errors.EmptyDataError:
                logging.warning(f"Empty DataFrame in {filename}")

        new_df = pd.DataFrame(details)
        new_df.to_csv(outputfile, index=False)
        logging.info(f"Dataset information saved to {outputfile}")

def download_and_filter_dataset():
    dataset_name = "pvduy/sharegpt_alpaca_oa_vicuna_format"
    tokenizer_path = "hf-internal-testing/llama-tokenizer"
    base_folder = "Input_Dataset"
    
    filter_tool = DatasetFilterer(dataset=dataset_name, tokenizer_path_or_repo_id=tokenizer_path, base_folder=base_folder)
    
    logging.info("Starting dataset download and filtering process.....")
    filter_tool.filter()
    filter_tool.extract_info()
    logging.info(f"Filtered_ShareGPT_Dataset saved at {base_folder} successfully!!")

if __name__ == "__main__":
    download_and_filter_dataset()