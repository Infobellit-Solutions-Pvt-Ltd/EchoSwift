import pandas as pd
from transformers import AutoTokenizer
import os
from datasets import load_dataset
import argparse


class DataFiltering:
    # Token length ranges for filtering
    TOKEN_LENGTH_RANGES = {
        "_32": (30, 40),
        "_64": (60, 70),
        "_128": (120, 135),
        "_256": (240, 260),
        "_512": (490, 530),
        "_1024": (1000, 1100),
        "_2048": (2000, 2100),
        "_4096": (4050, 4150)
    }

    def __init__(self, dataset, tokenizer_path_or_repo_id, base_folder):
        # Initialize DataFiltering instance
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_repo_id)
        self.dataset_name = dataset
        self.base_folder = base_folder
        self.no_of_prompts = 1000
        self.output_filename = "dataset_info.csv"

    def filter(self):
        # Load dataset and filter based on token length ranges
        data = load_dataset(self.dataset_name, split="train")
        filtered_dataset = {key: [] for key in self.TOKEN_LENGTH_RANGES.keys()}
        os.makedirs(self.base_folder, exist_ok=True)

        for text in data["prompt"]:
            tokens = self.tokenizer(text).input_ids

            for key, length_range in self.TOKEN_LENGTH_RANGES.items():
                if len(tokens) in range(*length_range):
                    filtered_dataset[key].append([text, len(tokens)])

        # Create DataFrames and save filtered datasets to CSV
        for key in filtered_dataset.keys():
            df = pd.DataFrame({
                "Input_Prompt": [row[0] for row in filtered_dataset[key][:self.no_of_prompts]],
                "No of Tokens": [row[1] for row in filtered_dataset[key][:self.no_of_prompts]],
            })

            output_file = os.path.join(self.base_folder, f'Dataset{key}.csv')
            df.to_csv(output_file, index=False)

    def extract_info(self):
        ''' Extracts the meta Information of the filtered dataset '''
        directory = self.base_folder
        details = {"filename": [], "min_tokens": [], "max_tokens": [], "Total_queries": [], "Range": []}
        outputfile = os.path.join(directory, self.output_filename)

        for filename in os.listdir(directory):
            if filename.lower().endswith('.csv'):
                filepath = os.path.join(directory, filename)

                try:
                    df = pd.read_csv(filepath)

                    if not df.empty:
                        prompts = df["Input_Prompt"].tolist()
                        tokens = df["No of Tokens"].tolist()

                        # Collect details for summary CSV
                        details["Total_queries"].append(len(prompts))
                        details["min_tokens"].append(min(tokens))
                        details["max_tokens"].append(max(tokens))
                        details["Range"].append((min(tokens), max(tokens)))
                        details["filename"].append(filename)
                except pd.errors.EmptyDataError:
                    print(f"Warning: Empty DataFrame in {filename}")

        # Save summary information to CSV
        new_df = pd.DataFrame(details)
        new_df.to_csv(outputfile, index=False)


if __name__ == "__main__":
    # Main execution
   
    dataset_name = "pvduy/sharegpt_alpaca_oa_vicuna_format"
    BASE_FOLDER = "Input_Dataset"
    tokenizer_path = "hf-internal-testing/llama-tokenizer"
    filter_tool = DataFiltering(dataset=dataset_name, tokenizer_path_or_repo_id=tokenizer_path, base_folder=BASE_FOLDER)

    filter_tool.filter()
    filter_tool.extract_info()
