from pathlib import Path
import requests
from huggingface_hub import HfApi
from tqdm import tqdm

def download_file(url: str, local_filename: Path) -> None:
    """Download a file from a given URL to a local file."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
            desc=local_filename.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                progress_bar.update(size)

def get_dataset_files(repo_id: str) -> list:
    """Get a list of dataset files from the HuggingFace repository."""
    api = HfApi()
    return [file for file in api.list_repo_files(repo_id, repo_type="dataset") 
            if file.endswith(('.csv', '.json'))]

def dataset_exists(output_dir: Path, files: list) -> bool:
    """Check if all dataset files already exist in the output directory."""
    return all((output_dir / Path(file).name).exists() for file in files)

def download_dataset_files(repo_id: str, output_dir: Path = Path("Input_Dataset")) -> None:
    """Download dataset files if they don't already exist."""
    try:
        files = get_dataset_files(repo_id)
        
        if not files:
            print(f"No compatible files found in the repository: {repo_id}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        
        if dataset_exists(output_dir, files):
            print(f"Dataset already exists in '{output_dir.resolve()}'. Skipping download.")
            return

        for file in files:
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file}"
            local_filename = output_dir / Path(file).name
            try:
                download_file(url, local_filename)
            except Exception as e:
                print(f"Error downloading {file}: {e}")
        
        print(f"Dataprep Done! Files saved at '{output_dir.resolve()}'.")
    except Exception as e:
        print(f"An error occurred while accessing the repository: {e}")

if __name__ == "__main__":
    download_dataset_files("sarthakdwi/EchoSwift-8k")