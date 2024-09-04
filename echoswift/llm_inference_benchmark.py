import os
import subprocess
import logging
from pathlib import Path
from typing import List
from tqdm import tqdm
import time
import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EchoSwift:
    def __init__(self, output_dir: str, api_url: str, provider: str, model_name: str = None,
                 max_requests: int = 5, user_counts: List[int] = [1],
                 input_tokens: List[int] = [32], output_tokens: List[int] = [256],
                 dataset_dir: str = "Input_Dataset"):
        self.output_dir = Path(output_dir)
        self.api_url = api_url
        self.provider = provider
        self.model_name = model_name
        self.max_requests = max_requests
        self.user_counts = user_counts
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.dataset_dir = Path(dataset_dir)

    def run_benchmark(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        locust_logs_dir = self.output_dir / "locust_logs"
        locust_logs_dir.mkdir(exist_ok=True)
        
        total_requests = sum(self.user_counts) * self.max_requests * len(self.input_tokens) * len(self.output_tokens)
        logging.info(f"Total requests to be sent: {total_requests}")

        for u in self.user_counts:
            user_dir = self.output_dir / f"{u}_User"
            user_dir.mkdir(exist_ok=True)

            for input_token in self.input_tokens:
                user_file = user_dir / f"{input_token}_input_tokens.csv"
                user_file.touch()

                for output_token in self.output_tokens:
                    logging.info(f"Running Locust with users={u}, input_tokens={input_token}, and output_tokens={output_token}")
                    self._run_locust(u, input_token, output_token, user_file, locust_logs_dir)

                self._calculate_average(user_dir, input_token)

    def _run_locust(self, users: int, input_tokens: int, output_tokens: int, output_file: Path, logs_dir: Path):
        env = os.environ.copy()
        env.update({
            "MAX_REQUESTS": str(self.max_requests),
            "NUM_USERS": str(users),
            "MAX_NEW_TOKENS": str(output_tokens),
            "API_URL": self.api_url,
            "PROVIDER": self.provider,
            "INPUT_DATASET": str(self.dataset_dir / f"Dataset_{input_tokens}.csv"),
            "OUTPUT_FILE": str(output_file)
        })

        if self.provider in ["Ollama", "vLLM"]:
            env["MODEL_NAME"] = self.model_name

        command = [
            "locust",
            "-f", "echoswift/llm_inference_master.py",
            "--headless",
            "-H", self.api_url,
            "-u", str(users),
            "-r", str(users)
        ]

        log_file_path = logs_dir / f"locust_log_u{users}_in{input_tokens}_out{output_tokens}.log"
        
        total_requests = users * self.max_requests
        with tqdm(total=total_requests, desc=f"Requests (u={users}, in={input_tokens}, out={output_tokens})", leave=True) as pbar, \
             open(log_file_path, 'w') as log_file:
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
            
            generated_text_count = 0
            for line in iter(process.stdout.readline, ''):
                log_file.write(line)
                log_file.flush()
                
                if "Generated Text:" in line:
                    generated_text_count += 1
                    if generated_text_count % users == 0:
                        update_amount = users
                        pbar.update(update_amount)

                if pbar.n >= total_requests:
                    process.terminate()
                    break

            remaining = generated_text_count - pbar.n
            if remaining > 0:
                pbar.update(remaining)

            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logging.warning("Locust didn't terminate gracefully. Forcing termination.")
                process.kill()

        if process.returncode != 0 and process.returncode != -signal.SIGTERM.value:
            logging.error(f"Locust command failed with return code {process.returncode}. Check the log file: {log_file_path}")

    def _calculate_average(self, user_dir: Path, input_token: int):
        input_file = user_dir / f"{input_token}_input_tokens.csv"
        output_file = user_dir / f"avg_{input_token}_input_tokens.csv"
        
        command = [
            "python3",
            "echoswift/utils/avg_locust_results.py",
            "--input_csv_filename", str(input_file),
            "--output_csv_filename", str(output_file),
            "--tokens"
        ] + [str(t) for t in self.output_tokens]

        subprocess.run(command, check=True)

def run_echoswift(output_dir: str, api_url: str, provider: str, model_name: str = None,
                  max_requests: int = 5, user_counts: List[int] = [1],
                  input_tokens: List[int] = [32], output_tokens: List[int] = [256]):
    benchmark = EchoSwift(output_dir, api_url, provider, model_name,
                          max_requests, user_counts, input_tokens, output_tokens)
    benchmark.run_benchmark()

if __name__ == "__main__":
    # This part will be replaced by CLI commands
    run_echoswift("test_output", "http://localhost:8080/generate_stream", "TGI")