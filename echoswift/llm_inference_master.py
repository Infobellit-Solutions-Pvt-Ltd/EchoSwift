import os
import csv
import time
import random
import logging
from datetime import datetime
from locust import HttpUser, task
from transformers import AutoTokenizer
from threading import Barrier, BrokenBarrierError
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Initialize the tokenizer for encoding/decoding text
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

# Global barrier to synchronize users
num_users = int(os.environ.get("NUM_USERS", 10))
barrier = Barrier(num_users)

class APITestUser(HttpUser):
    """
    Represents a Locust user for load testing an API.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the APITestUser instance.
        """
        super().__init__(*args, **kwargs)
        self.request_count = 0
        self.max_requests = int(os.environ.get("MAX_REQUESTS", 10))
        self.max_new_tokens = int(os.environ.get('MAX_NEW_TOKENS', 128))
        self.api_url = os.environ.get('API_URL', '')
        self.dataset_file = os.environ.get('INPUT_DATASET', '')
        self.questions = self.load_dataset(self.dataset_file)
        self.output_file_path = os.environ.get('OUTPUT_FILE', 'output.csv')
        self.provider = os.environ.get('PROVIDER', " ")
        self.model_name = os.environ.get('MODEL_NAME', " ")

    @staticmethod
    def load_dataset(csv_file):
        """
        Read questions from a CSV file.
        """
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            return [row['Input_Prompt'] for row in reader]

    def on_start(self):
        try:
            barrier.wait()
        except BrokenBarrierError:
            pass

    def format_prompt(self):
        """
        Format the prompt for the API request.
        """
        prompt = random.choice(self.questions)

        if self.provider == "TGI":
            data = {'inputs': prompt, 'parameters': {'max_new_tokens': self.max_new_tokens}}
        elif self.provider == "Ollama":
            data = {
                "model": self.model_name, 
                "prompt": prompt, 
                "stream": True, 
                "options": {"num_predict": self.max_new_tokens}
            }
        elif self.provider == "Llamacpp":
            data = {"prompt": prompt, "n_predict": self.max_new_tokens, "stream": True}

        elif self.provider == "vLLM":
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": self.max_new_tokens,
                "min_tokens": self.max_new_tokens,
                "stream": True
            }

        input_tokens = len(tokenizer.encode(prompt))
        return data, input_tokens

    def process_response(self, response):
        """
        Process the response from the API.
        """
        generated_text = ""
        ttft = None
        provider_handlers = {
            "TGI": self._process_tgi_response,
            "Ollama": self._process_ollama_response,
            "Llamacpp": self._process_llamacpp_response,
            "vLLM":self._process_vLLM_response
        }

        handler = provider_handlers.get(self.provider, None)
        if handler:
            generated_text, ttft = handler(response)

        output_tokens = len(tokenizer.encode(generated_text))
        return generated_text, output_tokens, ttft

    def _process_tgi_response(self, response):
        """
        Process the response for TGI provider.
        """
        generated_text = ""
        ttft = None

        for i, chunk in enumerate(response.iter_lines()):
            if chunk and i == 0 and ttft is None:
                ttft = (time.perf_counter() - start_time)
                logging.info(f"TTFT: {ttft*1000:.3f} ms")

            decoded_chunk = chunk.decode("utf-8")
            if "data:" in decoded_chunk:
                try:
                    json_data = decoded_chunk.split("data:")[1]
                    json_data = json.loads(json_data)
                    token = json_data["token"]["text"]
                    generated_text += token
                except (json.JSONDecodeError, KeyError):
                    print("Failed to extract decoded text from JSON")

        return generated_text, ttft

    def _process_ollama_response(self, response):
        """
        Process the response for Ollama provider.
        """
        generated_text = ""
        ttft = None

        for i, chunk in enumerate(response.iter_lines()):
            if chunk and i == 0 and ttft is None:
                ttft = (time.perf_counter() - start_time)
                logging.info(f"TTFT: {ttft*1000:.3f} ms")

            decoded_chunk = chunk.decode('utf-8')
            if decoded_chunk:
                try:
                    json_data = json.loads(decoded_chunk)
                    token = json_data["response"]
                    generated_text += token
                except (json.JSONDecodeError, KeyError):
                    print("Failed to extract decoded text from JSON")

        return generated_text, ttft

    def _process_llamacpp_response(self, response):
        """
        Process the response for Llamacpp provider.
        """
        generated_text = ""
        ttft = None
        for i, chunk in enumerate(response.iter_lines()):
            if chunk and i == 0 and ttft is None:
                ttft = (time.perf_counter() - start_time)
                logging.info(f"TTFT: {ttft*1000:.3f} ms")

            decoded_chunk = chunk.decode("utf-8")
            if "data:" in decoded_chunk:
                try:
                    json_data = decoded_chunk.split("data:")[1]
                    json_data = json.loads(json_data)
                    token = json_data["content"]
                    generated_text += token
                except (json.JSONDecodeError, KeyError):
                    print("Failed to extract decoded text from JSON")

        return generated_text, ttft

    def _process_vLLM_response(self, response):
        """
        Process the response for vLLM provider
        """
        generated_text = ""
        ttft = None
        for i, chunk in enumerate(response.iter_lines()):
            if chunk and i==0 and ttft is None:
                ttft = (time.perf_counter() - start_time)
                logging.info(f"TTFT: {ttft*1000:.3f} ms")
            
            decoded_chunk = chunk.decode("utf-8")
            if decoded_chunk == "data: [DONE]":
                    break
            if "data:" in decoded_chunk:
                try:
                    json_data = decoded_chunk.split("data:")[1]
                    json_data = json.loads(json_data)
                    token = json_data["choices"][0]["text"]
                    generated_text += token
                    #print(token,end="",flush=True)
                except (json.JSONDecodeError, KeyError) as e:
                    print("Failed to extract decoded text from JSON")
                    
        return generated_text, ttft

    @task
    def generate_text(self):
        """
        Task to generate text using the API and log the results.
        """
        if self.request_count > self.max_requests:
            self.environment.runner.quit()
            return

        input_data, input_tokens = self.format_prompt()

        # Record the start time of the API request
        global start_time
        start_time = time.perf_counter()
        try:
            response = self.client.post(self.api_url, json=input_data, stream=True)
            response.raise_for_status()
        except Exception as e:
            logging.error(f"Error making request: {e}")
            return
        
        generated_text, output_tokens, ttft = self.process_response(response)

        logging.info(f"Generated Text: {generated_text}")

        # Record the end time of the API request
        end_time = time.perf_counter()

        # End-to-end time for getting the response
        latency = (end_time - start_time)

        throughput = (output_tokens - 1) / (latency - ttft) if output_tokens > 1 else 0
        latency_per_token = (latency - ttft) * 1000 / (output_tokens - 1) if output_tokens > 1 else ttft * 1000

        # Convert start and stop times to datetime objects
        start_time_str = datetime.fromtimestamp(start_time).strftime('%H:%M:%S.%f')
        end_time_str = datetime.fromtimestamp(end_time).strftime('%H:%M:%S.%f')

        # Log the results to the output CSV file
        self.request_count += 1
        if self.request_count > self.max_requests:
            self.environment.runner.quit()

        self.log_results(start_time_str, end_time_str, input_tokens, output_tokens, latency, throughput, latency_per_token, ttft)
        try:
            barrier.wait()
        except BrokenBarrierError:
            pass

    def log_results(self, start_time, end_time, input_tokens, output_tokens, latency, throughput, latency_per_token, ttft):
        """
        Log the results to the output CSV file.
        """
        with open(self.output_file_path, 'a', newline='') as csvfile:
            fieldnames = [
                'request', 'start_time', 'end_time', 'input_tokens',
                'output_tokens', 'latency(ms)', 'throughput(tokens/second)',
                'latency_per_token(ms/tokens)', 'TTFT(ms)'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:
                writer.writeheader()

            writer.writerow({
                'request': self.request_count,
                'start_time': start_time,
                'end_time': end_time,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'latency(ms)': f"{latency * 1000:.3f}",
                'throughput(tokens/second)': f"{throughput:.3f}",
                'latency_per_token(ms/tokens)': f"{latency_per_token:.3f}",
                'TTFT(ms)': f"{ttft * 1000:.3f}"
            })

    def on_stop(self):
        """
        Perform actions on stopping the test.
        """
        try: 
            barrier.wait()
        except BrokenBarrierError:
            pass
        self.environment.runner.quit()