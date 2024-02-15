from locust import HttpUser, task
import random
import time
import csv
from transformers import AutoTokenizer
import os
from datetime import datetime
import json

# Initialize the tokenizer for encoding/decoding text
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")


class APITestUser(HttpUser):
    """
    Represents a Locust user for load testing an API.

    Attributes:
        i (int): Counter for the number of requests made.
        max_requests (int): Maximum number of requests to be made.
        max_new_tokens (int): Maximum number of tokens in each API request.
        api_url (str): URL of the API to be tested.
        questions_file (str): Path to the CSV file containing questions.
        questions (list): List of questions loaded from the CSV file.
        output_file (str): Path to the output CSV file for logging results.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the APITestUser instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ValueError: If required environment variables are not set.
        """
        super().__init__(*args, **kwargs)
        self.i = 0
        self.max_requests = int(os.environ.get("MAX_REQUESTS", 10))
        self.max_new_tokens = int(os.environ.get('MAX_NEW_TOKENS', 128))
        self.api_url = os.environ.get('API_URL', '')
        self.dataset_file = os.environ.get('INPUT_DATASET', '')
        self.questions = self.load_dataset(self.dataset_file)
        self.output_file_path = os.environ.get('OUTPUT_FILE', 'output.csv')
        self.users_list = []

    @staticmethod
    def load_dataset(csv_file):
        """
        Read questions from a CSV file.

        Args:
            csv_file (str): Path to the CSV file.

        Returns:
            list: List of questions.
        """
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            return [row['Input_Prompt'] for row in reader]

    def on_start(self):
        self.users_list.append({'requests_made': 0, 'total_requests': 0})

    @task
    def generate_text(self):
        """
        Task to generate text using the API and log the results.

        Raises:
            ValueError: If the API request fails.
        """
        ttft = 0  # Time to first token
        chunk_list = bytearray()

        if self.i > self.max_requests:
            self.environment.runner.quit()

        # Randomly select a prompt from the list of questions
        selected_prompt = random.choice(self.questions)
        data = {'inputs': selected_prompt, 'parameters': {'max_new_tokens': self.max_new_tokens}}

        # Record the start time of the API request
        start_time = time.perf_counter()
        response = self.client.post(self.api_url, json=data, stream=True)

        # Iterate over the content received in chunks
        for i, chunk in enumerate(response.iter_content()):
            if chunk and i == 0:
                chunk_list.extend(chunk)
                ttft = (time.perf_counter() - start_time) * 1000
                print(f"First token generation time: {ttft} ms")
            chunk_list.extend(chunk)

        combined_texts = []
        decoded_chunks = chunk_list.decode("utf-8").split("\n")
        for text in decoded_chunks:
            if text == '':
                pass
            else:
                combined_texts.append(text)

        text = combined_texts[-1].replace("data:", '')

        # Try to extract the generated text from the last chunk
        generated_text = ""
        try:
            last_chunk_data = json.loads(text)
            print("===" * 20)
            generated_text += last_chunk_data.get("generated_text", "Not Found")
            print(f"Generated Text: {generated_text}")
        except (json.JSONDecodeError, KeyError):
            print("Failed to extract generated text from the response.")

        # Record the end time of the API request
        end_time = time.perf_counter()
        self.i += 1
        if self.i > self.max_requests:
            self.environment.runner.quit()

        # Adding + 2 tokens to add the special tokens(bos/eos) in total no of tokens
        input_tokens = len(tokenizer.encode(data["inputs"]))
        output_tokens = len(tokenizer.encode(generated_text))

        # end-to-end time for getting the response
        latency = (end_time - start_time) * 1000
        throughput = (input_tokens + output_tokens) / (end_time - start_time)  # tokens/sec
        if output_tokens != 1:
            time_per_token = (latency - ttft) / (output_tokens - 1)  # Token latency(ms/tokens)
        else:
            print("only one output token generated, time_per_token will be same as TTFT")
            time_per_token = ttft

        # Convert start and stop times to datetime objects
        start_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S.%f')
        end_time = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S.%f')

        # Log the results to the output CSV file
        with open(self.output_file_path, 'a', newline='') as csvfile:
            fieldnames = ['request', 'start_time', 'end_time', 'input_tokens',
                          'output_tokens', 'latency(ms)', 'throughput(tokens/second)', 'time_per_token(ms/tokens)',
                          'TTFT(ms)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:
                writer.writeheader()

            writer.writerow({
                'request': self.i,
                'start_time': start_time,
                'end_time': end_time,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'latency(ms)': f"{latency:.3f}",
                'throughput(tokens/second)': f"{throughput:.3f}",
                'time_per_token(ms/tokens)': f"{time_per_token:.3f}",
                'TTFT(ms)': f"{ttft:.3f}"
            })

    def on_stop(self):
        # Calculate the total requests made by all users
        total_requests_made = sum(user_data.get('requests_made', 0) for user_data in self.users_list)
        self.environment.runner.quit()

        # Determine if all users have finished their requests
        if total_requests_made >= self.max_requests:
            self.environment.runner.quit()
