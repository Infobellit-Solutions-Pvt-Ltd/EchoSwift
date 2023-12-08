# Benchmark / Load-testing Suite by Infobellit

## LLM Benchmarking For Model Inference

### The main aspects we've focused on:
* Dataset to run the Benchmark.
* CPU and Memory Utilization.
* Load Testing for varying no of users.


The load test is designed to simulate continuous production load and observe the common LLM metrics that effects the user experience.


### Supported Generation Endpoints:

* Text Generation Inference (TGI)


### Captured metrics:

* Number of Input Tokens
* Number of Output Tokens

* Throughput (tokens/second)
* End-to-end Latency (ms)
* Token latency (ms/tokens)

* Time to first token (TTFT) for streaming
* Per token latency for streaming

* CPU and Memory Utilization while running the load test.

Metrics summary can be exported to CSV. This way multiple configuration can be scripted over. CSV file can be imported to Google Sheets/Excel or Jupyter for further analysis.

See the README file inside [`LLM_Inference_Benc`](LLM_Inference_Bench) folder for more detailed usage on how to get the dataset and run the becnhmark.
