# EchoSwift: LLM Inference Benchmarking Tool

EchoSwift is a powerful and flexible tool designed for benchmarking Large Language Model (LLM) inference. It allows users to measure and analyze the performance of LLM endpoints across various metrics, including latency, throughput, and time to first token (TTFT).

![EchoSwift](images/Echoswift.png)

## Features

- Benchmark LLM inference across multiple providers (e.g., Ollama, vLLM, TGI)
- Measure key performance metrics: latency, throughput, and TTFT
- Support for varying input and output token lengths
- Simulate concurrent users to test scalability
- Easy-to-use CLI interface
- Detailed logging and progress tracking

## Performance metrics:

The performance metrics captured for varying input and output tokens and parallel users while running the benchmark includes 
- Latency (ms/token)
- TTFT(ms)
- Throughput(tokens/sec) 

![metrics](images/metric.png)

## Installation

You can install EchoSwift using pip:

```bash
pip install echoswift
```

Alternatively, you can install from source:

```bash
git clone --branch akhil https://github.com/Infobellit-Solutions-Pvt-Ltd/EchoSwift.git
cd EchoSwift
pip install -e .
```

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Usage

EchoSwift provides a simple CLI interface for running benchmarks. Here are the main commands:

### 1. Download and Filter Dataset

Before running a benchmark, you need to download and filter the dataset:

```bash
echoswift dataprep
```

This command will download the ShareGPT dataset and filter it based on various input token lengths.

### 2. Configure the Benchmark

Create or modify the `config.yaml` file in the project root directory. Here's an example configuration:

```yaml
out_dir: "results"
base_url: "http://localhost:11434/api/generate"
provider: "Ollama"
model: "llama2" # Model is required for Ollama and vLLM
max_requests: 5
user_counts: [1, 3, 10]
input_tokens: [32]
output_tokens: [256]
```

Adjust these parameters according to your needs and the LLM endpoint you're benchmarking.

### 3. Run the Benchmark

To start the benchmark using the configuration from `config.yaml`:

```bash
echoswift start
```

If you want to use a different configuration file:

```bash
echoswift start --config path/to/your/config.yaml
```

## Output

EchoSwift will create a `results` directory (or the directory specified in `out_dir`) containing:

- CSV files with raw benchmark data
- Averaged results for each combination of users, input tokens, and output tokens
- Log files for each Locust run

## Analyzing Results

After the benchmark completes, you can find detailed CSV files in the output directory. These files contain information about latency, throughput, and TTFT for each test configuration.

<!-- ## Advanced Usage

For more advanced usage and customization options, please refer to the [documentation](link-to-your-documentation). -->

<!-- ## Contributing

We welcome contributions to EchoSwift! Please see our [Contributing Guide](CONTRIBUTING.md) for more details. -->

<!-- ## License

EchoSwift is released under the [MIT License](LICENSE). -->

## Citation

If you find our resource useful, please cite our paper:

### [EchoSwift: An Inference Benchmarking and Configuration Discovery Tool for Large Language Models (LLMs)](https://dl.acm.org/doi/10.1145/3629527.3652273)

```bibtex
@inproceedings{Krishna2024,
  series = {ICPE '24},
  title = {EchoSwift: An Inference Benchmarking and Configuration Discovery Tool for Large Language Models (LLMs)},
  url = {https://dl.acm.org/doi/10.1145/3629527.3652273},
  DOI = {10.1145/3629527.3652273},
  booktitle = {Companion of the 15th ACM/SPEC International Conference on Performance Engineering},
  publisher = {ACM},
  author = {Krishna, Karthik and Bandili, Ramana},
  year = {2024},
  month = May,
  collection = {ICPE '24}
}
```

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/yourusername/echoswift/issues) on our GitHub repository.