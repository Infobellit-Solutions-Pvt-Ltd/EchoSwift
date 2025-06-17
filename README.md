# EchoSwift: LLM Inference Benchmarking Tool by Infobell IT

EchoSwift is a powerful and flexible tool designed for benchmarking Large Language Model (LLM) inference. It allows users to measure and analyze the performance of LLM endpoints across various metrics, including token latency, throughput, and time to first token (TTFT).

![EchoSwift](images/Echoswift.png)

## 📚 Table of Contents

- [Features](#features)
- [Supported Inference Servers](#supported-inference-servers)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
  - [Step 1: Prepare Dataset](#1-download-the-dataset-and-create-a-default-configjson)
  - [Step 2: Configure Benchmark](#2-configure-the-benchmark)
  - [Step 3: Run the Benchmark](#3-run-the-benchmark)
  - [Step 4: Plot the Results](#4-plot-the-results)
- [CLI Reference](#cli-reference)
- [Output Structure](#output)
- [Citation](#citation)


## Features
- Intuitive CLI for seamless benchmarking setup and execution
- Evaluate LLM inference performance across various inference servers
- Capture essential metrics including latency, throughput, and Time to First Token (TTFT)
- Flexible testing with customizable input and output token lengths
- Simulate concurrent users to assess server scalability under load
- Automatically identify the optimal user load threshold while ensuring:
    - TTFT remains under 2000 ms
    - Token latency stays below 200 ms
- Comprehensive logging and real-time progress indicators for enhanced observability

## Supported Inference Servers
  - TGI
  - vLLM
  - Ollama
  - Llamacpp
  - NIMS
  - SGLang

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
git clone https://github.com/Infobellit-Solutions-Pvt-Ltd/EchoSwift.git
cd EchoSwift
pip install -e .
```

## Usage

EchoSwift provides a simple CLI interface for running LLM Inference benchmarks.

Below are the steps to run a sample test, assuming the generation endpoint is active.

### 1. Download the Dataset and create a default `config.json`

Before running a benchmark, you need to download and filter the dataset:

```bash
echoswift dataprep
```
This command will:
- Download the filtered ShareGPT dataset from Huggingface
- Create a default `config.json` file in your working directory


### 2. Configure the Benchmark

Edit the generated `config.json` file to match your LLM server configuration. Below is a sample:

```json
{
    "_comment": "EchoSwift Configuration",
    "out_dir": "Results",
    "base_url": "http://localhost:8000/v1/completions",
    "tokenizer_path": "/path/to/tokenizer/",
    "inference_server": "vLLM",
    "model": "/model",
    "random_prompt": true,
    "max_requests": 1,
    "user_counts": [
        10
    ],
    "increment_user": [
        100
    ],
    "input_tokens": [
        32
    ],
    "output_tokens": [
        256
    ]
}

```
**Note:** Modify base_url, tokenizer_path, model, and other fields according to your LLM deployment.

#### 🔧 Prompt Configuration Modes

EchoSwift supports two input modes depending on your test requirements:

##### ✅ Fixed Input Tokens

If you want to run the benchmark with a **fixed number of input tokens**:

* Set `"random_prompt": false`
* Define both `input_tokens` and `output_tokens` explicitly

##### 🎲 Random Input Length

If you prefer using **randomized prompts** from the dataset:

* Set `"random_prompt": true`
* Provide only `output_tokens` — EchoSwift will choose random input lengths from the dataset

You can use these configuration with both:

* `echoswift start` (standard benchmark)
* `echoswift optimaluserrun` (to determine optimal concurrency)

#### 👥 User Load Configuration (For `optimaluserrun`)

To perform optimal user benchmarking:

* Use `user_counts` to set the **initial number of concurrent users**
* Use `increment_user` to define how many users to add per step

Example:

```json
"user_counts": [10],
"increment_user": [100]
```

In this case, the benchmark will start with 10 users and increase by 100 in each iteration until performance thresholds are hit.

### 🔤 Tokenizer Configuration

EchoSwift allows two ways to configure the tokenizer used for benchmarking:

#### Option 1: Use a Custom Tokenizer

Set the `TOKENIZER` environment variable to the path of your desired tokenizer.

#### Option 2: Use Default Fallback

If `TOKENIZER` is not set or is empty, EchoSwift falls back to a built-in default tokenizer:

This ensures the tool remains functional, but the fallback tokenizer may not align with your model's behavior. Use it only for testing or when no tokenizer is specified.

---

> ✅ **Best Practice:** Always specify the correct tokenizer that matches your LLM model for accurate benchmarking results.

---

Use these combinations as per your requirement to effectively benchmark your LLM endpoint.


### 3. Run the Benchmark

**Option A: Standard Benchmarking**

Use the start command to run a basic benchmark:

```bash
echoswift start --config path/to/config.json
```

**Option B: Optimal User Load Benchmarking**

To find the optimal number of concurrent users for your LLM endpoint:

```bash
echoswift optimaluserrun --config path/to/config.json
```

### 4. Plot the Results

Visualize the benchmark results using the built-in plotting tool:

```bash
echoswift plot --results-dir path/to/your/results_dir
```

### CLI Reference
```bash
echoswift [OPTIONS] COMMAND [ARGS]...
```

#### Commands
| Command          | Description                                                                   |
| ---------------- | ----------------------------------------------------------------------------- |
| `dataprep`       | Download the ShareGPT dataset and create a sample config file                 |
| `start`          | Start the EchoSwift benchmark with the given configuration                    |
| `optimaluserrun` | Run benchmark iteratively to determine the optimal number of concurrent users |
| `plot`           | Generate performance plots from the benchmark results                         |


## Output

EchoSwift will create a `results` directory (or the directory specified in `out_dir`) containing:

- CSV files with raw benchmark data
- Averaged results for each combination of users, input tokens, and output tokens
- Log files for each Locust run

## Analyzing Results

After the benchmark completes, you can find CSV files in the output directory. These files contain information about latency, throughput, and TTFT for each test configuration.

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
Refer [AMD_BUILD_GUIDE](https://www.amd.com/content/dam/amd/en/documents/epyc-technical-docs/user-guides/58669-amd-epyc-9004-ug-openshift.pdf) for detailed steps on LLM deployment on Red Hat Openshift Container Platform.

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/yourusername/echoswift/issues) on our GitHub repository.
