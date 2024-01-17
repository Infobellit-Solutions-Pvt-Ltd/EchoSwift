## LLM-Inference-Bench: Benchmarking LLM Inference by Infobellit

## LLM-Inference-Bench Tool

 The objective of the LLM Inference Bench Tool is to identify the Latency of each request made and measured in millisecond/token,Time Taken for the First Token (TTFT) and the Throughput measured in number of tokens/second. The above metrics is measured with varying Input Tokens (Query Length), Output Tokens (Response Length) and Simulating Parallel Users.

## Summarizing the main aspects focused on:


![Img](Assets/LLM_Inference_Bench_Tool.png)
The Benchmark tool mainly focusses on data collection ,analyzing the CPU and Memory requirements and load testing with varying number of Users.
## Performance metrics:


![Img](Assets/Parameters.png)

The performance metrics captured while running the benchmark includes Latency,TTFT and Throughput for varying input and output tokens and parallel users. 
# Follow the below steps to reproduce similar results

## Setup the Environment

### Create a Virtual Environment and activate it before installing the dependencies.
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### Install the Dependencies
* The Inference Benchmark load test relies on [Locust package](https://locust.io/). Install it from pip.

```bash
pip install -r requirements.txt
```

## Download [Llama Tokenizer](https://huggingface.co/hf-internal-testing/llama-tokenizer)

* Add tokenizer to the root directory.

* Used to calculate no of Input and Output tokens while filtering the dataset and while running the load test. 
 
## Dataset Filtering

Here the dataset used is [ShareGPT](https://huggingface.co/datasets/pvduy/sharegpt_alpaca_oa_vicuna_format/viewer/default/train?p=1) Dataset from Hugging Face datasets.
* ShareGPT dataset has been filtered based on varying input token length.
* Different Input Token Lengths considered are 32,64,128,256,512,1k,2k
* 1000 prompts are filtered out for each token length specified above.

```bash
python3 Dataset_Filtering.py --Total_Prompts 1000
```
## Profiling Memory/CPU

* CPU and Memory Percentage utilization are observed while running the load test with varying no of parallel users, along with varying Input and Output tokens

## Run the Load Test

* Define the configurations  you want to test with in the ["locust.sh"] shell script.
* List of parallel users that we tested "[1,3,10,30]".
* Varying Input tokens "[32,64,128,256,512]" and Output tokens "[32,64,128,256,512]"

```bash
./run_benchmark.sh
```
* The above command starts server.py and client.py file.
* Server.py invokes the Text generation inference container with the default resources, the model is loaded and the url is provided to the config.yaml file.
* Client.py runs the locust script for load test and calls the container manager class in utils.py for monitoring and profiling containers and collects the CPU and 
  memory utilization stats while running the benchmark. 
* This way multiple configuration can be scripted over.
* All the results will be saved into csv files, you can analyze the data.

## Benchmark Result Analysis

* All the CSV's received are further processed by running app.py and the throughput, latency, ttft are analyzed with the help of graphs generated.

```bash
python3 app.py
```
![Sample_Output_plots](https://github.com/Infobellit-Solutions-Pvt-Ltd/LLM-Inference-Benchmark/assets/154504188/5ff09150-f419-4963-ac15-b03a0e61c554)



Refer [`LLM_Inference_Benchmark_pdf`](Inference-Benchmark-tool-public.pdf) for more detailed usage on how to get the dataset and run the benchmark.
