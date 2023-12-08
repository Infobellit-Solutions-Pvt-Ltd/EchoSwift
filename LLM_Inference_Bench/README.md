# Follow the below steps to reproduce similar results

## Setup the Environment

### Install the Dependencies
* The Inference Benchmark load test relies on [Locust package](https://locust.io/). Install it from pip.

```bash
pip install -r requirements.txt
```

## Download [Llama Tokenizer](https://huggingface.co/hf-internal-testing/llama-tokenizer)
* Add tokenizer to the root directory
* Used to calculate no of Input and Output tokens while filtering the dataset and while running the load test. 

## Dataset Filtering

We used [ShareGPT](https://huggingface.co/datasets/pvduy/sharegpt_alpaca_oa_vicuna_format/viewer/default/train?p=1) Dataset from HuggingFace datasets.


* we've filtered out the ShareGPT dataset based on varying input token length.
* Different Input Token Lengths considered are 32,64,128,256,512,1k,2k
* 1000 prompts filtered out for each token length specified above.

```bash
python Dataset_Filtering.py --Total_Prompts 1000
```

## Profiling Memory/CPU

* we're observing the CPU and Memory percentage utilization while running the load test with varying no of parallel users, also with varying Input and Output tokens

## Run the Load Test

* Define the configurations  you want to test with in the ["locust.sh"] shell script.
* List of parallel users that we tested "[1,3,10,30]".
* Varying Input tokens "[32,64,128,256,512]" and Output tokens "[32,64,128,256,512]"

```bash
python sut_loop_wrapper.py
```

* The above command runs the load test against the TGI endpoint and profiles the CPU/MEM utilization
* This way multiple configuration can be scripted over.

### All the results will be saved into csv files, you can analyze the data.
