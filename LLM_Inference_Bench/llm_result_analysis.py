import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
def process_directory(directory_path):
    # List to store directory paths
    directory_paths = []
 
    # Iterate over all items in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
 
        # Check if the item is a directory and matches the specified pattern
        if os.path.isdir(item_path) and re.match(r'\d+_User', item):
            directory_paths.append(item_path)
 
    # Sort the directory paths based on the numeric values in their names
    directory_paths.sort(key=lambda x: extract_numeric_value(os.path.basename(x)))
 
    return directory_paths
 
def extract_numeric_value(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


# Function to read and process CSV files
def process_csv_files(directory_path):
    # Dictionary to store data
    data_dict = {
        'token latency': {},
        'throughput': {},
        'ttft': {}
    }
 
    # List of CSV files in the directory, sorted based on numeric values and file name pattern
    csv_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.csv') and re.match(r'avg_\d+_input_tokens\.csv', f)],
                key=extract_numeric_value)
    #print(csv_files)
    # Iterate through each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
 
        # Read CSV file into a DataFrame
        df = pd.read_csv(file_path)
 
        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            output_token = row['output tokens']
            token_latency = row['time_per_token(ms/tokens)']
            throughput = row['throughput(tokens/second)']
            ttft = row['TTFT(ms)']
 
            # Append values to the corresponding lists in the dictionary
            data_dict['token latency'].setdefault(output_token, []).append(token_latency)
            data_dict['throughput'].setdefault(output_token, []).append(throughput)
            data_dict['ttft'].setdefault(output_token, []).append(ttft)
 
    return data_dict
 
def plotting_latency_throughput():
    # Plotting
    input_token_lengths = ["32_input_token", "128_input_token", "256_input_token"]
    x = np.arange(len(input_token_lengths))  # the label locations
    width = 0.20  # the width of the bars
 
    for directory in directories:
        result_dict = process_csv_files(directory)
        #print(result_dict)
 
        token_latency = result_dict["token latency"]
        throughput = result_dict["throughput"]
       
        # length for number of output tokens
        l = len(token_latency)
 
        # Create subplots with one row and two columns
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
 
        multiplier = 0.25  # set multiplier for the first subplot
 
        # Define a list of blue shades for bar colors
        colors = plt.cm.Blues(np.linspace(0.5, 1, l))
 
        for key, value, color in zip(token_latency.keys(), token_latency.values(), colors):
            offset = width * multiplier
            rects = axes[0].bar(x + offset, value, width, label=int(key), color=color)
            axes[0].bar_label(rects, fmt='%d', padding=2, size=10)
            multiplier += 1
 
        # Add some text for labels, title and custom x-axis tick labels, etc.
        axes[0].set_ylabel('Token Latency (ms/tokens)')
        axes[0].set_title(f"SUT with Container_config - (16 cores & 32G) for {(directory.split(os.path.sep))[-1]}",fontdict=title_prop)
        axes[0].set_xticks(x + width * (l/2))
        axes[0].set_xticklabels(input_token_lengths)
        axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1), title="Output Tokens")
 
        multiplier = 0.25  # Reset multiplier for the second subplot
 
        for key, value, color in zip(throughput.keys(), throughput.values(), colors):
            offset = width * multiplier
            rects = axes[1].bar(x + offset, value, width, label=int(key), color=color)
            axes[1].bar_label(rects, fmt='%.2f', padding=2, size=10)
            multiplier += 1
 
        axes[1].set_ylabel('Throughput (tokens/sec)')
        axes[1].set_xticks(x + width * (l/2))
        axes[1].set_xticklabels(input_token_lengths)
        axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1), title="Output Tokens")
 
        plt.savefig(f"{directory}.png")
 
def plotting_ttft():
    # Plotting
    input_token_lengths = ["32_input_token", "128_input_token", "256_input_token"]
    x = np.arange(len(input_token_lengths))  # the label locations
    width = 0.20  # the width of the bars
   
 
    for directory in directories:
        result_dict = process_csv_files(directory)
        #print(result_dict)
 
        ttft = result_dict["ttft"]
       
        # length for number of output tokens
        l = len(ttft)
 
        # Create subplots with one row and two columns
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
 
        multiplier = 0.25  # set multiplier for the first subplot
 
        # Define a list of blue shades for bar colors
        colors = plt.cm.Blues(np.linspace(0.5, 1, len(ttft)))
 
        for key, value, color in zip(ttft.keys(), ttft.values(), colors):
            offset = width * multiplier
            rects = axes.bar(x + offset, value, width, label=int(key), color=color)
            axes.bar_label(rects, fmt='%d', padding=2, size=10)
            multiplier += 1
 
        # Add some text for labels, title and custom x-axis tick labels, etc.
        axes.set_ylabel('T T F T (ms)')
        axes.set_title(f"SUT with Container_config - (16 cores & 32G) for {(directory.split(os.path.sep))[-1]}", fontdict=title_prop)
        axes.set_xticks(x + width * (l/2))
        axes.set_xticklabels(input_token_lengths)
        axes.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Output Tokens")
 
        plt.savefig(f"{directory}_ttft.png")
 
if __name__ == "__main__":
    base_directory = r"Llama-2-7b-chat-hf/SUT_25m/postloading/Locust_Test_Results"
    directories = process_directory(base_directory)
    title_prop = {"weight":"bold", "color":"black", "size":20}
    
    # Call the plotting functions without the break statement
    plotting_latency_throughput()
    plotting_ttft()