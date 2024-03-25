import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

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
    
    # Iterate through each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
 
        # Read CSV file into a DataFrame
        df = pd.read_csv(file_path)
 
        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            output_token = row['output tokens']
            token_latency = row['latency_per_token(ms/tokens)']
            throughput = row['throughput(tokens/second)']
            ttft = row['TTFT(ms)']
 
            # Append values to the corresponding lists in the dictionary
            data_dict['token latency'].setdefault(output_token, []).append(token_latency)
            data_dict['throughput'].setdefault(output_token, []).append(throughput)
            data_dict['ttft'].setdefault(output_token, []).append(ttft)
 
    return data_dict

def plot_metric(data, directory, ylabel,output_file,fmt):
    x = np.arange(len(input_token_lengths))
    width = 0.15

    l = len(data)

    multiplier = 0.25
    colors = plt.cm.Blues(np.linspace(0.5, 1, l))

    fig, ax = plt.subplots(figsize=(20, 10))

    for key, values, color in zip(data.keys(), data.values(), colors):
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=int(key), color=color)
        ax.bar_label(rects, fmt=fmt, padding=2, size=10)
        multiplier += 1

    ax.set_ylabel(ylabel)
    ax.set_title(f"Results for {(directory.split(os.path.sep))[-1]}", fontdict=title_prop)
    ax.set_xticks(x + width * (l / 2))
    ax.set_xticklabels(input_token_lengths)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Output Tokens")

    plt.savefig(output_file)
    
    return fig
        
def main():
    st.title("LLM-Inference Benchmark Results")

    for directory in directories:
        result_dict = process_csv_files(directory)
        token_latency = result_dict["token latency"]
        throughput = result_dict["throughput"]
        ttft = result_dict["ttft"]
        head = directory.split("/")[-1]
        st.header(f"Results for {head}")
        
        # Condition for viewing CSV files
        view_csv = st.checkbox(f"View CSV files for {head}")
        
        if view_csv:
            csv_files = [file for file in os.listdir(directory) if file.endswith(".csv") and "avg" in file ]
            selected_csv = st.selectbox("Select a CSV file:", csv_files)
            
            if selected_csv:
                csv_path = os.path.join(directory, selected_csv)
                df = pd.read_csv(csv_path)
                st.subheader(f"Results for {selected_csv.split('.')[0]}")
                st.write(df, index=False)

        # Plot Token Latency
        st.subheader("Token Latency")
        fig_latency = plot_metric(token_latency, directory, 'Token Latency (ms/tokens)', f"{directory}_token_latency.png", '%d')
        st.pyplot(fig_latency)

        # Plot Throughput
        st.subheader("Throughput")
        fig_throughput = plot_metric(throughput, directory, 'Throughput (Tokens/second)',f"{directory}_throughput.png", '%.2f')
        st.pyplot(fig_throughput)

        # Plot TTFT
        st.subheader("TTFT")
        fig_ttft = plot_metric(ttft, directory, 'TTFT (ms)', f"{directory}_ttft.png", '%d')
        st.pyplot(fig_ttft)

if __name__ == "__main__":

    # Path to the csv files 
    base_directory = r"Llama-2-7b-chat-hf/postloading/Locust_Test_Results"
    
    directories = process_directory(base_directory)

    input_token_lengths = ["32_input_token", "64_input_token", "128_input_token", "256_input_token"]
    title_prop = {"weight": "bold", "color": "black", "size": 20}
    main()
