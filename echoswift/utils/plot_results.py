import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def process_csv_files(directory_path):
    user_number = int(''.join(filter(str.isdigit, directory_path.name)))
    data = {}

    csv_files = sorted(directory_path.glob('avg_*_input_tokens.csv'), 
                       key=lambda x: int(''.join(filter(str.isdigit, x.stem))))
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            output_token = row['output tokens']
            token_latency = row['latency_per_token(ms/token)']
            throughput = row['throughput(tokens/second)']
            ttft = row['TTFT(ms)']
            data.setdefault(user_number, []).append((output_token, token_latency, throughput, ttft))

    return data

def write_to_csv(data, output_file):
    with open(output_file, 'w') as f:
        f.write('Number of Parallel Requests,Output Token,Token Latency (ms/token),Throughput (tokens/second),TTFT (ms)\n')
        for num_Requests, values in sorted(data.items()):
            for value in values:
                f.write(f'{num_Requests},{value[0]},{value[1]},{value[2]},{value[3]}\n')

def plot_line_chart(data, x_label, y_label, title, output_file):
    plt.figure(figsize=(10, 6))

    x_values = data[x_label]
    y_values = data[y_label]

    num_Requests = sorted(set(x_values))
    x_ticks_positions = np.arange(len(num_Requests))

    plt.plot(x_ticks_positions, y_values, marker='o', label=y_label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    for i, txt in enumerate(y_values):
        plt.annotate(f'{txt:.2f}', xy=(x_ticks_positions[i], y_values.iloc[i]), ha='center', va='bottom')

    plt.xticks(x_ticks_positions, [str(num) for num in num_Requests])
    
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_benchmark_results(base_directory):
    base_directory = Path(base_directory)
    output_file = base_directory / 'aggregated_data.csv'
    
    data = {}
    for directory in base_directory.iterdir():
        if directory.is_dir() and "_User" in directory.name:
            directory_data = process_csv_files(directory)
            for num_Requests, values in directory_data.items():
                data.setdefault(num_Requests, []).extend(values)

    write_to_csv(data, output_file)
    print(f"Aggregated data has been written to {output_file}")

    df = pd.read_csv(output_file)

    plot_line_chart(df[['Number of Parallel Requests', 'Token Latency (ms/token)']], 
                    'Number of Parallel Requests', 'Token Latency (ms/token)', 
                    'Parallel Requests vs Token Latency', 
                    base_directory / 'token_latency_plot.png')

    plot_line_chart(df[['Number of Parallel Requests', 'Throughput (tokens/second)']], 
                    'Number of Parallel Requests', 'Throughput (tokens/second)', 
                    'Parallel Requests vs Throughput', 
                    base_directory / 'throughput_plot.png')

    plot_line_chart(df[['Number of Parallel Requests', 'TTFT (ms)']], 
                    'Number of Parallel Requests', 'TTFT (ms)', 
                    'Parallel Requests vs Time to First Token', 
                    base_directory / 'ttft_plot.png')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process CSV files and generate plots.')
    parser.add_argument('base_directory', type=str, help='The base directory containing the result directories.')
    args = parser.parse_args()
    plot_benchmark_results(args.base_directory)