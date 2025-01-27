import subprocess
import json
import os
import shutil
import pandas as pd
from pathlib import Path
import argparse
import time

# Define the threshold for TTFT and latency
VALIDATION_CRITERION = {"TTFT": 2000, "latency_per_token": 200, "latency": 200}

def validate_criterion(ttft, latency_per_token, latency):
    """
    Validate the results against the defined threshold.
    """
    return (
        ttft <= VALIDATION_CRITERION["TTFT"]
        and latency_per_token <= VALIDATION_CRITERION["latency_per_token"]
    )

def run_benchmark(config_file):
    """
    Run the benchmark using subprocess and the provided configuration file.
    """
    try:
        start_time = time.time()
        result = subprocess.run(
            ['echoswift', 'start', '--config', config_file],
            capture_output=True,
            text=True
        )
        run_time = time.time() - start_time
        return result, run_time
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return None, None

def copy_avg_response(config_file, user_count):
    """
    Copy and rename avg_32_input_tokens.csv to the results folder based on the user count.
    """
    try:
        with open(config_file, "r") as file:
            config = json.load(file)

        out_dir = config.get("out_dir")
        if not out_dir:
            print("Error: 'out_dir' not specified in the config file.")
            return

        out_dir_path = Path(out_dir)
        results_dir = out_dir_path / "Results"
        results_dir.mkdir(parents=True, exist_ok=True)

        user_folder = out_dir_path / f"{user_count}_User"
        avg_response_path = user_folder / "avg_Response.csv"
        if avg_response_path.exists():
            new_name = f"avg_Response_User{user_count}.csv"
            shutil.copy(avg_response_path, results_dir / new_name)
            print(f"Copied and renamed avg_32_input_tokens.csv to {results_dir / new_name}")
        else:
            print(f"avg_32_input_tokens.csv not found in {user_folder}")

    except Exception as e:
        print(f"Unexpected error: {e}")

def extract_metrics_from_avg_response(result_dir, user_count):
    """
    Extract TTFT, latency, and latency_per_token values from the avg_32_input_tokens_User{user_count}.csv file.
    """
    try:
        avg_response_path = result_dir / f"avg_Response_User{user_count}.csv"
        if avg_response_path.exists():
            df = pd.read_csv(avg_response_path)
            ttft = df["TTFT(ms)"].iloc[0]
            latency_per_token = df["latency_per_token(ms/token)"].iloc[0]
            latency = df["latency(ms)"].iloc[0]
            throughput = df["throughput(tokens/second)"].iloc[0]
            total_throughput = throughput * user_count

            return ttft, latency_per_token, latency, throughput, total_throughput
        else:
            print(f"{avg_response_path} not found.")
            return None, None, None, None, None
    except Exception as e:
        print(f"Error extracting metrics from {avg_response_path}: {e}")
        return None, None, None, None, None

def binary_search_user_count(config_file, low, high, result_dir):
    """
    Perform binary search to refine the optimal user count after a failed validation.
    """
    while low < high:
        mid = (low + high) // 2

        # Update config with the mid user count
        with open(config_file, 'r') as file:
            config = json.load(file)

        config["user_counts"] = [mid]
        with open(config_file, 'w') as file:
            json.dump(config, file, indent=4)

        # Run the benchmark
        result, run_time = run_benchmark(config_file)
        if result is None:
            print("Benchmark run failed during binary search. Exiting.")
            break

        # Copy and extract metrics
        copy_avg_response(config_file, mid)
        metrics = extract_metrics_from_avg_response(result_dir, mid)
        if any(metric is None for metric in metrics):
            print("Error extracting metrics during binary search. Exiting.")
            break

        ttft, latency_per_token, latency, throughput, total_throughput = metrics
        print(f"Binary Search - User Count: {mid}, TTFT: {ttft} ms, Latency: {latency} ms, "
              f"Latency per Token: {latency_per_token} ms/token, Throughput: {throughput} tokens/second, Total Throughput: {total_throughput} tokens/second")

        if validate_criterion(ttft, latency_per_token, latency):
            # Threshold met; continue searching upward
            low = mid + 1
        else:
            # Threshold not met; search downward
            high = mid

    # Return the highest user count that met the criteria
    return low - 1

def generate_summary_report(out_dir, optimal_user_count, metrics, benchmark_time):
    """
    Generate a summary report with the benchmark metrics.
    """
    try:
        summary_report_path = Path(out_dir) / "Results" / "summary_report.csv"
        summary_report_path.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "Optimal User Count": [optimal_user_count],
            "TTFT (ms)": [metrics[0]],
            "Latency (ms)": [metrics[2]],
            "Latency per Token (ms/token)": [metrics[1]],
            "Throughput (tokens/second)": [metrics[3]],
            "Total Throughput (tokens/second)": [metrics[4]],
            "Benchmark Time (seconds)": [benchmark_time]
        }

        df = pd.DataFrame(report_data)
        df.to_csv(summary_report_path, index=False)
        print(f"Summary report generated at {summary_report_path}")

    except Exception as e:
        print(f"Error generating summary report: {e}")

def OptimalUser(config_file):
    """
    Adjust the user count and run the benchmark until the threshold is met or the optimal user count is found.
    """
    with open(config_file, 'r') as file:
        config = json.load(file)

    user_count = 10  # Start with 100 users
    previous_user_count = 0
    increment = 100
    out_dir = config.get("out_dir")
    if not out_dir:
        print("Error: 'out_dir' not specified in the config file.")
        return

    out_dir_path = Path(out_dir)
    result_dir = out_dir_path / "Results"
    result_dir.mkdir(parents=True, exist_ok=True)

    optimal_user_count = 0
    final_metrics = None
    benchmark_time = 0

    while True:
        # Update config with the current user count
        config["user_counts"] = [user_count]
        with open(config_file, 'w') as file:
            json.dump(config, file, indent=4)

        # Run the benchmark
        result, benchmark_time = run_benchmark(config_file)
        if result is None:
            print("Benchmark run failed. Exiting.")
            break

        # Copy and extract metrics
        copy_avg_response(config_file, user_count)
        metrics = extract_metrics_from_avg_response(result_dir, user_count)
        if any(metric is None for metric in metrics):
            print("Error extracting metrics. Exiting.")
            break

        ttft, latency_per_token, latency, throughput, total_throughput = metrics
        print(f"User Count: {user_count}, TTFT: {ttft} ms, Latency: {latency} ms, "
              f"Latency per Token: {latency_per_token} ms/token, Throughput: {throughput} tokens/second, "
              f"Total Throughput: {total_throughput} tokens/second")

        if validate_criterion(ttft, latency_per_token, latency):
            print(f"Threshold met for {user_count} users.")
            previous_user_count = user_count
            optimal_user_count = user_count
            final_metrics = metrics  # Store the metrics for the last valid run
            user_count += increment
        else:
            print(f"Threshold not met for {user_count} users.")
            optimal_user_count = binary_search_user_count(config_file, previous_user_count, user_count, result_dir)
            final_metrics = extract_metrics_from_avg_response(result_dir, optimal_user_count)
            break

    # Generate the summary report with the final metrics
    if final_metrics and optimal_user_count > 0:
        generate_summary_report(out_dir, optimal_user_count, final_metrics, benchmark_time)
    else:
        print("No valid optimal user count found. Exiting.")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Benchmark automation script")
    # parser.add_argument("config_file", type=str, help="Path to the benchmark configuration file")
    # args = parser.parse_args()

    OptimalUser(config_file="config.json")
