import subprocess
import json
import os
import shutil
import pandas as pd
from pathlib import Path
import keyboard  
import argparse
import time

# Define the threshold for TTFT and latency
VALIDATION_CRITERION = {"TTFT": 2000, "latency_per_token": 200}

def validate_criterion(ttft, latency_per_token):
    """
    Validate the results against the defined threshold.
    """
    return (
        ttft <= VALIDATION_CRITERION["TTFT"]
        and latency_per_token <= VALIDATION_CRITERION["latency_per_token"]
        #and latency <= VALIDATION_CRITERION["latency"]
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

def copy_avg_response(config_file, result_dir, user_count):
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

        if config.get("random_prompt"):
            user_folder = out_dir_path / f"{user_count}_User"
            avg_response_path = user_folder / "avg_Response.csv"
            if avg_response_path.exists():
                new_name = f"avg_Response_User{user_count}.csv"
                shutil.copy(avg_response_path, result_dir / new_name)
                print(f"Copied and renamed avg_Response.csv to {result_dir / new_name}")
            else:
                print(f"avg_Response.csv not found in {user_folder}")

        else:
            user_folder = out_dir_path / f"{user_count}_User"
            input_token = config.get("input_tokens")[0]
            avg_response_path = user_folder / f"avg_{input_token}_input_tokens.csv"
            if avg_response_path.exists():
                new_name = f"avg_{input_token}_input_token_User{user_count}.csv"
                shutil.copy(avg_response_path, result_dir / new_name)
                print(f"Copied and renamed avg_32_input_tokens.csv to {result_dir / new_name}")
            else:
                print(f"avg_32_input_tokens.csv not found in {user_folder}")

    except Exception as e:
        print(f"Unexpected error: {e}")

def extract_metrics_from_avg_response(config_file, result_dir, user_count):
    """
    Extract TTFT, latency, and latency_per_token values from the avg_32_input_tokens_User{user_count}.csv file.
    """
    try:
        with open(config_file, "r") as file:
            config = json.load(file)
        
        if config.get("random_prompt"):
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
        
        else:
            input_token = config.get("input_tokens")[0]
            avg_response_path = result_dir / f"avg_{input_token}_input_token_User{user_count}.csv"
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
        result, runtime = run_benchmark(config_file)
        if result is None:
            print("Benchmark run failed during binary search. Exiting.")
            break

        # Copy and extract metrics
        copy_avg_response(config_file, result_dir, mid)
        metrics = extract_metrics_from_avg_response(config_file, result_dir, mid)
        if any(metric is None for metric in metrics):
            print("Error extracting metrics during binary search. Exiting.")
            break

        ttft, latency_per_token, latency, throughput, total_throughput = metrics
        print(f"Binary Search - User Count: {mid}, TTFT: {ttft} ms, Latency: {latency} ms, "
              f"Latency per Token: {latency_per_token} ms/token, Throughput: {throughput} tokens/second, Total Throughput: {total_throughput} tokens/second")

        if validate_criterion(ttft, latency_per_token):
            # Threshold met; continue searching upward
            low = mid + 1
        else:
            # Threshold not met; search downward
            high = mid

    # Return the highest user count that met the criteria
    return low - 1
  
def run_benchmark_with_incremental_requests(config_file, optimal_user_count, result_dir):
    """
    Run the benchmark continuously after updating the config with the optimal user count.
    If validation fails due to saturation, reduce the user count and retry.
    """
    print(f"Starting continuous benchmark with {optimal_user_count} users...")
    try:
        while True:
            # Update config with current user count
            update_config(config_file, optimal_user_count)

            # Run the benchmark
            result, benchmark_time = run_benchmark(config_file)
            if result is None:
                print("Benchmark run failed. Stopping continuous benchmark.")
                break

            # Copy and extract metrics
            copy_avg_response(config_file, result_dir, optimal_user_count)

            metrics = extract_metrics_from_avg_response(config_file, result_dir, optimal_user_count)
            if not all(metric is not None for metric in metrics):
                print("Error extracting metrics. Exiting.")
                break
            
            ttft, latency_per_token, latency, throughput, total_throughput = metrics
            print(f"Continuous Run - User Count: {optimal_user_count}, TTFT: {ttft} ms, "
                  f"Latency: {latency} ms, Latency per Token: {latency_per_token} ms/token, "
                  f"Throughput: {throughput} tokens/second, Total Throughput: {total_throughput} tokens/second")

            # Check if performance is still acceptable
            if not validate_criterion(ttft, latency_per_token):
                print(f"Validation failed at user count {optimal_user_count}. Reducing user count.")
                optimal_user_count -= 1
                if optimal_user_count <= 0:
                    print("User count reduced to 0. Stopping benchmark.")
                    break
                continue  # Retry with reduced user count

            # Otherwise, save summary report and send data
            summary_report_path = generate_summary_report(
                config_file, optimal_user_count,
                [ttft, latency_per_token, latency, throughput, total_throughput],
                benchmark_time
            )

    except KeyboardInterrupt:
        print("\nContinuous benchmarking stopped by user.")

def generate_summary_report(config_file, optimal_user_count, metrics, benchmark_time):
    """
    Generate a summary report with the benchmark metrics.
    """
    try:
        out_dir = json.load(open(config_file))["out_dir"]
        summary_report_path = Path(out_dir) / "Results" / "summary_report.csv"
        summary_report_path.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "concurrent_user": [optimal_user_count],
            "ttft": int(metrics[0]),
            "token_latency": int(metrics[1]),
            "tokens_per_sec": round(metrics[4],3)
        }

        df = pd.DataFrame(report_data)
        df.to_csv(summary_report_path, index=False)
        print(f"Summary report generated at {summary_report_path}")
        return summary_report_path

    except Exception as e:
        print(f"Error generating summary report: {e}")

def adjust_user_count(config_file, result_dir):
    """
    Adjust the user count and run the benchmark until the threshold is met or the optimal user count is found.
    """
    with open(config_file, 'r') as file:
        config = json.load(file)

    user_count = config.get("user_counts")[0]  
    previous_user_count = 0
    increment = config.get("increment_user")[0]
    out_dir = config.get("out_dir")
    if not out_dir:
        print("Error: 'out_dir' not specified in the config file.")
        return
    
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
        copy_avg_response(config_file, result_dir, user_count)
        metrics = extract_metrics_from_avg_response(config_file, result_dir, user_count)
        if any(metric is None for metric in metrics):
            print("Error extracting metrics. Exiting.")
            break

        ttft, latency_per_token, latency, throughput, total_throughput = metrics
        print(f"User Count: {user_count}, TTFT: {ttft} ms, Latency: {latency} ms, "
              f"Latency per Token: {latency_per_token} ms/token, Throughput: {throughput} tokens/second, "
              f"Total Throughput: {total_throughput} tokens/second")

        if validate_criterion(ttft, latency_per_token):
            print(f"Threshold met for {user_count} users.")
            previous_user_count = user_count
            optimal_user_count = user_count
            final_metrics = metrics  # Store the metrics for the last valid run
            user_count += increment
        else:
            print(f"Threshold not met for {user_count} users.")
            optimal_user_count = binary_search_user_count(config_file, previous_user_count, user_count, result_dir)
            final_metrics = extract_metrics_from_avg_response(config_file, result_dir, optimal_user_count)
            break
    
    # Generate the summary report with the final metrics
    if final_metrics and optimal_user_count > 0:
        generate_summary_report(config_file, optimal_user_count, final_metrics, benchmark_time)
        return optimal_user_count  # Ensure a valid return
    else:
        print("No valid optimal user count found. Exiting.")
        return None  # Return None explicitly

def update_config(config_file, optimal_user_count):
    """
    Update the config file with the optimal user count.
    """
    with open(config_file, "r") as file:
        config = json.load(file)
    
    if optimal_user_count is not None and optimal_user_count > 0:
        config["optimal_user_count"] = optimal_user_count
        config["user_counts"] = [optimal_user_count]  # Ensure this is a valid list
        config["max_requests"] = 1
    else:
        print("Warning: No valid optimal user count found. Keeping existing configuration.")
    
    with open(config_file, "w") as file:
        json.dump(config, file, indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark automation script")
    parser.add_argument("config_file", type=str, help="Path to the benchmark configuration file")
    args = parser.parse_args()

    # Read the config file to get the output directory
    # with open(args.config_file, "r") as file:
    #     config = json.load(file)
    
    # out_dir = config.get("out_dir")
    # if not out_dir:
    #     print("Error: 'out_dir' not specified in the config file.")
        
    
    # result_dir = Path(out_dir) / "Results"
    # result_dir.mkdir(parents=True, exist_ok=True)
    
    # optimal_user_count = adjust_user_count(args.config_file, result_dir)    
    
    # if optimal_user_count is not None:
    #     run_benchmark_with_incremental_requests(args.config_file, optimal_user_count, result_dir)

