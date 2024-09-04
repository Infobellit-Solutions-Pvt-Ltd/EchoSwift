import argparse
import csv
import logging
import sys
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_csv(filename: str) -> List[List[str]]:
    try:
        with open(filename, 'r', newline='') as file:
            return list(csv.reader(file))
    except FileNotFoundError:
        logging.error(f"Input file not found: {filename}")
        sys.exit(1)
    except PermissionError:
        logging.error(f"Permission denied when trying to read: {filename}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading input file {filename}: {str(e)}")
        sys.exit(1)

def calculate_average(rows: List[List[str]], column_indices: List[int], start: int, end: int) -> List[Optional[float]]:
    try:
        values = [[float(row[i]) for i in column_indices] for row in rows[start:end-1] if row]
        return [sum(column) / len(column) if column else None for column in zip(*values)]
    except ValueError as e:
        logging.error(f"Error calculating average: {str(e)}. Check if all values are numeric.")
        return [None] * len(column_indices)

def calculate_averages(input_csv_filename: str, output_csv_filename: str, tokens: List[int]):
    column_names = ["throughput(tokens/second)", "latency(ms)", "TTFT(ms)", "latency_per_token(ms/tokens)"]
    rows = read_csv(input_csv_filename)

    if not rows:
        logging.error(f"Input file is empty: {input_csv_filename}")
        sys.exit(1)

    header = rows[0]
    try:
        column_indices = [header.index(column) for column in column_names]
    except ValueError as e:
        logging.error(f"Error finding column indices: {str(e)}. Check if all required columns are present.")
        sys.exit(1)

    empty_line_indices = [i for i, row in enumerate(rows) if not any(row)]
    if not empty_line_indices or empty_line_indices[-1] != len(rows) - 1:
        rows.append([''] * len(rows[0]))
    empty_line_indices = empty_line_indices + [len(rows)]

    try:
        with open(output_csv_filename, mode='w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["output tokens"] + column_names)

            for i in range(len(empty_line_indices)):
                start_index = 1 if i == 0 else empty_line_indices[i - 1] + 1
                end_index = empty_line_indices[i]
                average = calculate_average(rows, column_indices, start_index, end_index)

                if len(average) > 1 and i // 2 < len(tokens):
                    writer.writerow([tokens[i // 2]] + average)

        logging.info(f"Average calculations complete. Results written to {output_csv_filename}")
    except PermissionError:
        logging.error(f"Permission denied when trying to write to: {output_csv_filename}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error writing to output file {output_csv_filename}: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate averages from Locust results")
    parser.add_argument('--input_csv_filename', required=True, help='Input CSV file path')
    parser.add_argument('--output_csv_filename', required=True, help='Output CSV file path')
    parser.add_argument('--tokens', nargs='+', type=int, required=True, help='List of different output_tokens')
    args = parser.parse_args()

    calculate_averages(args.input_csv_filename, args.output_csv_filename, args.tokens)

# Example command to run this file:
# python3 avg_locust_results.py --input_csv_filename "Results_vLLM_Llama3_8b_32in_256out/100_User/32_input_tokens.csv" --output_csv_filename "Results_vLLM_Llama3_8b_32in_256out/100_User/avg_32_input_tokens.csv" --tokens 256