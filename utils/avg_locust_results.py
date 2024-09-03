import argparse
import csv


def calculate_averages(input_csv_filename, output_csv_filename, tokens):
    column_names = ["throughput(tokens/second)", "latency(ms)", "TTFT(ms)", "latency_per_token(ms/tokens)"]
    with open(input_csv_filename, 'r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    header = rows[0]

    # Map column names to their indices
    column_indices = [header.index(column) for column in column_names]

    def calculate_average(start, end):
        values = [[float(row[i]) for i in column_indices] for row in rows[start:end-1] if row]
        averages = [sum(column) / len(column) if column else None for column in zip(*values)]
        return averages

    # Find the indices of the empty lines
    empty_line_indices = [i for i, row in enumerate(rows) if not any(row)]

    if not empty_line_indices or empty_line_indices[-1] != len(rows) - 1:
        rows.append([''] * len(rows[0]))

    empty_line_indices = empty_line_indices + [len(rows)]

    # writing the avg values into a csv
    with open(output_csv_filename, mode='w', newline="") as file:
        writer = csv.writer(file)
        # Write the header (field names)
        writer.writerow(["output tokens"] + [f"{column}" for column in column_names])

        for i in range(len(empty_line_indices)):
            if i == 0:
                start_index = 1       # Skip the header row
            else:
                start_index = empty_line_indices[i - 1] + 1

            end_index = empty_line_indices[i] if i < len(empty_line_indices) else len(rows)
            average = calculate_average(start_index, end_index)     # Calculate average for specified columns

            if len(average) > 1:
                writer.writerow([tokens[i // 2]] + average)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_filename', required=True, help='Input CSV file path')
    parser.add_argument('--output_csv_filename', required=True, help='Output CSV file path')
    parser.add_argument('--tokens', nargs='+', type=int, help='List of different output_tokens')
    args = parser.parse_args()

    calculate_averages(args.input_csv_filename, args.output_csv_filename, args.tokens)


''' example command to run this file
python3 avg_locust_results.py --input_csv_filename "Llama-2-7b-chat-hf/SUT_32m/postloading/Locust_Test_Results/1_User/32_input_tokens.csv" --output_csv_filename "Llama-2-7b-chat-hf/SUT_32m/postloading/Locust_Test_Results/1_User/avg_32_input_tokens.csv" --tokens 64 128 256 '''
