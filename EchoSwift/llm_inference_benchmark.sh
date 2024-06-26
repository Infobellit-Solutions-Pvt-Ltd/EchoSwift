#!/bin/bash

export MAX_REQUESTS=5
USER_COUNTS=(1)
# (1 3 10 30 100)
INPUT_TOKENS=(32)
# (32 64 128 256 512 1024 2048)
OUTPUT_TOKENS=(256)
# (4 8 16 32 64 128 256 512 1024 2048)
DATASET_DIR="Input_Dataset"
OUTPUT_DIR=$1
export API_URL=$2
export PROVIDER=$3

# Set MODEL_NAME only if PROVIDER is Ollama
if [ "$PROVIDER" == "Ollama" ]; then
    export MODEL_NAME=$4
fi

mkdir -p "$OUTPUT_DIR"

# Iterate over each user count
for u in "${USER_COUNTS[@]}"; do
    export NUM_USERS="$u"

    # Create a directory for the current user
    USER_DIR="${OUTPUT_DIR}/${u}_User"
    mkdir -p "$USER_DIR"

    # Iterate over each input token length
    for input_token in "${INPUT_TOKENS[@]}"; do

        # Create a single CSV file for each user and input token combination
        USER_FILE="$USER_DIR/${input_token}_input_tokens.csv"
        touch "$USER_FILE"

        # Iterate over each output token
        for output_token in "${OUTPUT_TOKENS[@]}"; do

            # Run Locust for the current combination of user, input token, and output token
            echo "Running Locust with -users = $u, input_tokens = $input_token, and output_tokens = $output_token"
            MAX_NEW_TOKENS="$output_token" INPUT_DATASET="${DATASET_DIR}/Dataset_${input_token}.csv" OUTPUT_FILE="$USER_FILE" locust -f llm_inference_master.py --headless -H "$API_URL" -u "$u" -r "$u"

            # Add additional commands or logic here if needed
            echo >> "$USER_FILE"
            echo >> "$USER_FILE"

        done

        # Calculate average after generating all CSV files for the current user and input token
        python3 Utils/avg_locust_results.py --input_csv_filename "$USER_FILE" --output_csv_filename "$USER_DIR/avg_${input_token}_input_tokens.csv" --tokens "${OUTPUT_TOKENS[@]}"
    done

done
