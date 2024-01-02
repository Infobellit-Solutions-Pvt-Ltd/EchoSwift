#!/bin/bash

# Start the TGI server
python3 server.py

# Run the benchmark
python3 benchmark.py

# Display the results with streamlit application
streamlit run Utils/app.py