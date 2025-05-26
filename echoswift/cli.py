import click
import json
from pathlib import Path
from echoswift.llm_inference_benchmark import EchoSwift
from echoswift.dataset import download_dataset_files
from echoswift.utils.plot_results import plot_benchmark_results 
import logging
from tabulate import tabulate
import pandas as pd
from echoswift.optimaluser import adjust_user_count, run_benchmark_with_incremental_requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """
    EchoSwift: LLM Inference Benchmarking Tool

    \b
    Usage:
    1. Run 'echoswift dataprep' to download the dataset and create config.json
    2. Run 'echoswift start --config path/to/config.json' to start the benchmark
    3. Run 'echoswift plot --results-dir path/to/benchmark_results' to generate plots
    4. Run 'echoswift optimaluserrun --config path/to/config.json' to find optimal user count

    For more detailed information, visit: \n
    https://github.com/Infobellit-Solutions-Pvt-Ltd/EchoSwift
    """
    pass

def create_config(output='config.json'):
    config = {
        "_comment": "EchoSwift Configuration",
        "out_dir": "test_results",
        "base_url": "http://10.216.178.15:8000/v1/completions",
        "inference_server": "vLLM",
        "model": "meta-llama/Meta-Llama-3-8B",
        "random_prompt": False,
        "max_requests": 5,
        "user_counts": [3],
        "input_tokens": [32],
        "output_tokens": [256],
    }

    output_path = Path(output)
    if output_path.exists():
        click.echo(f"The file {output} already exists. please validate the config file.")

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    click.echo(f"Configuration file created: {output_path}")
    click.echo("Please review and modify this file before running the benchmark.")

@cli.command()
@click.option('--config', default='config.json', help='Name of the output configuration file')
def dataprep(config):
    """Download the filtered ShareGPT dataset and create the config.json file"""
    config_path = Path(config)
    cfg = load_config(config_path)

    click.echo("Downloading the filtered ShareGPT dataset...")
    download_dataset_files("epsilondelta1982/EchoSwift-20k")
    download_dataset_files("sarthakdwi/EchoSwift-8k")

    # Create config
    click.echo("\nCreating configuration file...")
    create_config(config)
    
    click.echo("Data preparation completed. You're now ready to run the benchmark.")

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to the configuration file')
def start(config):
    """Start the EchoSwift benchmark using the specified config file"""
    config_path = Path(config)
    cfg = load_config(config_path)
    
    
    dataset_dir = Path("Input_Dataset")
    
    if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        error_msg = "Filtered dataset not found. Please run 'echoswift dataprep' before starting the benchmark."
        logging.error(error_msg)
        click.echo(error_msg, err=True)
        raise click.Abort()
    
    logging.info("Using Filtered_ShareGPT_Dataset for the benchmark.")
    
    try:
        if cfg.get('random_prompt'):
            print("inside if condition")
            # Use random queries from Dataset.csv
            benchmark = EchoSwift(
                output_dir=cfg['out_dir'],
                api_url=cfg['base_url'],
                inference_server=cfg['inference_server'],
                model_name=cfg.get('model'),
                max_requests=cfg['max_requests'],
                user_counts=cfg['user_counts'],
                dataset_dir=str(dataset_dir) +"/"+ "EchoSwift-20k",
                random_prompt=cfg['random_prompt']
            )

            benchmark.run_benchmark()

            # Pretty print results after each user count completes
            all_results = []
            for u in cfg['user_counts']:
                user_dir = Path(cfg['out_dir']) / f"{u}_User"
                avg_file = user_dir / "avg_Response.csv"
                if avg_file.exists():
                    df = pd.read_csv(avg_file)
                    df['users'] = u
                    all_results.append(df)

            if all_results:
                combined_df = pd.concat(all_results, ignore_index=True)
                combined_df = combined_df[['users', 'input_tokens', 'output_tokens', 'throughput(tokens/second)', 'latency(ms)', 'TTFT(ms)', 'latency_per_token(ms/token)']]
                combined_df = combined_df.round(3)
                
                # Sort the DataFrame
                combined_df = combined_df.sort_values(['users', 'input_tokens', 'output_tokens'])

                click.echo(tabulate(combined_df, headers='keys', tablefmt='pretty', showindex=False))

                click.echo("Tests completed successfully !!")


        else:
            # Use all parameters from the config
            benchmark = EchoSwift(
                output_dir=cfg['out_dir'],
                api_url=cfg['base_url'],
                inference_server=cfg['inference_server'],
                model_name=cfg.get('model'),
                max_requests=cfg['max_requests'],
                user_counts=cfg['user_counts'],
                input_tokens=cfg['input_tokens'],
                output_tokens=cfg['output_tokens'],
                dataset_dir=str(dataset_dir) +"/"+ "EchoSwift-8k"
            )

            benchmark.run_benchmark()
            
            # Pretty print results after each user count completes
            all_results = []
            for u in cfg['user_counts']:
                user_dir = Path(cfg['out_dir']) / f"{u}_User"
                for input_token in cfg['input_tokens']:
                    avg_file = user_dir / f"avg_{input_token}_input_tokens.csv"
                    if avg_file.exists():
                        df = pd.read_csv(avg_file)
                        df['users'] = u
                        df['input_tokens'] = input_token
                        all_results.append(df)

            if all_results:
                combined_df = pd.concat(all_results, ignore_index=True)
                combined_df = combined_df[['users', 'input_tokens', 'output_tokens', 'throughput(tokens/second)', 'latency(ms)', 'TTFT(ms)', 'latency_per_token(ms/token)']]
                combined_df = combined_df.round(3)
                
                # Sort the DataFrame
                combined_df = combined_df.sort_values(['users', 'input_tokens', 'output_tokens'])

                click.echo(tabulate(combined_df, headers='keys', tablefmt='pretty', showindex=False))

                click.echo("Tests completed successfully !!")
                        
    except Exception as e:
        error_msg = f"An error occurred while running the benchmark: {str(e)}"
        logging.error(error_msg)
        click.echo(error_msg, err=True)
        raise click.Abort()

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to the configuration file')
def optimaluserrun(config):
    """Start the EchoSwift benchmark using the specified config file for finding optimal users"""
    config_path = Path(config)
    cfg = load_config(config_path)
    
    dataset_dir = Path("Input_Dataset") 
    if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        error_msg = "Filtered dataset not found. Please run 'echoswift dataprep' before starting the benchmark."
        logging.error(error_msg)
        click.echo(error_msg, err=True)
        raise click.Abort()

    logging.info("Using Filtered_ShareGPT_Dataset for the benchmark.")

    try:
        out_dir = cfg.get("out_dir")
        if not out_dir:
            print("Error: 'out_dir' not specified in the config file.")
        
        result_dir = Path(out_dir) / "Results"
        result_dir.mkdir(parents=True, exist_ok=True)
    
        optimal_user_count=adjust_user_count(config, result_dir)
        click.echo(f"Optimal user count is : {optimal_user_count}")
        if optimal_user_count is not None:
          run_benchmark_with_incremental_requests(config, optimal_user_count, result_dir)
        else:
          click.echo("Error: Could not determine an optimal user count. Exiting.")

    except Exception as e:
        error_msg = f"An error occurred while running the benchmark: {str(e)}"
        logging.error(error_msg)
        click.echo(error_msg, err=True)
        raise click.Abort()

@cli.command()
@click.option('--results-dir', required=True, type=click.Path(exists=True), help='Directory containing benchmark results')
def plot(results_dir, config="config.json"):
    """Plot graphs using benchmark results"""
    results_path = Path(results_dir)
    config_path = Path(config)
    cfg = load_config(config_path)

    if not results_path.is_dir():
        raise click.BadParameter("The specified results directory is not a directory.")
    
    try:
        if cfg.get('random_prompt'):
            random_prompt = True
        else:
            random_prompt = False
 
        plot_benchmark_results(results_path, random_prompt)
        click.echo(f"Plots have been generated and saved in {results_path}")
    except Exception as e:
        click.echo(f"An error occurred while plotting results: {e}", err=True)

if __name__ == '__main__':
    cli()
