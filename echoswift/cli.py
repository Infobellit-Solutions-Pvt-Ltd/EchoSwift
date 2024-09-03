import click
import yaml
from pathlib import Path
from echoswift.llm_inference_benchmark import EchoSwift
from echoswift.dataset import download_dataset_files
from utils.plot_results import plot_benchmark_results
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

@click.group()
def cli():
    """EchoSwift: LLM Inference Benchmarking Tool"""
    pass

@cli.command()
@click.option('--config', default='config.yaml', help='Path to the configuration file')
def start(config):
    """Start the EchoSwift benchmark with the specified config file"""
    config_path = Path(config)
    if not config_path.exists():
        raise click.FileError(config, hint='Configuration file not found')
    
    cfg = load_config(config_path)
    
    dataset_dir = Path("Input_Dataset")
    if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        logging.warning("Filtered dataset not found. Please run 'echoswift download-dataset' before starting the benchmark.")
        return

    logging.info("Using Filtered_ShareGPT_Dataset for the benchmark.")
    
    benchmark = EchoSwift(
        output_dir=cfg['out_dir'],
        api_url=cfg['base_url'],
        provider=cfg['provider'],
        model_name=cfg.get('model'),
        max_requests=cfg['max_requests'],
        user_counts=cfg['user_counts'],
        input_tokens=cfg['input_tokens'],
        output_tokens=cfg['output_tokens'],
        dataset_dir=str(dataset_dir)
    )
    
    benchmark.run_benchmark()

@cli.command()
def dataprep():
    """Download the filtered ShareGPT dataset"""
    download_dataset_files("sarthakdwi/EchoSwift-8k")

@cli.command()
@click.option('--results-dir', required=True, help='Directory containing benchmark results')
def plot(results_dir):
    """Plot graphs to analyze the benchmark results"""
    results_path = Path(results_dir)
    if not results_path.exists() or not results_path.is_dir():
        raise click.BadParameter("The specified results directory does not exist or is not a directory.")
    
    try:
        plot_benchmark_results(results_path)
        click.echo(f"Plots have been generated and saved in {results_path}")
    except Exception as e:
        click.echo(f"An error occurred while plotting results: {e}", err=True)

if __name__ == '__main__':
    cli()