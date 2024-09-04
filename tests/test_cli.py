import pytest
from click.testing import CliRunner
from echoswift.cli import cli
from pathlib import Path
import json
from unittest.mock import patch, Mock, MagicMock

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def mock_config_file(tmp_path):
    config = {
        "out_dir": "test_results",
        "base_url": "http://localhost:8000/v1/completions",
        "provider": "vLLM",
        "model": "meta-llama/Meta-Llama-3-8B",
        "max_requests": 5,
        "user_counts": [3],
        "input_tokens": [32],
        "output_tokens": [256]
    }
    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)
    return str(config_file)

def test_cli_help(runner):
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'EchoSwift: LLM Inference Benchmarking Tool' in result.output
    assert 'Usage:' in result.output
    assert 'start' in result.output
    assert 'dataprep' in result.output
    assert 'plot' in result.output

@patch('echoswift.cli.download_dataset_files')
@patch('echoswift.cli.create_config')
def test_dataprep_command(mock_create_config, mock_download, runner):
    result = runner.invoke(cli, ['dataprep'])
    assert result.exit_code == 0
    mock_download.assert_called_once_with("sarthakdwi/EchoSwift-8k")
    mock_create_config.assert_called_once_with('config.json')
    assert "Downloading the filtered ShareGPT dataset..." in result.output
    assert "Creating configuration file..." in result.output
    assert "Data preparation completed." in result.output

@patch('echoswift.cli.download_dataset_files')
@patch('echoswift.cli.create_config')
def test_dataprep_command_custom_config(mock_create_config, mock_download, runner):
    result = runner.invoke(cli, ['dataprep', '--config', 'custom_config.json'])
    assert result.exit_code == 0
    mock_download.assert_called_once_with("sarthakdwi/EchoSwift-8k")
    mock_create_config.assert_called_once_with('custom_config.json')

def test_start_command_without_config(runner):
    result = runner.invoke(cli, ['start'])
    assert result.exit_code != 0
    assert 'Error: Missing option \'--config\'' in result.output

@patch('echoswift.cli.Path')
@patch('echoswift.cli.EchoSwift')
@patch('echoswift.cli.load_config')
@patch('echoswift.cli.pd.read_csv')
@patch('echoswift.cli.tabulate')
def test_start_command_with_config(mock_tabulate, mock_read_csv, mock_load_config, mock_echoswift, mock_path, runner, mock_config_file):
    # Mock the config loading
    mock_config = {
        "out_dir": "test_results",
        "base_url": "http://localhost:8000/v1/completions",
        "provider": "vLLM",
        "model": "meta-llama/Meta-Llama-3-8B",
        "max_requests": 5,
        "user_counts": [3],
        "input_tokens": [32],
        "output_tokens": [256]
    }
    mock_load_config.return_value = mock_config

    # Mock the dataset directory to exist and have files
    mock_path.return_value.exists.return_value = True
    mock_path.return_value.iterdir.return_value = [Mock()]
    
    mock_benchmark_instance = mock_echoswift.return_value

    # Mock DataFrame and tabulate
    mock_df = MagicMock()
    mock_df.round.return_value = mock_df
    mock_read_csv.return_value = mock_df

    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['start', '--config', mock_config_file])
    
    assert result.exit_code == 0, f"Command failed with error: {result.output}"
    mock_echoswift.assert_called_once_with(
        output_dir=mock_config['out_dir'],
        api_url=mock_config['base_url'],
        provider=mock_config['provider'],
        model_name=mock_config['model'],
        max_requests=mock_config['max_requests'],
        user_counts=mock_config['user_counts'],
        input_tokens=mock_config['input_tokens'],
        output_tokens=mock_config['output_tokens'],
        dataset_dir=str(mock_path.return_value)
    )
    mock_benchmark_instance.run_benchmark.assert_called_once()
    mock_read_csv.assert_called()
    mock_tabulate.assert_called()

@patch('echoswift.cli.Path')
@patch('echoswift.cli.load_config')
def test_start_command_without_dataset(mock_load_config, mock_path, runner, mock_config_file):
    mock_config = {
        "out_dir": "test_results",
        "base_url": "http://localhost:8000/v1/completions",
        "provider": "vLLM",
        "model": "meta-llama/Meta-Llama-3-8B",
        "max_requests": 5,
        "user_counts": [3],
        "input_tokens": [32],
        "output_tokens": [256]
    }
    mock_load_config.return_value = mock_config
    mock_path.return_value.exists.return_value = False
    mock_path.return_value.iterdir.return_value = []

    result = runner.invoke(cli, ['start', '--config', mock_config_file])
    
    assert result.exit_code != 0
    assert "Filtered dataset not found" in result.output

def test_plot_command_without_results_dir(runner):
    result = runner.invoke(cli, ['plot'])
    assert result.exit_code != 0
    assert 'Error: Missing option \'--results-dir\'' in result.output

@patch('echoswift.cli.plot_benchmark_results')
def test_plot_command_with_results_dir(mock_plot, runner, tmp_path):
    results_dir = tmp_path / "test_results"
    results_dir.mkdir()
    
    result = runner.invoke(cli, ['plot', '--results-dir', str(results_dir)])
    
    assert result.exit_code == 0
    mock_plot.assert_called_once_with(results_dir)
    assert f"Plots have been generated and saved in {results_dir}" in result.output

def test_plot_command_with_invalid_results_dir(runner):
    result = runner.invoke(cli, ['plot', '--results-dir', '/non/existent/path'])
    assert result.exit_code != 0
    assert 'Error: Invalid value for \'--results-dir\'' in result.output