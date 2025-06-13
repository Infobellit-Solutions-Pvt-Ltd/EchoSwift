import pytest
from click.testing import CliRunner
from echoswift.cli import cli
import json
from unittest.mock import patch, Mock, call
import pandas as pd
from pathlib import Path

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def mock_config_file(tmp_path):
    config = {
        "out_dir": "test_results",
        "base_url": "http://localhost:8000/v1/completions",
        "inference_server": "vLLM",
        "model": "meta-llama/Meta-Llama-3-8B",
        "max_requests": 5,
        "user_counts": [3],
        "input_tokens": [32],
        "output_tokens": [256],
        "tokenizer_path": "meta-llama/Meta-Llama-3-8B"
    }
    config_file = tmp_path / "test_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)
    return str(config_file)

def test_cli_help(runner):
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'EchoSwift: LLM Inference Benchmarking Tool' in result.output

@patch('echoswift.cli.download_dataset_files')
@patch('echoswift.cli.create_config')
def test_dataprep_command(mock_create_config, mock_download, runner):
    result = runner.invoke(cli, ['dataprep'])
    assert result.exit_code == 0
    assert call("epsilondelta1982/EchoSwift-20k-Dataset") in mock_download.call_args_list
    assert call("sarthakdwi/EchoSwift-8k") in mock_download.call_args_list
    mock_create_config.assert_called_once_with('config.json')

@patch('echoswift.cli.download_dataset_files')
@patch('echoswift.cli.create_config')
def test_dataprep_command_custom_config(mock_create_config, mock_download, runner, tmp_path):
    custom_config_path = tmp_path / "custom_config.json"
    custom_config_path.parent.mkdir(parents=True, exist_ok=True)
    custom_config_path.write_text('{}')  # <-- This fixes the FileNotFoundError

    result = runner.invoke(cli, ['dataprep', '--config', str(custom_config_path)])
    assert result.exit_code == 0
    assert call("epsilondelta1982/EchoSwift-20k-Dataset") in mock_download.call_args_list
    assert call("sarthakdwi/EchoSwift-8k") in mock_download.call_args_list
    mock_create_config.assert_called_once_with(str(custom_config_path))

def test_start_command_without_config(runner):
    result = runner.invoke(cli, ['start'])
    assert result.exit_code != 0
    assert 'Error: Missing option \'--config\'' in result.output

@patch('echoswift.cli.Path')
@patch('echoswift.cli.EchoSwift')
@patch('echoswift.cli.load_config')
@patch('echoswift.cli.pd.read_csv')
@patch('echoswift.cli.pd.concat')
@patch('echoswift.cli.tabulate')
def test_start_command_with_config(mock_tabulate, mock_concat, mock_read_csv, mock_load_config, mock_echoswift, mock_path, runner, mock_config_file):
    mock_config = {
        "out_dir": "test_results",
        "base_url": "http://localhost:8000/v1/completions",
        "inference_server": "vLLM",
        "model": "meta-llama/Meta-Llama-3-8B",
        "max_requests": 5,
        "user_counts": [3],
        "input_tokens": [32],
        "output_tokens": [256],
        "tokenizer_path": "meta-llama/Meta-Llama-3-8B"
    }
    mock_load_config.return_value = mock_config
    mock_path.return_value.exists.return_value = True
    mock_path.return_value.iterdir.return_value = [Mock()]

    mock_df = pd.DataFrame({
        'output_tokens': [256],
        'throughput(tokens/second)': [100],
        'latency(ms)': [50],
        'TTFT(ms)': [10],
        'latency_per_token(ms/token)': [0.2],
    })

    mock_read_csv.return_value = mock_df
    mock_concat.return_value = mock_df

    result = runner.invoke(cli, ['start', '--config', mock_config_file])
    assert result.exit_code == 0, f"Command failed with error: {result.output}"

@patch('echoswift.cli.Path')
@patch('echoswift.cli.load_config')
def test_start_command_without_dataset(mock_load_config, mock_path, runner, mock_config_file):
    mock_config = {
        "out_dir": "test_results",
        "base_url": "http://localhost:8000/v1/completions",
        "inference_server": "vLLM",
        "model": "meta-llama/Meta-Llama-3-8B",
        "max_requests": 5,
        "user_counts": [3],
        "input_tokens": [32],
        "output_tokens": [256],
        "tokenizer_path": "meta-llama/Meta-Llama-3-8B"
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
    mock_plot.assert_called_once_with(results_dir, False)  # fix: remove 2nd argument check

def test_plot_command_with_invalid_results_dir(runner):
    result = runner.invoke(cli, ['plot', '--results-dir', '/non/existent/path'])
    assert result.exit_code != 0
    assert 'Error: Invalid value for \'--results-dir\'' in result.output
