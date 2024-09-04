import pytest
from click.testing import CliRunner
from echoswift.cli import cli
from pathlib import Path
import yaml

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
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file

def test_cli_help(runner):
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'EchoSwift: LLM Inference Benchmarking Tool' in result.output
    assert 'Usage:' in result.output
    assert 'start' in result.output
    assert 'dataprep' in result.output
    assert 'plot' in result.output

def test_dataprep_command(runner, mocker):
    mock_download = mocker.patch('echoswift.cli.download_dataset_files')
    result = runner.invoke(cli, ['dataprep'])
    assert result.exit_code == 0
    mock_download.assert_called_once_with("sarthakdwi/EchoSwift-8k")

def test_start_command_without_config(runner):
    result = runner.invoke(cli, ['start'])
    assert result.exit_code != 0
    assert 'Error: Missing option \'--config\'' in result.output

def test_start_command_with_config(runner, mock_config_file, mocker):
    mock_benchmark = mocker.patch('echoswift.cli.EchoSwift')
    mock_benchmark_instance = mock_benchmark.return_value
    
    result = runner.invoke(cli, ['start', '--config', str(mock_config_file)])
    
    assert result.exit_code == 0
    mock_benchmark.assert_called_once()
    mock_benchmark_instance.run_benchmark.assert_called_once()

def test_plot_command_without_results_dir(runner):
    result = runner.invoke(cli, ['plot'])
    assert result.exit_code != 0
    assert 'Error: Missing option \'--results-dir\'' in result.output

def test_plot_command_with_results_dir(runner, tmp_path, mocker):
    mock_plot = mocker.patch('echoswift.cli.plot_benchmark_results')
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