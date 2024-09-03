import pytest
from click.testing import CliRunner
from echoswift.cli import cli

@pytest.fixture
def runner():
    return CliRunner()

def test_cli_help(runner):
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'EchoSwift: LLM Inference Benchmarking Tool' in result.output

def test_dataprep_command(runner):
    result = runner.invoke(cli, ['dataprep'])
    assert result.exit_code == 0
    assert 'Dataprep Done' in result.output

def test_start_command_without_config(runner):
    result = runner.invoke(cli, ['start'])
    assert result.exit_code != 0
    assert 'The --config option is required' in result.output

def test_plot_command_without_results_dir(runner):
    result = runner.invoke(cli, ['plot'])
    assert result.exit_code != 0
    assert 'The --results-dir option is required' in result.output

# Add more test cases for different scenarios and edge cases