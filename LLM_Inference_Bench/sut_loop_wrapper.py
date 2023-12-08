import subprocess
import os
import docker
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pandas as pd
import logging
import threading

def get_config():
    configurations = {"cpu_mem": []}
    # cpus = [128, 96, 64, 32, 16, 8]
    # memory = [20, 40, 80, 150, 300, 600]
    
    cpus = [192]
    memory = [128]

    for mem in memory:
        for cpu in cpus:
            configurations["cpu_mem"].append((cpu, mem))

    return configurations

def filter_best_container_config(csv_path,model):
    # Read the CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"File not found at path {csv_path}")
        return
    except pd.errors.EmptyDataError:
        logging.error(f"The CSV file at path {csv_path} is empty.")
        return

    # Check if the required columns are present in the DataFrame
    required_columns = ['container_name', 'Assigned_Memory', 'Assigned_CPUs', 'Load_Time']
    if not df.columns.isin(required_columns).all():
        logging.error(f"The CSV file at path {csv_path} is missing required columns.")
        return

    # Filter the DataFrame based on the least Load_Time
    try:
        min_load_time_row = df.loc[df['Load_Time'].idxmin()]
    except KeyError:
        logging.error(f"'Load_Time' column is empty.")
        return

    # Create the dictionary directly from the row using `to_dict()` method
    best_config = min_load_time_row.to_dict()
    best_config_df = pd.DataFrame(best_config)
    base_floder = model.split("/")[1]
    output_file = 'best_container_config.csv'
    os.makedirs(base_floder, exist_ok=True)
    output_path = os.path.join(base_floder,output_file)
    best_config_df.to_csv(output_path, index=False)
    
def plot_bar_graph(csv_path, output_file):
    """
    Plot a bar graph of model load time based on different configurations of assigned memory and CPUs.

    Args:
        csv_path (str): The path to the CSV file containing the data for the bar graph.
        output_file (str): The name of the output file to save the bar graph as an image.

    Returns:
        None. The function saves the bar graph as an image file but does not return any value.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError as e:
        raise e
    except pd.errors.EmptyDataError as e:
        raise e

    time = df["Load_Time"]
    cpu = df["Assigned_CPUs"]
    mem = df["Assigned_Memory"][0]
    x_labels = []
    for c in cpu:
        x_labels.append(f"{c} cpus")

    min_load_time_index = time.idxmin()
    min_load_time = time[min_load_time_index]

    bars = plt.bar(x_labels, time, label=f'Least Load Time: {min_load_time:.2f} seconds')
    plt.xlabel('No of Cores Assigned')
    plt.ylabel('Model Load Time (seconds)')
    plt.title(f'Model Load Time Graph with {mem} GiB memory for each SUT')
    plt.legend()  # Add legend

    # Add time values on top of each bar
    for bar, time_val in zip(bars, time):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f'{time_val:.2f}', ha='center')

    plt.xticks(ha='right')
    plt.savefig(output_file)


class ContainerManager:
    
    def __init__(self, models, config):
        self._token = "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"
        self.port = 8181
        self._volume = os.getcwd() + "/data"
        self.models = models
        self.config = config
        self.container_name = ""
        self.model_loaded = None
        self.status = ""
        self.dataset_path = "Input_Dataset"
        
        self.container_details = {"container_name": [], "Assigned_Memory": [], "Assigned_CPUs": [], "Load_Time": []}

        try:
            self.client = docker.from_env()
        except Exception as e:
            print(f"Unable to connect to the Docker Deamon: {e}")

    def run_container(self, cpus, memory, model_id):
        docker_command = [
            "docker", "run",
            "-d",
            "--name", self.container_name,
            "--cpus", f'{cpus}',
            "--memory", f'{memory}GiB',
            "--shm-size", "1g",
            "-e", f"HUGGING_FACE_HUB_TOKEN={self._token}",
            "-p", f"{self.port}:80",
            "-v", f"{self._volume}:/data",
            "ghcr.io/huggingface/text-generation-inference:1.1.1",
            "--model-id", model_id,
            "--dtype", 'bfloat16',
            "--disable-custom-kernels"
        ]

        docker_start_command = ["docker", "start", self.container_name]

        try:
            if self.container_name in self.get_container_list():
                subprocess.run(docker_start_command, check=True)
                print("Docker container started successfully.")
            else:
                subprocess.run(docker_command, check=True)
                print("Docker container started successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running Docker container: {e}")

    def get_container_list(self):
        containers = self.client.containers.list(all=True)
        return [container.name for container in containers]

    def stop_container(self):
        container = self.client.containers.get(self.container_name)
        container.stop()
        
    def remove_container(self):
        container = self.client.containers.get(self.container_name)
        container.remove()

    @staticmethod
    def calculate_percentage(stats):
        cpu_stats = stats.get('cpu_stats', {})
        precpu_stats = stats.get('precpu_stats', {})
        cpu_delta = cpu_stats.get('cpu_usage', {}).get('total_usage', 0) - precpu_stats.get('cpu_usage', {}).get(
            'total_usage', 0)
        system_cpu_delta = cpu_stats.get('system_cpu_usage', 0) - precpu_stats.get('system_cpu_usage', 0)
        num_cpus = cpu_stats.get('online_cpus', 1)
        cpu_percentage = (cpu_delta / system_cpu_delta) * num_cpus * 100.0

        memory_stats = stats.get('memory_stats', {})
        memory_usage = memory_stats.get('usage', 0)
        memory_limit = memory_stats.get('limit', 1)  # Default to 1 if not available
        mem_percentage = (memory_usage / memory_limit) * 100

        return cpu_percentage, mem_percentage    

    def get_container_stats(self):

        container_stats = {
            'MemoryPercentage': [],
            'CpuPercentage': [],
            'timestamps': []
        }
        
        model_name, _, cpu, mem = self.container_name.split("_")
        base_folder = f"{model_name}/SUT_{mem}/{self.status}/Profiling_Results/ContainerStats"
        output_file = f"{self.status}_stats_{cpu}.csv"
        os.makedirs(base_folder, exist_ok=True)
        output_csv = os.path.join(base_folder, output_file)

        try:
            container = self.client.containers.get(self.container_name)

            check_list = []
            cpu_list = []
            start = time.perf_counter()

            for i, stats in enumerate(container.stats(stream=True, decode=True)):
                cpu_perc, mem_perc = ContainerManager.calculate_percentage(stats)
                container_stats["MemoryPercentage"].append(mem_perc)
                container_stats["CpuPercentage"].append(cpu_perc)
                container_stats["timestamps"].append(datetime.now())
                
                if self.model_loaded:
                                        
                    print("Collecting the container stats while locust benchmark script is running")
                    
                    print("Mem Percentage:", mem_perc)
                    print("CPU Percentage :", cpu_perc)
                    
                    if len(check_list) < 5 and len(cpu_list) < 5:
                        check_list.insert(0, mem_perc)
                        cpu_list.insert(0,int(cpu_perc))
                    else:
                        check_list.pop()
                        check_list.insert(0, mem_perc)
                        cpu_list.pop()
                        cpu_list.insert(0,int(cpu_perc))
                        
                    if len(set(check_list)) == 1 and len(set(cpu_list)) == 1 and i > 5:
                        print("Locust script stopped.")
                        break
                    
                elif len(check_list) < 5:
                    check_list.insert(0, mem_perc)
                else:
                    check_list.pop()
                    check_list.insert(0, mem_perc)


                if len(set(check_list)) == 1 and i > 5:
                    end = time.perf_counter()
                    load_time = end - start
                    self.container_details["Load_Time"].append(load_time)
                    self.model_loaded = True
                    
                    print(f"It took {load_time} seconds to successfully load the model.")
                    break

            print(f"Docker container stats for '{self.container_name}' saved successfully.")

        except docker.errors.NotFound:
            print(f"Container '{self.container_name}' not found.")
        except docker.errors.APIError as e:
            print(f"Error retrieving stats for container '{self.container_name}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        stats_df = pd.DataFrame(container_stats)
        stats_df.to_csv(output_csv, index=False)
        
        return stats_df

    def plot(self, stats):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        timestamps = stats["timestamps"].to_list()
        mem_perc = stats["MemoryPercentage"].to_list()
        cpu_perc = stats["CpuPercentage"].to_list()

        max_mem_value = max(mem_perc)
        max_cpu_value = max(cpu_perc)

        ax1.plot(timestamps, mem_perc, linestyle='-', color='r', label=f"Max Memory Percentage: {max_mem_value:.3f} %")
        ax1.set_title('Time vs Memory')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Memory Usage')

        ax2.plot(timestamps, cpu_perc, linestyle='-', color='g', label=f'Max CPU Percentage: {max_cpu_value:.3f} %')
        ax2.set_title('Time vs CPU')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('CPU Percentage (%)')

        ax1.legend()
        ax2.legend()

        plt.tight_layout()

        
        model_name, _, cpu, mem = self.container_name.split("_")
        plot_filename = f"plot_{self.status}_{cpu}.png"
        base_folder = f"{model_name}/SUT_{mem}/{self.status}/Profiling_Results/Plots"  # creating separate folders inside 'Plots' for containers based on memory
        os.makedirs(base_folder, exist_ok=True)
        output_file = os.path.join(base_folder, plot_filename)
        plt.savefig(output_file)

        print(f"Plot saved to {plot_filename} successfully")
        
    
    def run_benchmark(self, api_url, name, mem, cpu):

        if self.model_loaded:
            self.status = "postloading"
            
            print()
            print(f"Collecting container stats {self.status} while running the benchmark.")
            
            os.environ["API_URL"] = api_url
        
            stats_thread = threading.Thread(target=self.run_get_container_stats)
            
            script_path = "./locust.sh"
            aggregate_dir = f"{name}/SUT_{mem}m/postloading/Locust_Test_Results/{cpu}"
            locust_thread = threading.Thread(target=self.run_locust_script, args=(script_path, aggregate_dir))
            
            # Start both threads
            locust_thread.start()
            stats_thread.start()
            
            # Wait for both threads to finish
            stats_thread.join()
            locust_thread.join()

    def run_get_container_stats(self):
        stats = self.get_container_stats()
        self.plot(stats)

    def run_locust_script(self, script_path, aggregate_dir):
        try:
            subprocess.run(["bash", script_path, aggregate_dir], check=True)
            print("Locust script executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running the shell script: {e}")

    def main(self):

        for model in self.models:
            for i, (cpu, mem) in enumerate(self.config["cpu_mem"]):
                    
                self.port += 1

                self.container_name = model.split("/")[1]
                self.container_name = f"{self.container_name}_BF16_{cpu}c_{mem}m"
                self.container_details["container_name"].append(self.container_name)
                self.container_details["Assigned_Memory"].append(mem)
                self.container_details["Assigned_CPUs"].append(cpu)

                print("###" * 20)
                print()
                print(f"Assigned port number for {self.container_name} is {self.port}")
                print()
                print(f"The resources attached for {self.container_name} are {cpu}cpus and {mem}GiB memory")
                print()
                
                self.run_container(cpu, mem, model)
                
                if self.model_loaded is None:
                    self.model_loaded = False
                    self.status = "preloading"
                
                
                print()
                print(f"Collecting container stats {self.status}")
                stats = self.get_container_stats()
                self.plot(stats)

                print(f"Successfully launched {i + 1} models till now")
                
                url = f"http://localhost:{self.port}/generate_stream"
                
                self.run_benchmark(api_url=url, name=model.split("/")[1], mem=mem, cpu=cpu)
                
                self.stop_container()
                self.model_loaded = None
                self.status = ""
                
            
            # model_name = self.container_name.split("_")[0]
            # path = f"{model_name}/SUT_{mem}m/Model_Load_Time_Analysis_{mem}.csv"
            # load_time_df = pd.DataFrame(self.container_details)
            # load_time_df.to_csv(path, index=False)


if __name__ == "__main__":
    
    models_list = ["meta-llama/Llama-2-7b-chat-hf"]

    config_data = get_config()

    container_manager = ContainerManager(models_list, config_data)

    container_manager.main()
    
    container_manager.remove_container()
    
    # To plot a bar graph for different load times for the SUTs
    # best_load_time_file = "Load_Time_graph.png"
    # filepath = "Llama-2-7b-chat-hf/Model_Load_Time_Analysis.csv"
    # plot_bar_graph(filepath, best_load_time_file)
