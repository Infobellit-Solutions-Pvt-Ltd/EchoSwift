import subprocess
import os
import docker
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pandas as pd



def get_config():
    configurations = {"cpu_mem": []}
    cpus = [96]
    memory = [128]

    for mem in memory:
        for cpu in cpus:
            configurations["cpu_mem"].append((cpu, mem))

    return configurations


class ContainerManager:

    def __init__(self, config):
        self.token = ""
        self.port = None
        self.volume = os.getcwd() + "/data"
        self.model = ""
        self.config = config
        self.container_name = ""
        self.model_loaded = None
        self.status = "preloading"

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
            "--cpuset-cpus", "0-95",
            "--memory", f'{memory}GiB',
            "--shm-size", "1g",
            "-e", f"HUGGING_FACE_HUB_TOKEN={self.token}",
            "-p", f"{self.port}:80",
            "-v", f"{self.volume}:/data",
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

        base_folder = f"{self.container_name}/{self.status}/Profiling_Results/ContainerStats"
        output_file = f"{self.status}_stats.csv"
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

                    if len(check_list) < 5 and len(cpu_list) < 5:
                        check_list.insert(0, mem_perc)
                        cpu_list.insert(0, int(cpu_perc))
                    else:
                        check_list.pop()
                        check_list.insert(0, mem_perc)
                        cpu_list.pop()
                        cpu_list.insert(0, int(cpu_perc))

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
                    #self.model_loaded = True  # Env

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

        plot_filename = f"{self.status}_plot.png"

        base_folder = f"{self.container_name}/{self.status}/Profiling_Results/Plots"
        os.makedirs(base_folder, exist_ok=True)
        output_file = os.path.join(base_folder, plot_filename)
        plt.savefig(output_file)

        print(f"Plot saved to {plot_filename} successfully")
