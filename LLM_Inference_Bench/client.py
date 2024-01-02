import threading
import subprocess
from util import ContainerManager, get_config
import yaml
import os


def load_config(file_path='./config.yaml'):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None

def save_config(config, file_path='./config.yaml'):
    with open(file_path, 'w') as file:
        yaml.dump(config, file)

class Benchmark:

    def __init__(self, obj: ContainerManager):
        self.obj = obj
        
        config = load_config()
        url = config["API_URL"]
        os.environ["API_URL"] = url
        
        model_name = config["MODEL_NAME"]
        
        self.obj.container_name = model_name.split("/")[1]

    @staticmethod
    def run_locust_script(script_path, aggregate_dir):
        """
        Run the Locust script.

        Args:
            script_path (str): Path to the Locust script.
            aggregate_dir (str): Aggregate directory.

        Returns:
            None
        """
        try:
            subprocess.run(["bash", script_path, aggregate_dir], check=True)
            print("Locust script executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running the shell script: {e}")

    def run_get_container_stats(self):
        """
        Run the thread to get container stats.

        Returns:
            None
        """
        stats = self.obj.get_container_stats()
        self.obj.plot(stats)

    def run_benchmark(self):
    
        if self.obj.model_loaded is None:
            self.obj.model_loaded = True
            
        if self.obj.model_loaded:
            self.obj.status = "postloading"
            
            

            stats_thread = threading.Thread(target=self.run_get_container_stats)

            script_path = "./locust.sh"  # Locust script path
            aggregate_dir = f"{self.obj.container_name}/{self.obj.status}/Locust_Test_Results"

            locust_thread = threading.Thread(target=self.run_locust_script, args=(script_path, aggregate_dir))

            # Start both threads
            locust_thread.start()
            stats_thread.start()

            # Wait for both threads to finish
            stats_thread.join()
            locust_thread.join()

            self.obj.stop_container()
            self.obj.remove_container()
            self.obj.model_loaded = None
            self.obj.status = ""


if __name__ == "__main__":

    #models_list = ["meta-llama/Llama-2-7b-chat-hf"]
    var = ContainerManager(get_config())

    bench = Benchmark(var)
    bench.run_benchmark()
