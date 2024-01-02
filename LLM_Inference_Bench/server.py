from util import get_config, ContainerManager
import os
import yaml


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


def main(var: ContainerManager):

    existing_config = load_config()
    
    cpu, mem = var.config["cpu_mem"][0]
    
    var.model = existing_config["MODEL_NAME"]
    var.token = existing_config["HF_TOKEN"]
    var.port = existing_config["PORT"]
    
    

    var.container_name = var.model.split("/")[1]

    var.run_container(cpu, mem, var.model)
    if var.model_loaded is None:
        var.model_loaded = False

    print()
    print(f"Collecting container stats {var.status}")
    stats = var.get_container_stats()
    var.plot(stats)

    print("Model loaded Successfully.")

    
    #os.environ["API_URL"] = url

    
    if existing_config:
        url = f"http://localhost:{var.port}/generate_stream"
        existing_config['API_URL'] = url
        
        save_config(existing_config)


if __name__ == "__main__":

    #model_name = "meta-llama/Llama-2-7b-chat-hf"
    obj = ContainerManager(get_config())

    main(obj)
    
