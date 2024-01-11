from util import get_config, ContainerManager, load_config, save_config
import os
import yaml


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
    
    if existing_config:
        url = f"http://localhost:{var.port}/generate_stream"
        existing_config['API_URL'] = url
        
        save_config(existing_config)


if __name__ == "__main__":

    #model_name = "meta-llama/Llama-2-7b-chat-hf"
    obj = ContainerManager(get_config())

    main(obj)
    
