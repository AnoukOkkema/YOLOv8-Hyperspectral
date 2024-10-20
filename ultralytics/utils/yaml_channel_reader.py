import yaml

def get_channels_from_yaml(file_path):
    with open(file_path, 'r') as file:
        # Load the YAML file
        yaml_content = yaml.safe_load(file)
        
        # Check if 'ch' exists in the loaded YAML content
        if 'ch' in yaml_content:
            return yaml_content['ch']
        else:
            print("'ch' key not found in the YAML file.")
            return None