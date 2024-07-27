import yaml

def retrieve_hparams(path):
    with open(path, 'r') as file:
        hparams = yaml.safe_load(file)
    return hparams