import yaml

def load_yaml(path):
    """Loads the data from a .yaml file

    Gets the data from a .yaml file, the user should specify the full path to the file.

    Arguments:
        path_to_file -- full path to the .yaml file to load

    Returns:
        data contained on the .yaml file
    """

    try:
        with open(path) as file_obj:
            config = yaml.load(file_obj, Loader=yaml.FullLoader)
        return config
    except Exception:
        raise Exception('Failed to load config file from: {}.'.format(path))