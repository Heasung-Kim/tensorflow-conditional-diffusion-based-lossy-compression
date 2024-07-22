import os
import tarfile
import numpy as np

def load_dataset(data_config):
    """
    Downloads and loads a variety of standard benchmark sequence datasets.
    Arguments:
        data_config (dict): dictionary containing data configuration arguments
    Returns:
        tuple of (train, val), each of which is a PyTorch dataset.
    """
    data_path = data_config["data_path"]  # path to data directory
    if data_path is not None:
        assert os.path.exists(data_path), "Data path {} not found.".format(data_path)

    # the name of the dataset to load
    dataset_name = data_config["dataset_name"]
    dataset_name = dataset_name.lower()  # cast dataset_name to lower case
    train = val = None
    if dataset_name == "your_dataset":
        pass
    else:
        raise Exception("Dataset name not found.")

    return train, val
