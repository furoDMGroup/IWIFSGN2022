from pathlib import PurePath

# please enter a path, where you want save datasets
datasets_path = 'C:\\'


def concatenate_path_os_independent(dataset):
    return str(PurePath(datasets_path + '/' + dataset))