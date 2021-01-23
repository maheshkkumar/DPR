import os


def check_folder(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
