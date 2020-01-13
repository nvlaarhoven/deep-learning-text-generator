import os

from utils.constants import INPUT_FILE


def disable_tensorflow_info_logs():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def load_data():
    data = open(INPUT_FILE).read()
    return data.lower()
