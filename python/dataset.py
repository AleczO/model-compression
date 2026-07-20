import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset

## backward, down, follow, forward, learn, left, no, off, on, right, stop, up, yes

COMMAND_WORDS = ["backward", "down", "follow", "forward", "learn", "left",
                 "no", "off", "on", "right", "stop", "up", "yes"]

LABELS = COMMAND_WORDS + ["unknown", "background"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}

SAMPLE_RATE = 16000
CLIP_LENGTH = SAMPLE_RATE

BACKGROUND_VAL_TEST_FILES = {
    "val": ["doing_the_dishes.wav"],
    "test": ["running_tap.wav"],
}

def load_split_lists(root):
    def read(name):
        with open(os.path.join(root, name)) as f:
            return set(f.read().splitlines())
    return read("validation_list.txt"), read("testing_list.txt")


def assign_split(rel_path, val_set, test_set):
    if rel_path in val_set:
        return "val"
    if rel_path in test_set:
        return "test"
    return "train"
