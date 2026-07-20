import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset

COMMAND_WORDS = ["backward", "down", "follow", "forward", "learn", "left",
                 "no", "off", "on", "right", "stop", "up", "yes"]

LABELS = COMMAND_WORDS + ["unknown", "background"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
BACKGROUND_IDX = LABEL_TO_IDX["background"]

SAMPLE_RATE = 16000
CLIP_LENGTH = SAMPLE_RATE  # 1 second

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


class SpeechCommandsDataset(Dataset):
    def __init__(self, root, split, unknown_per_split=3500, seed=42):
        self.root = root
        self.split = split
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_mels=40
        )

        val_set, test_set = load_split_lists(root)
        rng = random.Random(seed)

        # touple: (path, label_idx, offset) 
        self.samples = []

        for word in COMMAND_WORDS:
            word_dir = os.path.join(root, word)
            for fname in os.listdir(word_dir):
                rel_path = f"{word}/{fname}"
                if assign_split(rel_path, val_set, test_set) == split:
                    self.samples.append(
                        (os.path.join(word_dir, fname), LABEL_TO_IDX[word], None)
                    )


        all_word_dirs = [d for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d))
                          and d not in COMMAND_WORDS
                          and d != "_background_noise_"]
        
        unknown_candidates = []
        for word in all_word_dirs:
            word_dir = os.path.join(root, word)
            for fname in os.listdir(word_dir):
                rel_path = f"{word}/{fname}"
                if assign_split(rel_path, val_set, test_set) == split:
                    unknown_candidates.append(os.path.join(word_dir, fname))


        rng.shuffle(unknown_candidates)

        for path in unknown_candidates[:unknown_per_split]:
            self.samples.append((path, LABEL_TO_IDX["unknown"], None))


        noise_dir = os.path.join(root, "_background_noise_")
        noise_files = [f for f in os.listdir(noise_dir) if f.endswith(".wav")]
    
        if split == "train":
            split_noise_files = [f for f in noise_files
                                  if f not in BACKGROUND_VAL_TEST_FILES["val"]
                                  and f not in BACKGROUND_VAL_TEST_FILES["test"]]
        else:
            split_noise_files = BACKGROUND_VAL_TEST_FILES[split]

        n_background = unknown_per_split
        for _ in range(n_background):
            fname = rng.choice(split_noise_files)
            path = os.path.join(noise_dir, fname)
            info = torchaudio.info(path)
            max_start = info.num_frames - CLIP_LENGTH
            offset = rng.randint(0, max_start)
            self.samples.append((path, BACKGROUND_IDX, offset))
        

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        path, label_idx, offset = self.samples[idx]

        if label_idx == BACKGROUND_IDX:
            waveform, _ = torchaudio.load(
                path, frame_offset=offset, num_frames=CLIP_LENGTH
            )
        else:
            waveform, _ = torchaudio.load(path)
            if waveform.shape[1] < CLIP_LENGTH:
                pad = CLIP_LENGTH - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            else:
                waveform = waveform[:, :CLIP_LENGTH]

        mel = self.transform(waveform)
        log_mel = torch.log(mel + 1e-6)

        return log_mel, label_idx