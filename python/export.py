import struct
import torch
from model import Res8

CHECKPOINT_PATH = "../data/checkpoints/best_model.pth"
OUTPUT_PATH = "../data/exported/weights.bin"


def export_state_dict(state_dict, output_path):

    with open(output_path, "wb") as f:
        
        f.write(b"RES8")  
        f.write(struct.pack("<I", 1))  
        f.write(struct.pack("<I", len(state_dict)))

        for name, tensor in state_dict.items():

            print(name, " ", tensor)

            tensor = tensor.detach().cpu().contiguous().float()
            name_bytes = name.encode("utf-8")

            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            shape = list(tensor.shape)
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<I", dim))

            f.write(tensor.numpy().tobytes())

            print(f"Exported {name:40s} shape={shape}")


if __name__ == "__main__":
    import os
    os.makedirs("../data/exported", exist_ok=True)

    model = Res8(n_blocks=3)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    export_state_dict(model.state_dict(), OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")