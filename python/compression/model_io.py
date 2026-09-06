import struct
import torch
import math
from model import Res8

import os
os.makedirs("../data/exported", exist_ok=True)

CHECKPOINT_PATH = "../data/checkpoints/best_model.pth"
OUTPUT_PATH = "../data/exported/weights.bin"
INPUT_PATH = "../data/exported/reconstructed_weights.bin"


def import_state_dict(model, input_path="../data/exported/reconstructed_weights.bin"):

    net_dict = dict()

    with open(input_path, "rb") as f:
        arch = struct.unpack("<4s", f.read(4))[0].decode("UTF-8")
        assert(arch == 'RES8')

        version = struct.unpack("<I", f.read(4))[0]
        assert(version == 1)

        net_size = struct.unpack("<I", f.read(4))[0]

        for i in range(net_size):
            name_len = struct.unpack("<I", f.read(4))[0]
            name_string = "<" + str(name_len) + "s"
            name = struct.unpack(name_string, f.read(name_len))[0].decode("UTF-8")

            ndims = struct.unpack("<I", f.read(4))[0]
            ndims_string = "<" + str(ndims) + "I"
            dims = struct.unpack(ndims_string, f.read(ndims * struct.calcsize("I")))

            data_len = math.prod(dims)
            data_string = "<" + str(data_len) + "f"
            data = struct.unpack(data_string, f.read(data_len * struct.calcsize("f")))

            net_dict[name] = torch.Tensor(data).reshape(dims)


    return model.load_state_dict(net_dict)


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

    model = Res8(n_blocks=3)

    import_state_dict(model)
    # model.load_state_dict()
    # model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    # model.eval()

    # export_state_dict(model.state_dict(), OUTPUT_PATH)
    # print(f"\nSaved to {OUTPUT_PATH}")