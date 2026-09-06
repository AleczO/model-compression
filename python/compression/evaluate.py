from model import Res8
from dataset import SpeechCommandsDataset

from compression.model_io import import_state_dict

from torch.utils.data import DataLoader
import torch



BATCH_SIZE = 32

device = torch.device("cuda")
criterion = torch.nn.CrossEntropyLoss()

def evaluate(model):
    val_dataset = SpeechCommandsDataset(root="../data/raw/speech_commands_v0.02", split="val")

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=8, pin_memory=True, persistent_workers=True
    )

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms, labels = waveforms.to(device), labels.to(device)
                logits = model(waveforms)
                loss = criterion(logits, labels)
    
                val_loss += loss.item() * waveforms.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

    val_acc = correct / total
    print(val_acc * 100)


if __name__ == "__main__":
    model = Res8(n_blocks=3)
    model.to(device)
    import_state_dict(model)

    evaluate(model)