import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Res8
from dataset import SpeechCommandsDataset, LABELS

EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 32
CHECKPOINT_DIR = "../data/checkpoints"

device = torch.device("cuda")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train_dataset = SpeechCommandsDataset(root="../data/raw/speech_commands_v0.02", split="train")
val_dataset = SpeechCommandsDataset(root="../data/raw/speech_commands_v0.02", split="val")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=8, pin_memory=True, persistent_workers=True
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=8, pin_memory=True, persistent_workers=True
)

model = Res8(n_blocks=3).to(device)
optimizer = optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

best_val_acc = 0.0
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        waveforms, labels = waveforms.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * waveforms.size(0)

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
    print(f"Epoch {epoch + 1}/{EPOCHS} | "
          f"train_loss={train_loss / len(train_dataset):.4f} | "
          f"val_loss={val_loss / total:.4f} | val_acc={val_acc:.4f}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last_model.pth"))

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        print(f"  -> new best model (val_acc={val_acc:.4f}), saved")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs, stopping early "
                  f"at epoch {epoch + 1}")
            break

print(f"Training finished. Best val_acc={best_val_acc:.4f}")