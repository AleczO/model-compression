import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Res8
from dataset import SpeechCommandsDataset, LABELS

EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 32
CHECKPOINT_DIR = "../data/checkpoints"

SOURCE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
BEST_TUNED_PATH = os.path.join(CHECKPOINT_DIR, "best_tuned_model.pth")
LAST_TUNED_PATH = os.path.join(CHECKPOINT_DIR, "last_tuned_model.pth")

device = torch.device("cuda")

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
criterion = torch.nn.CrossEntropyLoss()

model.load_state_dict(torch.load(SOURCE_MODEL_PATH, map_location=device))
print(f"Loaded weights from {SOURCE_MODEL_PATH}")


def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            logits = model(waveforms)
            loss = criterion(logits, labels)

            total_loss += loss.item() * waveforms.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


start_val_loss, best_val_acc = evaluate(val_loader)
print(f"Starting point: val_loss={start_val_loss:.4f} | val_acc={best_val_acc:.4f}")

optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

epochs_without_improvement = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for waveforms, labels in tqdm(train_loader, desc=f"Tune epoch {epoch + 1}"):
        waveforms, labels = waveforms.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(waveforms)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * waveforms.size(0)

    val_loss, val_acc = evaluate(val_loader)
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Tune epoch {epoch + 1}/{EPOCHS} | "
          f"train_loss={train_loss / len(train_dataset):.4f} | "
          f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
          f"lr={current_lr:.2e}")

    scheduler.step(val_acc)

    torch.save(model.state_dict(), LAST_TUNED_PATH)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
        torch.save(model.state_dict(), BEST_TUNED_PATH)
        print(f"  -> new best tuned model (val_acc={val_acc:.4f}), saved")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs, stopping early "
                  f"at epoch {epoch + 1}")
            break
