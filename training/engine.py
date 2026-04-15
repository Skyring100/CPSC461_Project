import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device, process):
    """Run one full training epoch and return (avg_loss, peak_ram_gb, peak_vram_gb)."""
    model.train()
    running_loss = 0.0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.float().to(device), y.squeeze().long().to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Capture peak memory reached during this epoch
    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device) / 1e9

    peak_ram = process.memory_info().rss / 1e9

    return running_loss / len(loader), peak_ram, peak_vram


def validate(model, loader, device) -> float:
    """Return accuracy (%) on loader using the current model weights."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.float().to(device), y.view(-1).long().to(device)
            _, predicted = torch.max(model(x), dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100.0 * correct / total