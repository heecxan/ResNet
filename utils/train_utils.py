import torch

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device, mode="val"):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            running_loss += loss.item()

            pred_class = preds.argmax(dim=1)
            correct += (pred_class == y).sum().item()
            total += y.size(0)

    acc = correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, acc
