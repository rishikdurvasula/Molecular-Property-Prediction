import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import get_dataloaders
from model import GNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, _ = get_dataloaders()
model = GNNModel(input_dim=11, hidden_dim=64, output_dim=1).to(device)  # QM9 has 11 input features
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y[:, 0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y[:, 0])
            total_loss += loss.item()
    return total_loss / len(val_loader)

if __name__ == "__main__":
    for epoch in range(20):
        train_loss = train()
        val_loss = validate()
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "gnn_model.pth")
    print("Model saved!")
