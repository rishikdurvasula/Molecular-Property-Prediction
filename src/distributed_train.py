import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import get_dataloaders
from model import GNNModel

def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

setup_distributed()

device = torch.device(f"cuda:{dist.get_rank()}")
train_loader, val_loader, _ = get_dataloaders()
model = GNNModel(input_dim=11, hidden_dim=64, output_dim=1).to(device)
model = DDP(model, device_ids=[dist.get_rank()])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

if __name__ == "__main__":
    for epoch in range(20):
        train_loss = train()
        val_loss = validate()
        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "gnn_model_distributed.pth")
        print("Distributed Model saved!")
