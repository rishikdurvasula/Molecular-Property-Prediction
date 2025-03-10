import torch
from model import GNNModel

device = torch.device("cpu")
model = GNNModel(input_dim=11, hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load("gnn_model.pth", map_location=device))
model.eval()

def predict(graph_data):
    with torch.no_grad():
        prediction = model(graph_data)
    return prediction.item()
