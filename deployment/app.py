from flask import Flask, request, jsonify
import torch
from src.model import GNNModel

app = Flask(__name__)

device = torch.device("cpu")
model = GNNModel(input_dim=11, hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load("gnn_model.pth", map_location=device))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    graph_data = torch.tensor(data['graph'])
    prediction = model(graph_data)
    return jsonify({'prediction': prediction.item()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
