import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm
from sklearn.metrics import f1_score
from torch.nn import Linear
    
class GraphDataset(Dataset):
    """Custom dataset for loading graph data into a PyTorch-Geometric format."""
    def __init__(self, data_frame, root=None, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data_frame = data_frame

    def len(self):
        return len(self.data_frame)

    def get(self, idx):
        row = self.data_frame.iloc[idx]
        node_labels = np.nan_to_num(row['node_labels'], nan=-1).astype(int)
        edge_index = row['edge_index'].clone().detach()
        edge_weight = row['edge_weights'].clone().detach()
        node_features = torch.tensor(np.vstack(row['node_features']), dtype=torch.float)
        train_mask = torch.tensor(row['train_mask'], dtype=torch.bool)
        test_mask = torch.tensor(row['test_mask'], dtype=torch.bool)

        return Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight, y=torch.tensor(node_labels, dtype=torch.long),
                    train_mask=train_mask, test_mask=test_mask)

class FluxGAT(torch.nn.Module):
    """FluxGAT model for essentiality prediction."""
    def __init__(self, num_features, embedding_dim=150, hidden_channels=150, heads=2, num_layers=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_features, embedding_dim)
        self.conv1 = GATConv(embedding_dim, hidden_channels, heads=heads, dropout=0.2, concat=True)
        self.ln1 = LayerNorm(hidden_channels * heads)
        self.hidden_layers = torch.nn.ModuleList(
            [GATConv(heads * hidden_channels, hidden_channels, heads=heads, dropout=0.5, concat=True) for _ in range(num_layers - 1)]
        )
        self.fc = Linear(heads * hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.embedding(torch.argmax(x, dim=1))
        x = F.leaky_relu(self.ln1(self.conv1(x, edge_index, edge_weight)))
        x = F.dropout(x, p=0.2, training=self.training)
        for layer in self.hidden_layers:
            x = F.dropout(F.leaky_relu(layer(x, edge_index, edge_weight)), p=0.5, training=self.training)
        return self.fc(x).squeeze(-1)
    
def train(model, data, optimizer, criterion):
    """
    Trains the model on the training nodes.

    Parameters:
    model (torch.nn.Module): The model to train.
    data (Data): The graph data containing features, labels, and masks.
    optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
    criterion (torch.nn.modules.loss._Loss): The loss function.

    Returns:
    tuple: A tuple containing the loss value and accuracy of the model on the training set.
    """
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].float())
    loss.backward()
    optimizer.step()

    pred = torch.sigmoid(out[data.train_mask]) > 0.65
    acc = pred.eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()

    return loss.item(), acc

def test(model, data):
    """
    Tests the model on the testing dataset.

    Parameters:
    model (torch.nn.Module): The trained model.
    data (Data): The graph data containing features, labels, and masks.

    Returns:
    tuple: A tuple containing accuracy, F1 score, and probabilities of the test set.
    """
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = torch.sigmoid(out[data.test_mask]) > 0.65
        true = data.y[data.test_mask]

        acc = pred.eq(true).sum().item() / data.test_mask.sum().item()
        f1 = f1_score(true.cpu(), pred.cpu())
        probabilities = torch.sigmoid(out[data.test_mask]).cpu().numpy()

    return acc, f1, probabilities, true.cpu(), pred.cpu()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_pickle('../data/k_fold_masks.pkl')
    dataset = GraphDataset(df)

    num_repeats = 5
    num_epochs = 100

    for repeat in range(num_repeats):
        print(f'Repeat {repeat + 1}/{num_repeats}')

        fold_metrics = []

        for fold, data in enumerate(dataset):
            print(f'  Fold {fold + 1}:')
            data = data.to(device)
            model = FluxGAT(num_features=data.x.size(1), hidden_channels=150, embedding_dim=300, heads=2, num_layers=2).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
            
            number_of_zeros = (data.y == 0).sum()
            number_of_ones = (data.y == 1).sum()
            weight = torch.tensor([number_of_zeros / number_of_ones])
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

            for epoch in range(num_epochs):
                train_loss, train_acc = train(model, data, optimizer, criterion)
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            
            test_acc, test_f1, probabilities, true, pred = test(model, data)
            print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
            print('--------------------------------------------------')

            fold_metrics.append((test_acc, test_f1))

        avg_metrics = np.mean(fold_metrics, axis=0)
        print(f'Average Test Accuracy: {avg_metrics[0]:.4f}, Average Test F1: {avg_metrics[1]:.4f}')

if __name__ == '__main__':
    main()