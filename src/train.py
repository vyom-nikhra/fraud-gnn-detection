import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import mlflow
from torch_geometric.data import Data
from sklearn.metrics import classification_report
from model import FraudSAGE  # Importing your architecture from model.py

def set_seed(seed=42):
    """Locks all random seeds for reproducible MLOps tracking."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_data():
    print("Loading data from data/raw/...")
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    
    df_features = pd.read_csv(os.path.join(base_dir, 'elliptic_txs_features.csv'), header=None)
    df_edges = pd.read_csv(os.path.join(base_dir, 'elliptic_txs_edgelist.csv'))
    df_classes = pd.read_csv(os.path.join(base_dir, 'elliptic_txs_classes.csv'))

    class_mapping = {'1': 1, '2': 0, 'unknown': -1}
    df_classes['label'] = df_classes['class'].map(class_mapping)
    node_mapping = {tx_id: index for index, tx_id in enumerate(df_features[0].values)}

    df_edges['source'] = df_edges['txId1'].map(node_mapping)
    df_edges['target'] = df_edges['txId2'].map(node_mapping)
    df_edges = df_edges.dropna(subset=['source', 'target'])
    
    # Cast edge lists to integers to prevent tensor shape/type errors
    df_edges['source'] = df_edges['source'].astype(int)
    df_edges['target'] = df_edges['target'].astype(int)

    x = torch.tensor(df_features.drop(columns=[0, 1]).values, dtype=torch.float)
    y = torch.tensor(df_classes['label'].values, dtype=torch.long)
    edge_index = torch.tensor(df_edges[['source', 'target']].values.T, dtype=torch.long)

    time_steps = df_features[1].values
    train_mask = torch.tensor((time_steps <= 34) & (y.numpy() != -1), dtype=torch.bool)
    test_mask = torch.tensor((time_steps > 34) & (y.numpy() != -1), dtype=torch.bool)

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

def main():
    # 1. Lock the seed
    set_seed(42)
    
    # 2. Set up MLflow
    mlflow.set_experiment("Fraud_Ring_Detection")
    
    # Define hyperparameters for this run
    params = {
        "hidden_channels": 64,
        "learning_rate": 0.01,
        "weight_decay": 5e-4,
        "illicit_weight": 12.0,
        "epochs": 100 # Kept shorter for local testing
    }

    device = torch.device('cpu') # Defaulting to CPU for local testing
    data = load_data().to(device)
    
    model = FraudSAGE(num_node_features=165, hidden_channels=params["hidden_channels"], num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    
    class_weights = torch.tensor([1.0, params["illicit_weight"]], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    with mlflow.start_run():
        mlflow.log_params(params)
        print("Starting training...")
        
        best_f1 = 0
        best_recall = 0
        
        for epoch in range(1, params["epochs"] + 1):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            mlflow.log_metric("train_loss", loss.item(), step=epoch)
            
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    pred = model(data.x, data.edge_index).argmax(dim=1)
                    test_preds = pred[data.test_mask].numpy()
                    test_labels = data.y[data.test_mask].numpy()
                    
                    report = classification_report(test_labels, test_preds, output_dict=True, zero_division=0)
                    illicit_f1 = report['1']['f1-score']
                    illicit_rec = report['1']['recall']
                    
                    mlflow.log_metric("val_f1_illicit", illicit_f1, step=epoch)
                    mlflow.log_metric("val_recall", illicit_rec, step=epoch)
                    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Illicit F1: {illicit_f1:.4f} | Recall: {illicit_rec:.4f}")
                    
                    if illicit_f1 > best_f1:
                        best_f1 = illicit_f1
                        best_recall = illicit_rec
                        torch.save(model.state_dict(), os.path.join('models', 'best_fraud_sage.pt'))
                        
        mlflow.log_metric("best_val_f1", best_f1)
        mlflow.log_metric("best_recall", best_recall)
        mlflow.log_artifact(os.path.join('models', 'best_fraud_sage.pt'))
        print(f"Training complete. Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()