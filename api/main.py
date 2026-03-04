import os
import sys
import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
from dotenv import load_dotenv
import google.generativeai as genai
import torch.nn.functional as F

# Add the root directory to the Python path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FraudSAGE

# --- Setup ---
load_dotenv()
app = FastAPI(title="Enterprise Fraud Ring Detection API")

# Globals
driver = None
features_store = {}
gnn_model = None
device = torch.device('cpu')

class PredictionResponse(BaseModel):
    transaction_id: int
    prediction: str
    confidence: float
    nodes_in_subgraph: int
    suspicious_activity_report: str

@app.on_event("startup")
def startup_event():
    global driver, features_store, gnn_model
    
    print("Initializing Database Connection...")
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"), 
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    
    print("Initializing LLM...")
    genai.configure(api_key=os.getenv("LLM_API_KEY"))
    
    print("Loading GraphSAGE model into memory...")
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_fraud_sage.pt')
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model weights not found at {model_path}")

    gnn_model = FraudSAGE(num_node_features=165, hidden_channels=64, num_classes=2)
    gnn_model.load_state_dict(torch.load(model_path, map_location=device))
    gnn_model.to(device)
    gnn_model.eval()

    print("Loading Feature Store (Local Cache)...")
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "elliptic_txs_features.csv")
    if not os.path.exists(csv_path):
        raise RuntimeError(f"Feature CSV not found at {csv_path}")
        
    df = pd.read_csv(csv_path, header=None)
    for row in df.itertuples(index=False):
        features_store[int(row[0])] = list(row[2:])
        
    print("API Ready to receive traffic!")

@app.on_event("shutdown")
def shutdown_event():
    if driver:
        driver.close()

def get_subgraph_edges(tx_id: int):
    """Queries Neo4j for the 2-hop neighborhood edges."""
    query = """
    MATCH (target:Transaction {txId: $tx_id})
    OPTIONAL MATCH (target)-[*1..2]-(m:Transaction)
    WITH target, collect(DISTINCT m) AS neighbors
    WITH [target] + neighbors AS nodes
    UNWIND nodes AS n1
    MATCH (n1)-[:PAYS]->(n2:Transaction)
    WHERE n2 IN nodes
    RETURN DISTINCT n1.txId AS source, n2.txId AS target
    """
    with driver.session() as session:
        result = session.run(query, tx_id=tx_id)
        return [{"source": record["source"], "target": record["target"]} for record in result]

@app.get("/health")
def health_check():
    """Restored health check for Docker container management."""
    return {
        "status": "healthy", 
        "model_loaded": gnn_model is not None,
        "database_connected": driver is not None
    }

@app.get("/predict/{tx_id}", response_model=PredictionResponse)
def predict_fraud(tx_id: int):
    if gnn_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    if tx_id not in features_store:
        raise HTTPException(status_code=404, detail="Transaction ID not found in the Feature Store.")

    try:
        # 1. Fetch Topology from Neo4j
        edges = get_subgraph_edges(tx_id)
        
        unique_nodes = list(set([tx_id] + [e["source"] for e in edges] + [e["target"] for e in edges]))
        node_mapping = {nid: i for i, nid in enumerate(unique_nodes)}
        target_idx = node_mapping[tx_id]

        # 2. Build PyTorch Tensors
        if len(edges) > 0:
            edge_index = torch.tensor([
                [node_mapping[e["source"]] for e in edges],
                [node_mapping[e["target"]] for e in edges]
            ], dtype=torch.long).to(device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

        x = torch.tensor([features_store[nid] for nid in unique_nodes], dtype=torch.float).to(device)

        # 3. Model Inference
        with torch.no_grad():
            logits = gnn_model(x, edge_index)
            probabilities = F.softmax(logits, dim=1)
            fraud_prob = probabilities[target_idx][1].item()
            
        predicted_class = "Illicit" if fraud_prob > 0.5 else "Licit"
        confidence = fraud_prob * 100 if predicted_class == "Illicit" else (1 - fraud_prob) * 100

        # 4. Generate Narrative with Gemini
        llm = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
        SYSTEM: You are a senior Anti-Money Laundering (AML) investigator. 
        Write a brief, highly professional 3-sentence Suspicious Activity Report explaining this AI prediction to a bank manager.

        DATA:
        - Target Transaction ID: {tx_id}
        - AI Prediction: {predicted_class} ({confidence:.1f}% confidence)
        - Network Context: This transaction is connected to a local graph of {len(unique_nodes)} wallets and {edge_index.shape[1]} transaction routes within 2 hops.
        
        TASK: Write the summary. If Illicit, mention the complex network topology is indicative of 'layering' or 'peeling chains'. If Licit, state the network neighborhood appears compliant and standard.
        """
        
        report = llm.generate_content(prompt).text

        return PredictionResponse(
            transaction_id=tx_id,
            prediction=predicted_class,
            confidence=round(confidence, 2),
            nodes_in_subgraph=len(unique_nodes),
            suspicious_activity_report=report.strip()
        )
        
    except Exception as e:
        # Restored try/except block to catch Neo4j/LLM timeouts or tensor errors
        raise HTTPException(status_code=400, detail=str(e))