import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm

# --- Setup & Auth ---
load_dotenv()
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

# --- 1. Load Data ---
print("Loading local CSVs...")
df_features = pd.read_csv(os.path.join(base_dir, 'elliptic_txs_features.csv'), header=None)
df_classes = pd.read_csv(os.path.join(base_dir, 'elliptic_txs_classes.csv'))
df_edges = pd.read_csv(os.path.join(base_dir, 'elliptic_txs_edgelist.csv'))

# --- 2. Process Nodes ---
df_features.rename(columns={0: 'txId', 1: 'time_step'}, inplace=True)
df_nodes = pd.merge(df_features[['txId', 'time_step']], df_classes, on='txId')

class_mapping = {'1': 1, '2': 0, 'unknown': -1}
df_nodes['label'] = df_nodes['class'].map(class_mapping)

# Filter: Keep only first 30 time steps to stay under AuraDB Free limits
df_nodes = df_nodes[df_nodes['time_step'] <= 30]

# --- 3. Process Edges ---
valid_txids = set(df_nodes['txId'])
df_edges = df_edges[df_edges['txId1'].isin(valid_txids) & df_edges['txId2'].isin(valid_txids)]

# --- 4. Database Helper ---
def execute_batch(session, query, data_list, desc, batch_size=5000):
    """Runs a Cypher query in chunks to prevent memory overload."""
    for i in tqdm(range(0, len(data_list), batch_size), desc=desc):
        session.run(query, batch=data_list[i:i+batch_size])

# --- 5. Main Execution ---
if __name__ == "__main__":
    print(f"Connecting to Neo4j at {URI}...")
    driver = GraphDatabase.driver(URI, auth=AUTH)

    try:
        with driver.session() as session:
            # Create indexing constraint for fast insertion
            print("Setting up database constraints...")
            session.run("CREATE CONSTRAINT tx_id_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.txId IS UNIQUE")

            # Prepare and insert nodes
            print(f"Ingesting {len(df_nodes)} nodes...")
            nodes_list = df_nodes[['txId', 'label', 'time_step']].to_dict('records')
            node_query = """
            UNWIND $batch AS row
            MERGE (t:Transaction {txId: row.txId})
            SET t.label = row.label, t.time_step = row.time_step
            """
            execute_batch(session, node_query, nodes_list, "Nodes")

            # Prepare and insert edges
            print(f"Ingesting {len(df_edges)} edges...")
            edges_list = df_edges[['txId1', 'txId2']].to_dict('records')
            edge_query = """
            UNWIND $batch AS row
            MATCH (src:Transaction {txId: row.txId1})
            MATCH (dst:Transaction {txId: row.txId2})
            MERGE (src)-[:PAYS]->(dst)
            """
            execute_batch(session, edge_query, edges_list, "Edges")

        print("Ingestion Complete! Your graph database is now live.")
    except Exception as e:
        print(f"Error during ingestion: {e}")
    finally:
        driver.close()