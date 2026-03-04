from fastapi.testclient import TestClient
import sys
import os

# Ensure Python can find the api folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.main import app

client = TestClient(app)

def test_health_check():
    """Test that the API boots up and the health endpoint is reachable."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    # We don't check database/model loaded here in CI because the GitHub server 
    # won't have your massive weights file or Neo4j credentials during the basic test.