import pytest
import httpx
from env.environment import InventoryEnv
from api.server import app
from fastapi.testclient import TestClient

def test_env_reset():
    env = InventoryEnv()
    obs = env.reset()
    assert isinstance(obs, dict)
    assert "item_states" in obs
    assert len(obs["item_states"]) == 10
    assert obs["steps_elapsed"] == 0

def test_env_step():
    env = InventoryEnv()
    env.reset()
    # Step with action 5 (hold)
    obs, reward, done, info = env.step(5)
    assert obs["steps_elapsed"] == 1
    assert isinstance(reward, float)
    assert "orders_completed" in info

def test_env_action_restock():
    env = InventoryEnv()
    env.reset()
    # Action 1 is restock
    obs, reward, done, info = env.step(1)
    # Check that restock was executed (budget decreases from 1000)
    assert obs["budget_remaining"] < 1000.0

def test_api_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_api_reset():
    client = TestClient(app)
    response = client.post("/reset")
    assert response.status_code == 200
    assert "state" in response.json()
    assert "budget_remaining" in response.json()["state"]

def test_api_step():
    client = TestClient(app)
    # Reset first
    client.post("/reset")
    # Action 5 (hold)
    response = client.post("/step", json={"action": 5})
    assert response.status_code == 200
    assert "reward" in response.json()
    assert "state" in response.json()
    assert "info" in response.json()

def test_api_grader_endpoint():
    client = TestClient(app)
    response = client.post("/grade")
    assert response.status_code == 200
    assert "grader_output" in response.json()
