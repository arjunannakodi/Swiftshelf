from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any
import subprocess
import json

from env.environment import InventoryEnv
from tasks import TASKS

app = FastAPI(title="SwiftShelf++ Inventory Decision API")

# Initialize global environment
env = InventoryEnv()

class ActionRequest(BaseModel):
    action: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    return {"state": env.reset()}

@app.get("/state")
def get_state():
    return {"state": env.state()}

@app.post("/step")
async def step_endpoint(request: Request):
    try:
        # Load body
        body = await request.json()
        action = int(body.get("action", 5))

        # Call env.step(action)
        state, reward, done, info = env.step(action)

        # Build response
        return {
            "state": state,
            "reward": float(reward),
            "done": bool(done),
            "info": {
                "orders_completed": int(info.get("orders_completed", 0)),
                "expired_count": int(info.get("expired_count", 0))
            }
        }

    except Exception as e:
        # Fallback response
        return {
            "state": {},
            "reward": -50.0,
            "done": True,
            "info": {
                "orders_completed": 0,
                "expired_count": 0,
                "error": str(e)
            }
        }

@app.get("/tasks")
def tasks():
    return [task["name"] for task in TASKS.values()]

@app.post("/grade")
def grade():
    try:
        result = subprocess.run(["python", "grader.py"], capture_output=True, text=True)
        return {"status": "success", "grader_output": result.stdout}
    except Exception as e:
        return {"status": "error", "message": f"Failed to run grader: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
