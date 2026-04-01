from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List

import grader
from env.environment import InventoryEnv
from tasks import TASKS

# ------------------------------------------------------------------ #
# App Setup
# ------------------------------------------------------------------ #
app = FastAPI(
    title="SwiftShelf++ Inventory Decision API",
    description=(
        "OpenEnv-compliant FastAPI server for the SwiftShelf++ "
        "logistics RL environment. Meta PyTorch Hackathon submission."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (shared across requests)
env = InventoryEnv()


# ------------------------------------------------------------------ #
# Request / Response Models
# ------------------------------------------------------------------ #
class StepRequest(BaseModel):
    action: int = Field(..., ge=0, le=5, description="Action index in [0, 5]")


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@app.get("/health", summary="Health check")
def health() -> Dict[str, str]:
    return {"status": "ok", "env": "SwiftShelf++ v1.0"}


@app.post("/reset", summary="Reset environment")
def reset() -> Dict[str, Any]:
    obs, info = env.reset()
    return {"observation": obs, "info": info}


@app.get("/state", summary="Current observation")
def get_state() -> Dict[str, Any]:
    return env._get_obs()


@app.post("/step", summary="Execute one action step")
def step(request: StepRequest) -> Dict[str, Any]:
    # Pydantic ge/le already validates range, but double-check:
    if not (0 <= request.action <= 5):
        raise HTTPException(
            status_code=422,
            detail=f"Action {request.action} must be in range(6): 0–5.",
        )
    obs, reward, terminated, truncated, info = env.step(request.action)
    return {
        "observation": obs,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
    }


@app.get("/tasks", summary="List available evaluation tasks")
def get_tasks() -> List[Dict[str, Any]]:
    """
    Returns task metadata (name + description).
    Does NOT serialise Python class objects.
    """
    return [
        {
            "id": task_id,
            "name": task_data["name"],
            "description": task_data.get("description", ""),
        }
        for task_id, task_data in TASKS.items()
    ]


@app.post("/grade", summary="Run heuristic grader — 3 episodes of 200 steps")
def grade_endpoint() -> Dict[str, Any]:
    avg_reward = grader.run_episodes(num_episodes=3, max_steps=200)
    reward_threshold = -500.0
    status = "PASS" if avg_reward >= reward_threshold else "FAIL"
    return {
        "avg_reward": round(float(avg_reward), 4),
        "threshold": reward_threshold,
        "status": status,
    }


# ------------------------------------------------------------------ #
# Dev entrypoint
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
