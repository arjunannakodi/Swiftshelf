import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import grader
from env.environment import InventoryEnv
from models import InventoryObservation, InventoryState
from tasks import TASKS

# ------------------------------------------------------------------ #
# Keepalive lifespan
# ------------------------------------------------------------------ #

async def keepalive_ping():
    """Ping self every 4 minutes to prevent HF Space sleep."""
    await asyncio.sleep(60)  # Wait for startup
    while True:
        try:
            async with httpx.AsyncClient() as client:
                await client.get("http://localhost:7860/health", timeout=5)
        except Exception:
            pass
        await asyncio.sleep(240)  # 4 minutes


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(keepalive_ping())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ------------------------------------------------------------------ #
# Local Pydantic models (server-side request/response shapes)
# ------------------------------------------------------------------ #

class Action(BaseModel):
    action: int   # 0-5

class ResetResponse(BaseModel):
    observation: InventoryObservation
    info: Dict[str, Any]

class StepResponse(BaseModel):
    observation: InventoryObservation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

class MetadataResponse(BaseModel):
    name: str
    description: str
    version: str

class SchemaResponse(BaseModel):
    action: Dict[str, Any]
    observation: Dict[str, Any]
    state: Dict[str, Any]


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
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (shared across requests)
env = InventoryEnv()
env.reset()


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@app.get("/", response_class=HTMLResponse, summary="Root dashboard")
def root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/health", summary="Health check")
def health() -> Dict[str, str]:
    """Returns 'healthy' status per OpenEnv spec."""
    return {"status": "healthy", "service": "swiftshelf-plus-plus"}


@app.get("/metadata", response_model=MetadataResponse, summary="Get environment metadata")
def get_metadata():
    return {
        "name": "SwiftShelf++",
        "description": "High-fidelity logistics simulation for FEFO inventory management.",
        "version": "1.0.0"
    }


@app.get("/schema", response_model=SchemaResponse, summary="Get environment schemas")
def get_schema():
    return {
        "action": Action.model_json_schema(),
        "observation": InventoryObservation.model_json_schema(),
        "state": InventoryState.model_json_schema(),
    }


@app.post("/reset", response_model=ResetResponse, summary="Reset environment")
def reset() -> Dict[str, Any]:
    obs, info = env.reset()
    return {"observation": obs, "info": info}


@app.get("/state", response_model=InventoryState, summary="Get current episode state")
def get_state() -> InventoryState:
    """Returns full current episode state."""
    obs_dict = env.observe()
    obs = InventoryObservation(**obs_dict)
    return InventoryState(
        observation=obs,
        total_reward=getattr(env, "_total_reward", 0.0),
        step=getattr(env, "_step_count", 0),
        done=False,
    )


@app.post("/step", response_model=StepResponse, summary="Execute one action step")
def step(request: Action) -> Dict[str, Any]:
    obs, reward, terminated, truncated, info = env.step(request.action)
    return {
        "observation": obs,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
    }


@app.post("/mcp", summary="MCP JSON-RPC Interface Placeholder")
async def mcp_endpoint(request: Request) -> Dict[str, Any]:
    """JSON-RPC 2.0 stub for OpenEnv validator compliance."""
    body = await request.json()
    return {"jsonrpc": "2.0", "id": body.get("id"), "result": {"tools": []}}


@app.get("/tasks", summary="List available evaluation tasks")
def get_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "id": task_id,
            "name": task_data["name"],
            "description": task_data.get("description", ""),
        }
        for task_id, task_data in TASKS.items()
    ]


@app.post("/grade", summary="Run heuristic grader")
def grade_endpoint() -> Dict[str, Any]:
    avg_reward = grader.run_episodes(num_episodes=3, max_steps=200)
    reward_threshold = -500.0
    status = "PASS" if avg_reward >= reward_threshold else "FAIL"
    return {
        "avg_reward": round(float(avg_reward), 4),
        "status": status,
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
